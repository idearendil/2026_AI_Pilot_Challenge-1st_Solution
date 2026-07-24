# -*- coding: utf-8 -*-
"""REDQ Ray rollout worker (Phase 1).

PPO 의 RolloutWorker(parallel.py)와 같은 역할이지만 두 가지가 다르다:
  1. 정책이 MLPDiscreteActor(가치망 없음)다. rollout action 은 actor.act_stochastic.
  2. GAE 를 계산하지 않고 **raw transition** (obs, next_obs, action_idx, reward,
     terminated, opp_slot)을 그대로 driver 로 보낸다. driver 가 중앙 replay 에 넣는다.

opponent pool 배관(BT slot0 고정 + snapshot 후보)은 기존 SelfPlayProvider /
PoolSelfPlayProvider / make_bt_provider 를 그대로 재사용한다. 상대 snapshot 은
MLPDiscreteActor frozen copy 로 만든다.

**gen 태깅**: 매 step 활성 opponent 의 pool slot index(_pool_provider.current)를 기록한다.
driver 가 이 slot→gen 을 자기 pool 메타데이터로 매핑한다(한 collect 청크 동안 pool 은
driver 가 고정하므로 slot index 의미가 일관됨).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.model import MLPDiscreteActor, discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd
from claude_code.ppo import rollout_outcome, OBS_CLIP


def make_redq_worker_cls():
    """Ray import 시점에 actor 클래스를 정의(미설치 환경 import 안전)."""
    import ray

    @ray.remote
    class RedqRolloutWorker:
        def __init__(self, worker_id, env_kwargs, actor_kwargs, cfg_dict, self_play, seed):
            for _p in (str(ROOT), str(ROOT / "src")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            os.chdir(str(ROOT))
            import torch as _torch
            _torch.set_num_threads(1)
            _torch.manual_seed(seed)
            np.random.seed(seed)

            from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
            self.env = make_env(runner_index=f"redqw{worker_id}", **env_kwargs)
            self._actor_kwargs = dict(actor_kwargs)
            self.actor = MLPDiscreteActor(**actor_kwargs)
            self.actor.eval()
            self.step_ratio = int(STANDARD_ENV_CONFIG["step_ratio"])

            self.obs_dim = int(actor_kwargs["obs_dim"])
            self.act_dim = int(actor_kwargs["act_dim"])
            self.num_bins = int(actor_kwargs["num_bins"])
            self.normalize_obs = cfg_dict["normalize_obs"]
            self.reconstruct = cfg_dict["reconstruct_state"]
            self.warmup_steps = int(cfg_dict["warmup_steps"])
            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if self.normalize_obs else None

            if self.reconstruct:
                from claude_code.my_observation import reset_reconstructor, advance_reconstructor
                self._reset_recon = reset_reconstructor
                self._advance_recon = advance_reconstructor
                self._reset_recon()
            else:
                self._reset_recon = self._advance_recon = None

            self._pool_provider = None
            self._bt_slots = 0
            self._pool_max = 1
            self._bt_provider = None

            if self_play:
                # pool 설치 전까지는 임시로 자기 자신(actor) self-play 로 둔다.
                from claude_code.self_play import SelfPlayProvider
                self.env._target_action_provider = SelfPlayProvider(
                    self.actor, self.obs_rms, self.env._observation_fn,
                    self.env._observation_mode, self.step_ratio, "cpu", explore=True)

            obs, _ = self.env.reset(seed=seed)
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._ep_return = 0.0
            self._ep_len = 0
            self._global_step = 0

        # ── weight / 통계 broadcast ────────────────────────────────────────────
        def set_weights(self, state_dict):
            self.actor.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})

        def set_obs_rms(self, mean, var, count):
            if self.obs_rms is not None:
                self.obs_rms.mean = np.asarray(mean, dtype=np.float64)
                self.obs_rms.var = np.asarray(var, dtype=np.float64)
                self.obs_rms.count = float(count)

        def set_global_step(self, gs):
            self._global_step = int(gs)

        # ── opponent 생성 헬퍼 (MLPDiscreteActor frozen copy) ──────────────────
        def _build_opp_provider(self, state_dict, mean, var, count):
            from claude_code.self_play import SelfPlayProvider
            m = MLPDiscreteActor(**self._actor_kwargs)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})
            m.eval()
            rms = None
            if self.obs_rms is not None:
                rms = RunningMeanStd(shape=(self.obs_dim,))
                rms.mean = np.asarray(mean, dtype=np.float64)
                rms.var = np.asarray(var, dtype=np.float64)
                rms.count = float(count)
            return SelfPlayProvider(
                m, rms, self.env._observation_fn, self.env._observation_mode,
                self.step_ratio, "cpu", explore=True)

        def _get_bt_provider(self, bt_dll, bt_rule):
            if self._bt_provider is None:
                from claude_code.self_play import make_bt_provider
                self._bt_provider = make_bt_provider(bt_dll, bt_rule)
            return self._bt_provider

        def pool_init(self, state_dict, mean, var, count, weights, pool_max, seed,
                      bt_dll="", bt_rule=""):
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            provs = []
            self._bt_slots = 0
            if bt_dll:
                provs.append(self._get_bt_provider(bt_dll, bt_rule))
                self._bt_slots = 1
            provs.append(self._build_opp_provider(state_dict, mean, var, count))
            self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        def pool_add(self, state_dict, mean, var, count):
            prov = self._build_opp_provider(state_dict, mean, var, count)
            providers = list(self._pool_provider.providers)
            providers.append(prov)
            if len(providers) > self._pool_max:
                providers.pop(int(self._bt_slots))   # 가장 오래된 snapshot 제거(BT 보존)
            self._pool_provider.set_pool(providers)

        def pool_set_weights(self, weights):
            if self._pool_provider is not None:
                self._pool_provider.set_weights(weights)

        def pool_set_all(self, state_dicts, means, vars_, counts, weights, pool_max, seed,
                         bt_dll="", bt_rule=""):
            """checkpoint 의 pool 전체를 복원(resume). state_dicts=snapshot 후보만
            (오래된→최신). bt_dll 있으면 slot0 에 BT 를 넣어 driver 인덱스와 일치."""
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            provs = []
            self._bt_slots = 0
            if bt_dll:
                provs.append(self._get_bt_provider(bt_dll, bt_rule))
                self._bt_slots = 1
            provs += [self._build_opp_provider(st, mn, vr, ct)
                      for st, mn, vr, ct in zip(state_dicts, means, vars_, counts)]
            self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        # ── rollout: n_steps 수집, raw transition + per-step opp_slot 반환 ─────────
        def collect(self, n_steps):
            obs_b = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            next_b = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            act_b = np.zeros((n_steps, self.act_dim), dtype=np.int64)
            rew_b = np.zeros(n_steps, dtype=np.float32)
            term_b = np.zeros(n_steps, dtype=np.float32)
            slot_b = np.zeros(n_steps, dtype=np.int64)

            ep_returns, ep_lengths, ep_components, ep_outcomes = [], [], [], []
            ep_opp_indices, ep_end_conditions = [], []

            for t in range(n_steps):
                raw = self._next_obs
                obs_b[t] = raw
                random_policy = self._global_step < self.warmup_steps
                if random_policy:
                    act_idx = np.random.randint(0, self.num_bins, size=self.act_dim)
                else:
                    norm = self._norm(raw)
                    with torch.no_grad():
                        act_idx = self.actor.act_stochastic(
                            torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0)
                        ).squeeze(0).numpy().astype(np.int64)
                act_b[t] = act_idx
                # 이 step 에 활성인 opponent pool slot(episode 시작 시 고정됨).
                slot_b[t] = int(getattr(self._pool_provider, "current", 0)) if self._pool_provider else 0

                cont = discrete_indices_to_continuous(act_idx, self.num_bins)
                next_obs, reward, term, trunc, info = self.env.step(cont)
                if self._advance_recon is not None:
                    self._advance_recon(self.env._ownship_state, self.env._target_state)

                next_b[t] = np.asarray(next_obs, dtype=np.float32)
                rew_b[t] = float(reward)
                term_b[t] = float(term)     # truncated(타임아웃)는 bootstrap 차단 안 함
                self._ep_return += float(reward)
                self._ep_len += 1
                self._global_step += 1
                done = bool(term or trunc)

                if done:
                    ep_returns.append(self._ep_return)
                    ep_lengths.append(self._ep_len)
                    ep_end_conditions.append(
                        str(info.get("end_condition", "")) if isinstance(info, dict) else "")
                    comp = info.get("ep_reward_components") if isinstance(info, dict) else None
                    if isinstance(comp, dict):
                        ep_components.append(dict(comp))
                    oc = rollout_outcome(info)
                    if oc is not None:
                        ep_outcomes.append(oc)
                        prov = self._pool_provider
                        ep_opp_indices.append(int(getattr(prov, "last_index", 0)) if prov else 0)
                    self._ep_return = 0.0
                    self._ep_len = 0
                    if self._reset_recon is not None:
                        self._reset_recon()
                    next_obs, _ = self.env.reset()
                self._next_obs = np.asarray(next_obs, dtype=np.float32)

            if self.obs_rms is not None:
                self.obs_rms.update(obs_b)

            return {
                "obs": obs_b, "next_obs": next_b, "actions": act_b,
                "rewards": rew_b, "terminated": term_b, "opp_slot": slot_b,
                "ep_returns": ep_returns, "ep_lengths": ep_lengths,
                "ep_components": ep_components, "ep_outcomes": ep_outcomes,
                "ep_opp_indices": ep_opp_indices, "ep_end_conditions": ep_end_conditions,
                "rms_mean": (obs_b.mean(0) if self.obs_rms is not None else None),
                "rms_var": (obs_b.var(0) if self.obs_rms is not None else None),
                "rms_count": n_steps,
            }

        def _norm(self, obs):
            if self.obs_rms is None:
                return np.asarray(obs, dtype=np.float32)
            n = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            return np.clip(n, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    return RedqRolloutWorker


__all__ = ["make_redq_worker_cls"]
