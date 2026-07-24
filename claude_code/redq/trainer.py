# -*- coding: utf-8 -*-
"""REDQ 학습 driver.

Phase 0 (현재): **단일 프로세스** RedqTrainer — 하나의 env 로 rollout 을 모으고,
replay 에서 뽑아 UTD 배만큼 gradient step 을 밟는 discrete SAC + REDQ 루프. self-play
비정상성/버그를 배제하고 SAC 코어 자체를 검증하기 위해 scripted 상대(loiter 등)로 돈다.

Phase 1 에서 Ray worker 다중화 + opponent pool 연동을 workers.py 로 추가한다. 이 파일의
로깅/replay/update 흐름은 그대로 재사용된다.

로깅 축은 **global_step(수집 env-step)** 이다. PPO 로그와 동일 env-step 기준으로 겹쳐
비교할 수 있게 컬럼/키 이름을 맞춘다.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
from claude_code.model import discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd
from claude_code.ppo import rollout_outcome, _outcome_counts, _count_altitude_terms
from claude_code.redq.replay import OpponentTaggedReplay
from claude_code.redq.sac import RedqLearner

# scripted 상대(self-play 아님) transition 태그. BT(-1)와 구분되는 '만료 없음' gen.
SCRIPTED_GEN = -1


class RedqTrainer:
    """단일 프로세스 discrete SAC + REDQ 트레이너 (Phase 0)."""

    def __init__(self, cfg, env_kwargs: dict | None = None):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        env_kwargs = dict(env_kwargs or {})
        self.env = make_env(runner_index="redq", **env_kwargs)
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        assert self.act_dim == cfg.act_dim, f"act_dim mismatch {self.act_dim} != {cfg.act_dim}"

        self.learner = RedqLearner(cfg, self.obs_dim)
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if cfg.normalize_obs else None
        self.replay = OpponentTaggedReplay(cfg.buffer_size, self.obs_dim, self.act_dim,
                                           seed=cfg.seed)

        self.reconstruct = cfg.reconstruct_state
        if self.reconstruct:
            from claude_code.my_observation import reset_reconstructor, advance_reconstructor
            self._reset_recon = reset_reconstructor
            self._advance_recon = advance_reconstructor
            self._reset_recon()
        else:
            self._reset_recon = self._advance_recon = None

        obs, _ = self.env.reset(seed=cfg.seed)
        if self._reset_recon is not None:
            self._reset_recon()
        self._next_obs = np.asarray(obs, dtype=np.float32)

        self.global_step = 0
        self.grad_steps = 0
        self._ep_return = 0.0
        self._ep_len = 0

    # ── rollout: n_steps 수집해 replay 에 삽입, 완료 episode 통계 반환 ─────────────
    def collect(self, n_steps: int) -> dict:
        obs_b = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
        next_b = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
        act_b = np.zeros((n_steps, self.act_dim), dtype=np.int64)
        rew_b = np.zeros(n_steps, dtype=np.float32)
        term_b = np.zeros(n_steps, dtype=np.float32)
        gen_b = np.full(n_steps, SCRIPTED_GEN, dtype=np.int64)

        ep_returns, ep_lengths, ep_components, ep_outcomes, ep_ends = [], [], [], [], []
        random_policy = self.global_step < self.cfg.warmup_steps

        for t in range(n_steps):
            raw = self._next_obs
            obs_b[t] = raw
            if random_policy:
                act_idx = np.random.randint(0, self.cfg.num_bins, size=self.act_dim)
            else:
                act_idx = self.learner.act(raw, deterministic=False)
            act_b[t] = act_idx

            cont = discrete_indices_to_continuous(act_idx, self.cfg.num_bins)
            next_obs, reward, term, trunc, info = self.env.step(cont)
            if self._advance_recon is not None:
                self._advance_recon(self.env._ownship_state, self.env._target_state)

            next_b[t] = np.asarray(next_obs, dtype=np.float32)
            rew_b[t] = float(reward)
            term_b[t] = float(term)   # truncated(타임아웃)는 bootstrap 차단 안 함
            self._ep_return += float(reward)
            self._ep_len += 1
            done = bool(term or trunc)

            if done:
                ep_returns.append(self._ep_return)
                ep_lengths.append(self._ep_len)
                ep_ends.append(str(info.get("end_condition", "")) if isinstance(info, dict) else "")
                comp = info.get("ep_reward_components") if isinstance(info, dict) else None
                if isinstance(comp, dict):
                    ep_components.append(dict(comp))
                oc = rollout_outcome(info)
                if oc is not None:
                    ep_outcomes.append(oc)
                self._ep_return = 0.0
                self._ep_len = 0
                if self._reset_recon is not None:
                    self._reset_recon()
                next_obs, _ = self.env.reset()
            self._next_obs = np.asarray(next_obs, dtype=np.float32)

        # obs 정규화 통계 갱신 (raw obs 기준)
        if self.obs_rms is not None:
            self.obs_rms.update(obs_b)

        self.replay.add_batch(obs_b, next_b, act_b, rew_b, term_b, gen_b, self.global_step)
        self.global_step += n_steps

        stats = {"ep_returns": ep_returns, "ep_lengths": ep_lengths,
                 "ep_components": ep_components, "ep_outcomes": ep_outcomes,
                 "ep_ends": ep_ends}
        return stats

    # ── UTD 배 gradient update ─────────────────────────────────────────────────
    def update(self, n_env_steps: int) -> dict:
        if (len(self.replay) < self.cfg.min_buffer_for_update
                or self.global_step < self.cfg.warmup_steps):
            return {}
        if self.obs_rms is not None:
            self.learner.set_obs_stats(self.obs_rms.mean, self.obs_rms.var)

        n_updates = int(self.cfg.utd_ratio) * int(n_env_steps)
        metrics_accum: dict = {}
        for _ in range(n_updates):
            batch = self.replay.sample(self.cfg.batch_size)
            m = self.learner.update(batch)
            for k, v in m.items():
                metrics_accum.setdefault(k, []).append(v)
            self.grad_steps += 1
        return {k: float(np.mean(v)) for k, v in metrics_accum.items()}

    def purge_opponent(self, gen: int) -> int:
        return self.replay.purge_opponent(gen)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


__all__ = ["RedqTrainer", "SCRIPTED_GEN"]
