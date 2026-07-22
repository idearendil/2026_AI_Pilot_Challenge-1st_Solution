"""Ray 기반 병렬 rollout 데이터 수집.

큰 모델 + GPU 학습을 대비해, CPU-bound 한 환경 stepping 을 여러 프로세스(Ray actor)로
병렬화한다. 각 worker 는 자신의 env(self-play 포함) + 로컬 model 복사본을 갖고 rollout
을 모은다. 매 iteration:
  driver --(policy weights + obs_rms)--> workers --(rollout batch)--> driver --update(GPU 가능)

worker 수는 **물리 CPU 코어 수** 기준(논리 코어 아님)으로 정한다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.model import make_actor_critic, discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd
from claude_code.ppo import (PPOConfig, PPOTrainer, IterationStats, compute_gae, OBS_CLIP,
                             rollout_outcome, _outcome_counts, _outcome_counts_by_opp,
                             _count_altitude_terms)


def physical_cpu_count() -> int:
    """물리 CPU 코어 수 (논리 아님). psutil 없으면 OS 별 조회 → 마지막엔 logical//2."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | "
                 "Measure-Object -Property NumberOfCores -Sum).Sum"],
                text=True, timeout=15)
            n = int(out.strip())
            if n > 0:
                return n
        elif sys.platform.startswith("linux"):
            ids = set()
            with open("/proc/cpuinfo") as fh:
                phys = core = None
                for line in fh:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        if phys is not None:
                            ids.add((phys, core))
            if ids:
                return len(ids)
    except Exception:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


# ── Ray worker ────────────────────────────────────────────────────────────────

def _make_worker_cls():
    """Ray 가 import 된 시점에 actor 클래스를 정의(미설치 환경에서 import 안전)."""
    import ray

    @ray.remote
    class RolloutWorker:
        def __init__(self, worker_id, env_kwargs, model_kwargs, cfg_dict, self_play, seed):
            for _p in (str(ROOT), str(ROOT / "src")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            os.chdir(str(ROOT))   # 상대 경로 자산(aircraft/engine/xml) 로드 안전
            import torch as _torch
            _torch.manual_seed(seed)
            np.random.seed(seed)

            from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
            self.env = make_env(runner_index=f"w{worker_id}", **env_kwargs)
            self.model = make_actor_critic(**model_kwargs)
            self.model.eval()
            self._model_kwargs = dict(model_kwargs)   # 평가용 상대 network 재생성에 사용
            self._opp_model = None

            self.gamma = cfg_dict["gamma"]
            self.gae_lambda = cfg_dict["gae_lambda"]
            self.normalize_obs = cfg_dict["normalize_obs"]
            self.reconstruct = cfg_dict["reconstruct_state"]
            self.obs_dim = model_kwargs["obs_dim"]
            self.act_dim = model_kwargs["act_dim"]

            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if self.normalize_obs else None

            if self.reconstruct:
                from claude_code.my_observation import reset_reconstructor, advance_reconstructor
                self._reset_recon = reset_reconstructor
                self._advance_recon = advance_reconstructor
                self._reset_recon()
            else:
                self._reset_recon = self._advance_recon = None

            if self_play:
                from claude_code.self_play import SelfPlayProvider
                sr = int(STANDARD_ENV_CONFIG["step_ratio"])
                self.env._target_action_provider = SelfPlayProvider(
                    self.model, self.obs_rms, self.env._observation_fn,
                    self.env._observation_mode, sr, "cpu", explore=True)

            obs, _ = self.env.reset(seed=seed)
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0

        def set_weights(self, state_dict):
            self.model.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})

        def set_obs_rms(self, mean, var, count):
            if self.obs_rms is not None:
                self.obs_rms.mean = np.asarray(mean, dtype=np.float64)
                self.obs_rms.var = np.asarray(var, dtype=np.float64)
                self.obs_rms.count = float(count)

        def set_frozen_opponent(self, state_dict, rms_mean, rms_var, rms_count):
            """self-play 상대를 '초기 actor net 고정'(별도 frozen 모델)으로 교체한다.

            기본 self-play 는 상대가 self.model(매 iter 갱신되는 학습 agent)을 참조한다.
            여기서는 학습 시작 시점의 weights 로 별도 frozen 모델을 만들어 상대 provider 가
            그걸 쓰게 한다(학습이 진행돼도 상대는 고정). obs_rms 도 그 시점 값으로 고정.
            """
            from claude_code.env_utils import STANDARD_ENV_CONFIG
            from claude_code.self_play import SelfPlayProvider
            self._frozen_opp_model = make_actor_critic(**self._model_kwargs)
            self._frozen_opp_model.load_state_dict(
                {k: torch.as_tensor(v) for k, v in state_dict.items()})
            self._frozen_opp_model.eval()
            frozen_rms = None
            if self.obs_rms is not None:
                frozen_rms = RunningMeanStd(shape=(self.obs_dim,))
                frozen_rms.mean = np.asarray(rms_mean, dtype=np.float64)
                frozen_rms.var = np.asarray(rms_var, dtype=np.float64)
                frozen_rms.count = float(rms_count)
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            self.env._target_action_provider = SelfPlayProvider(
                self._frozen_opp_model, frozen_rms, self.env._observation_fn,
                self.env._observation_mode, sr, "cpu", explore=True)

        def _build_opp_provider(self, state_dict, mean, var, count, explore=True):
            """broadcast 받은 state+rms 로 frozen opponent SelfPlayProvider 생성(worker)."""
            from claude_code.env_utils import STANDARD_ENV_CONFIG
            from claude_code.self_play import SelfPlayProvider
            m = make_actor_critic(**self._model_kwargs)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})
            m.eval()
            rms = None
            if self.obs_rms is not None:
                rms = RunningMeanStd(shape=(self.obs_dim,))
                rms.mean = np.asarray(mean, dtype=np.float64)
                rms.var = np.asarray(var, dtype=np.float64)
                rms.count = float(count)
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            return SelfPlayProvider(
                m, rms, self.env._observation_fn, self.env._observation_mode,
                sr, "cpu", explore=explore)

        def pool_init(self, state_dict, mean, var, count, weights, pool_max, seed):
            """opponent pool 초기화(후보 1개 = 초기 정책). target provider 를 pool 로 교체."""
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            prov = self._build_opp_provider(state_dict, mean, var, count)
            self._pool_provider = PoolSelfPlayProvider([prov], weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        def pool_add(self, state_dict, mean, var, count):
            """현재 정책 frozen copy 를 pool 에 추가(초과 시 oldest 제거) + env 리셋."""
            prov = self._build_opp_provider(state_dict, mean, var, count)
            providers = list(self._pool_provider.providers)
            providers.append(prov)
            if len(providers) > self._pool_max:
                providers.pop(0)
            self._pool_provider.set_pool(providers)
            if self._reset_recon is not None:
                self._reset_recon()
            obs, _ = self.env.reset()
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0

        def pool_set_weights(self, weights):
            """opponent 샘플링 가중치 갱신."""
            if getattr(self, "_pool_provider", None) is not None:
                self._pool_provider.set_weights(weights)

        def pool_set_all(self, state_dicts, means, vars_, counts, weights, pool_max, seed):
            """checkpoint 의 opponent pool 전체를 복원(worker). state_dicts 는 오래된→최신."""
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            provs = [self._build_opp_provider(st, mn, vr, ct)
                     for st, mn, vr, ct in zip(state_dicts, means, vars_, counts)]
            self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        def _norm(self, obs):
            if self.obs_rms is None:
                return np.asarray(obs, dtype=np.float32)
            n = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            return np.clip(n, -OBS_CLIP, OBS_CLIP).astype(np.float32)

        def collect(self, n_steps):
            obs_buf = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            raw_buf = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            act_buf = np.zeros((n_steps, self.act_dim), dtype=np.float32)
            logp_buf = np.zeros(n_steps, dtype=np.float32)
            rew_buf = np.zeros(n_steps, dtype=np.float32)
            done_buf = np.zeros(n_steps, dtype=np.float32)
            val_buf = np.zeros(n_steps, dtype=np.float32)
            ep_returns, ep_lengths, ep_components, ep_outcomes = [], [], [], []
            ep_opp_indices = []
            ep_end_conditions = []

            for t in range(n_steps):
                raw = self._next_obs
                norm = self._norm(raw)
                raw_buf[t] = raw
                obs_buf[t] = norm
                done_buf[t] = float(self._next_done)
                with torch.no_grad():
                    a, lp, _, v = self.model.get_action_and_value(
                        torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0))
                a_np = a.squeeze(0).numpy().astype(np.float32)   # 카테고리 index (저장용)
                act_buf[t] = a_np
                logp_buf[t] = float(lp.item())
                val_buf[t] = float(v.item())

                env_action = discrete_indices_to_continuous(a_np, self.model.num_bins)
                next_obs, reward, term, trunc, info = self.env.step(env_action)
                done = bool(term or trunc)
                if self._advance_recon is not None:
                    self._advance_recon(self.env._ownship_state, self.env._target_state)
                self._ep_return += float(reward)
                self._ep_len += 1
                rew_buf[t] = float(reward)   # raw 보상 그대로 (스케일링 없음)
                if done:
                    ep_returns.append(self._ep_return)
                    ep_lengths.append(self._ep_len)
                    ep_end_conditions.append(
                        str(info.get("end_condition", "")) if isinstance(info, dict) else "")
                    comp = info.get("ep_reward_components")
                    if isinstance(comp, dict):
                        ep_components.append(dict(comp))
                    oc = rollout_outcome(info)
                    if oc is not None:
                        ep_outcomes.append(oc)
                        prov = getattr(self.env, "_target_action_provider", None)
                        ep_opp_indices.append(int(getattr(prov, "last_index", 0)))
                    self._ep_return = 0.0
                    self._ep_len = 0
                    if self._reset_recon is not None:
                        self._reset_recon()
                    next_obs, _ = self.env.reset()
                self._next_obs = np.asarray(next_obs, dtype=np.float32)
                self._next_done = done

            with torch.no_grad():
                last_v = float(self.model.get_value(
                    torch.as_tensor(self._norm(self._next_obs), dtype=torch.float32).unsqueeze(0)).item())
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, self._next_done,
                                   self.gamma, self.gae_lambda)
            return {
                "obs": obs_buf, "actions": act_buf, "logp": logp_buf,
                "advantages": adv, "returns": ret, "values": val_buf,
                "ep_returns": ep_returns, "ep_lengths": ep_lengths,
                "ep_components": ep_components, "ep_outcomes": ep_outcomes,
                "ep_opp_indices": ep_opp_indices,
                "ep_end_conditions": ep_end_conditions,
                "rms_mean": (raw_buf.mean(0) if self.obs_rms is not None else None),
                "rms_var": (raw_buf.var(0) if self.obs_rms is not None else None),
                "rms_count": n_steps,
            }

        def eval_games(self, opp_state, opp_model_kwargs, opp_rms_dict,
                       cur_mean, cur_var, seeds, stochastic):
            """현재 self.model vs 과거 snapshot(opp_*) 으로 seeds 만큼 평가.

            env 의 상대를 과거 network 로 잠시 교체하고, 끝나면 원래 self-play
            상대를 복원한다. 평가가 진행 중 episode 를 끊으므로 이후 collect 의
            연속성을 위해 env / reconstructor / episode 누적을 새로 리셋한다.
            """
            from claude_code import evaluation
            if self._opp_model is None:
                self._opp_model = make_actor_critic(**opp_model_kwargs)
            self._opp_model.load_state_dict({k: torch.as_tensor(v) for k, v in opp_state.items()})
            self._opp_model.eval()

            prev = self.env._target_action_provider
            self.env._target_action_provider = evaluation.make_opponent(
                self.env, self._opp_model, opp_rms_dict, stochastic)
            try:
                results = evaluation.play_games(
                    self.env, self.model, cur_mean, cur_var, seeds, stochastic,
                    self.reconstruct, "cpu")
            finally:
                self.env._target_action_provider = prev
                # 다음 collect 의 rollout 연속성 복구 (진행 중 episode 폐기)
                if self._reset_recon is not None:
                    self._reset_recon()
                obs, _ = self.env.reset()
                if self._reset_recon is not None:
                    self._reset_recon()
                self._next_obs = np.asarray(obs, dtype=np.float32)
                self._next_done = False
                self._ep_return = 0.0
                self._ep_len = 0
            return results

    return RolloutWorker


# ── 병렬 driver ───────────────────────────────────────────────────────────────

class ParallelPPOTrainer:
    """Ray worker 들로 rollout 을 병렬 수집하고, driver 에서 PPO update(=PPOTrainer.update)."""

    def __init__(self, env_kwargs, config: PPOConfig, num_workers: int,
                 self_play: bool, obs_dim: int, act_dim: int):
        import ray
        self.cfg = config
        self.num_workers = max(1, int(num_workers))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_step = 0

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.model = make_actor_critic(
            obs_dim, act_dim, hidden=config.hidden, activation=config.activation,
            critic_hidden=config.critic_hidden, critic_activation=config.critic_activation,
            num_bins=config.num_bins).to(config.device)
        # actor / critic 별도 optimizer (완전 분리). PPOTrainer.update 가 이 둘을 사용.
        self.actor_lr0 = config.lr
        self.critic_lr0 = config.critic_lr if config.critic_lr is not None else config.lr
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=self.actor_lr0, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=self.critic_lr0, eps=1e-5)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None

        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(config.hidden),
                            activation=config.activation, num_bins=config.num_bins,
                            critic_hidden=(tuple(config.critic_hidden) if config.critic_hidden else None),
                            critic_activation=config.critic_activation)
        cfg_dict = dict(gamma=config.gamma, gae_lambda=config.gae_lambda,
                        normalize_obs=config.normalize_obs,
                        reconstruct_state=config.reconstruct_state)

        if not ray.is_initialized():
            pythonpath = os.pathsep.join(
                [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
            ray.init(num_cpus=self.num_workers, include_dashboard=False,
                     ignore_reinit_error=True, log_to_driver=False,
                     runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        WorkerCls = _make_worker_cls()
        self.workers = [
            WorkerCls.remote(i, env_kwargs, model_kwargs, cfg_dict, self_play,
                             config.seed + 1 + i)
            for i in range(self.num_workers)
        ]
        print(f"[claude_code/PPO] Ray 병렬 수집: workers={self.num_workers} (물리 코어 기준)")

    def _broadcast(self):
        import ray
        state = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        ref = ray.put(state)
        calls = [w.set_weights.remote(ref) for w in self.workers]
        if self.obs_rms is not None:
            calls += [w.set_obs_rms.remote(self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count)
                      for w in self.workers]
        ray.get(calls)

    def collect_rollout(self):
        import ray
        self._broadcast()
        per = max(1, self.cfg.rollout_steps // self.num_workers)
        results = ray.get([w.collect.remote(per) for w in self.workers])

        device = self.cfg.device
        cat = lambda key: np.concatenate([r[key] for r in results], axis=0)
        batch = {
            "obs": torch.as_tensor(cat("obs"), device=device),
            "actions": torch.as_tensor(cat("actions"), device=device),
            "logp": torch.as_tensor(cat("logp"), device=device),
            "advantages": torch.as_tensor(cat("advantages"), device=device),
            "returns": torch.as_tensor(cat("returns"), device=device),
            "values": torch.as_tensor(cat("values"), device=device),
        }
        self.global_step += batch["obs"].shape[0]

        # obs_rms 병합 (각 worker batch 통계를 authoritative obs_rms 에 누적)
        if self.obs_rms is not None:
            for r in results:
                if r["rms_mean"] is not None:
                    self.obs_rms._update_from_moments(
                        np.asarray(r["rms_mean"]), np.asarray(r["rms_var"]), r["rms_count"])

        ep_returns = [x for r in results for x in r["ep_returns"]]
        ep_lengths = [x for r in results for x in r["ep_lengths"]]
        ep_components = [x for r in results for x in r["ep_components"]]
        ep_outcomes = [x for r in results for x in r.get("ep_outcomes", [])]
        ep_opp_indices = [x for r in results for x in r.get("ep_opp_indices", [])]
        ep_end_conditions = [x for r in results for x in r.get("ep_end_conditions", [])]
        return (batch, ep_returns, ep_lengths, ep_components, ep_outcomes,
                ep_opp_indices, ep_end_conditions)

    # 업데이트 로직은 PPOTrainer.update 재사용
    def update(self, batch):
        return PPOTrainer.update(self, batch)

    # ── past-self stochastic 평가 (멀티프로세스: worker 재사용) ───────────────
    def evaluate_vs(self, opp_state, opp_model_kwargs, opp_rms_dict,
                    n_games, stochastic, base_seed):
        """현재 정책(post-update) vs 과거 snapshot 을 worker 들에 분배해 평가."""
        import ray
        from claude_code import evaluation
        self._broadcast()   # worker 에 현재 weights + obs_rms 반영
        mean = self.obs_rms.mean if self.obs_rms is not None else None
        var = self.obs_rms.var if self.obs_rms is not None else None
        seeds = [base_seed + i for i in range(n_games)]
        chunks = [seeds[i::self.num_workers] for i in range(self.num_workers)]
        opp_ref = ray.put(opp_state)
        futs = []
        for w, ch in zip(self.workers, chunks):
            if not ch:
                continue
            futs.append(w.eval_games.remote(
                opp_ref, opp_model_kwargs, opp_rms_dict, mean, var, ch, stochastic))
        results = [r for sub in ray.get(futs) for r in sub]
        return evaluation.summarize(results)

    def set_frozen_opponent(self, state_dict, rms_mean, rms_var, rms_count):
        """모든 worker 의 self-play 상대를 '초기 actor net 고정' 으로 교체(broadcast)."""
        import ray
        ref = ray.put(state_dict)
        ray.get([w.set_frozen_opponent.remote(ref, rms_mean, rms_var, rms_count)
                 for w in self.workers])

    def _current_state_rms(self):
        state = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        mean = self.obs_rms.mean if self.obs_rms is not None else None
        var = self.obs_rms.var if self.obs_rms is not None else None
        count = self.obs_rms.count if self.obs_rms is not None else 0.0
        return state, mean, var, count

    def install_opponent_pool(self, pool_max: int = 5) -> None:
        """모든 worker 의 opponent pool 초기화(후보 1개 = 현재 정책). broadcast.

        학습 시작 시 1회 호출. worker 별로 다른 seed 를 줘서 opponent 샘플 순서가
        분산되게 한다(pool 다양성).
        """
        import ray
        self._pool_max = max(1, int(pool_max))
        state, mean, var, count = self._current_state_rms()
        ref = ray.put(state)
        ray.get([
            w.pool_init.remote(ref, mean, var, count, [1.0], self._pool_max,
                               self.cfg.seed + 101 + i)
            for i, w in enumerate(self.workers)
        ])

    def pool_add_current(self) -> None:
        """현재 정책 frozen copy 를 모든 worker 의 pool 에 추가(초과 시 oldest 제거)."""
        import ray
        state, mean, var, count = self._current_state_rms()
        ref = ray.put(state)
        ray.get([w.pool_add.remote(ref, mean, var, count) for w in self.workers])

    def pool_set_weights(self, weights) -> None:
        """모든 worker 의 opponent 샘플링 가중치 갱신(broadcast)."""
        import ray
        wl = list(weights)
        ray.get([w.pool_set_weights.remote(wl) for w in self.workers])

    def snapshot_current(self) -> dict:
        """현재 정책 weights + obs_rms 통계를 checkpoint 용 dict 로 반환(numpy)."""
        state, mean, var, count = self._current_state_rms()
        rms = None if self.obs_rms is None else {
            "mean": np.asarray(mean, dtype=np.float64),
            "var": np.asarray(var, dtype=np.float64),
            "count": float(count),
        }
        return {"state": state, "rms": rms}

    def set_opponent_pool(self, entries, weights, pool_max) -> None:
        """checkpoint 의 opponent pool 전체를 모든 worker 에 복원(broadcast)."""
        import ray
        self._pool_max = max(1, int(pool_max))
        states = [e["state"] for e in entries]
        means = [(e["rms"]["mean"] if e.get("rms") else None) for e in entries]
        vars_ = [(e["rms"]["var"] if e.get("rms") else None) for e in entries]
        counts = [(e["rms"]["count"] if e.get("rms") else 0.0) for e in entries]
        sref = ray.put(states)
        wl = list(weights)
        ray.get([
            w.pool_set_all.remote(sref, means, vars_, counts, wl, self._pool_max,
                                  self.cfg.seed + 101 + i)
            for i, w in enumerate(self.workers)
        ])

    def train(self, on_iteration=None, start_iteration: int = 1):
        import time
        history = []
        for it in range(int(start_iteration), self.cfg.total_iterations + 1):
            t0 = time.time()
            (batch, ep_returns, ep_lengths, ep_components,
             ep_outcomes, ep_opp_indices, ep_end_conditions) = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            comp_means = {}
            if ep_components:
                for key in ("pursuit", "damage", "shaping", "terminal", "safety", "step"):
                    comp_means[key] = float(np.mean([c.get(key, 0.0) for c in ep_components]))
            comp_means.update(_outcome_counts(ep_outcomes))
            comp_means["per_opp"] = _outcome_counts_by_opp(ep_opp_indices, ep_outcomes)
            comp_means["alt_term"] = _count_altitude_terms(ep_end_conditions)
            stats = IterationStats(
                iteration=it, global_step=self.global_step, mean_return=mean_ret,
                mean_length=mean_len, completed_episodes=len(ep_returns),
                policy_loss=pl, value_loss=vl, entropy=ent, approx_kl=kl,
                explained_variance=ev, elapsed_sec=time.time() - t0, extra=comp_means)
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats)
        return history

    def close(self):
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


__all__ = ["physical_cpu_count", "ParallelPPOTrainer"]
