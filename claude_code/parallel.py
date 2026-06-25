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

from claude_code.model import MLPActorCritic
from claude_code.normalizers import RunningMeanStd, RewardScaler
from claude_code.ppo import PPOConfig, PPOTrainer, IterationStats, compute_gae, OBS_CLIP


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
            self.model = MLPActorCritic(**model_kwargs)
            self.model.eval()

            self.gamma = cfg_dict["gamma"]
            self.gae_lambda = cfg_dict["gae_lambda"]
            self.normalize_obs = cfg_dict["normalize_obs"]
            self.scale_reward = cfg_dict["scale_reward"]
            self.reconstruct = cfg_dict["reconstruct_state"]
            self.obs_dim = model_kwargs["obs_dim"]
            self.act_dim = model_kwargs["act_dim"]

            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if self.normalize_obs else None
            self.reward_scaler = RewardScaler(self.gamma) if self.scale_reward else None

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
                    self.env._observation_mode, sr, "cpu")

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
            ep_returns, ep_lengths, ep_components = [], [], []

            for t in range(n_steps):
                raw = self._next_obs
                norm = self._norm(raw)
                raw_buf[t] = raw
                obs_buf[t] = norm
                done_buf[t] = float(self._next_done)
                with torch.no_grad():
                    a, lp, _, v = self.model.get_action_and_value(
                        torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0))
                a_np = a.squeeze(0).numpy().astype(np.float32)
                act_buf[t] = a_np
                logp_buf[t] = float(lp.item())
                val_buf[t] = float(v.item())

                next_obs, reward, term, trunc, info = self.env.step(a_np)
                done = bool(term or trunc)
                if self._advance_recon is not None:
                    self._advance_recon(self.env._ownship_state, self.env._target_state)
                self._ep_return += float(reward)
                self._ep_len += 1
                rew_buf[t] = (self.reward_scaler.scale(float(reward), done)
                              if self.reward_scaler is not None else float(reward))
                if done:
                    ep_returns.append(self._ep_return)
                    ep_lengths.append(self._ep_len)
                    comp = info.get("ep_reward_components")
                    if isinstance(comp, dict):
                        ep_components.append(dict(comp))
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
                "ep_components": ep_components,
                "rms_mean": (raw_buf.mean(0) if self.obs_rms is not None else None),
                "rms_var": (raw_buf.var(0) if self.obs_rms is not None else None),
                "rms_count": n_steps,
            }

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

        self.model = MLPActorCritic(
            obs_dim, act_dim, hidden=config.hidden, activation=config.activation,
            log_std_init=config.log_std_init).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None

        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(config.hidden),
                            activation=config.activation, log_std_init=config.log_std_init)
        cfg_dict = dict(gamma=config.gamma, gae_lambda=config.gae_lambda,
                        normalize_obs=config.normalize_obs, scale_reward=config.scale_reward,
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
        return batch, ep_returns, ep_lengths, ep_components

    # 업데이트 로직은 PPOTrainer.update 재사용
    def update(self, batch):
        return PPOTrainer.update(self, batch)

    def train(self, on_iteration=None):
        import time
        history = []
        for it in range(1, self.cfg.total_iterations + 1):
            t0 = time.time()
            if self.cfg.anneal_lr:
                frac = 1.0 - (it - 1) / self.cfg.total_iterations
                for g in self.optimizer.param_groups:
                    g["lr"] = frac * self.cfg.lr
            batch, ep_returns, ep_lengths, ep_components = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            comp_means = {}
            if ep_components:
                for key in ("pursuit", "damage", "terminal", "safety", "step"):
                    comp_means[key] = float(np.mean([c.get(key, 0.0) for c in ep_components]))
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
