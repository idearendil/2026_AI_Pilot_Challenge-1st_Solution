"""순수 PyTorch PPO (RLlib 의존 없음).

단일 DogFightWrapper 환경에서 on-policy rollout 을 모으고, GAE 로 advantage 를
계산한 뒤 clipped surrogate objective 로 정책을 업데이트한다. 학습 진행 상황을
iteration 단위로 출력하고(평균 episode return 등), 종료 시 2-파일 번들로 저장한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from claude_code.model import MLPActorCritic
from claude_code.normalizers import RunningMeanStd, RewardScaler

OBS_CLIP = 10.0


@dataclass
class PPOConfig:
    total_iterations: int = 50
    rollout_steps: int = 2048      # iteration 당 환경 step 수
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 256
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05   # 초과 시 epoch 조기 종료 (None 이면 비활성)
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    log_std_init: float = -0.5
    normalize_obs: bool = True          # 관측 running mean/std 정규화
    scale_reward: bool = True           # 할인 누적 보상 std 로 보상 스케일링
    anneal_lr: bool = True              # 학습률 선형 감쇠
    seed: int = 0
    device: str = "cpu"


@dataclass
class IterationStats:
    iteration: int
    global_step: int
    mean_return: float
    mean_length: float
    completed_episodes: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    explained_variance: float
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


class PPOTrainer:
    def __init__(self, env, config: PPOConfig):
        self.env = env
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        self.model = MLPActorCritic(
            obs_dim, act_dim,
            hidden=config.hidden,
            activation=config.activation,
            log_std_init=config.log_std_init,
        ).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_step = 0

        # 정규화기
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None
        self.reward_scaler = RewardScaler(config.gamma) if config.scale_reward else None

        # rollout 가로지르며 유지되는 환경 상태
        obs, _ = env.reset(seed=config.seed)
        self._next_obs = np.asarray(obs, dtype=np.float32)
        self._next_done = False
        self._ep_return = 0.0
        self._ep_len = 0

    def _normalize_obs(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        if update:
            self.obs_rms.update(obs)
        norm = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(norm, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    # ── rollout 수집 ─────────────────────────────────────────────────────────
    def collect_rollout(self):
        cfg = self.cfg
        T = cfg.rollout_steps
        device = cfg.device

        obs_buf = np.zeros((T, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, self.act_dim), dtype=np.float32)
        logp_buf = np.zeros(T, dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)
        val_buf = np.zeros(T, dtype=np.float32)

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_components: list[dict] = []

        for t in range(T):
            norm_obs = self._normalize_obs(self._next_obs, update=True)
            obs_buf[t] = norm_obs
            done_buf[t] = float(self._next_done)

            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, _, value = self.model.get_action_and_value(obs_tensor)
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            act_buf[t] = action_np
            logp_buf[t] = float(logp.item())
            val_buf[t] = float(value.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = bool(terminated or truncated)
            self.global_step += 1
            self._ep_return += float(reward)   # 로깅용 raw return
            self._ep_len += 1
            # GAE 용 보상은 스케일링 (학습에만 영향, raw return 은 별도 누적)
            rew_buf[t] = (
                self.reward_scaler.scale(float(reward), done)
                if self.reward_scaler is not None
                else float(reward)
            )

            if done:
                ep_returns.append(self._ep_return)
                ep_lengths.append(self._ep_len)
                comp = info.get("ep_reward_components")
                if isinstance(comp, dict):
                    ep_components.append(dict(comp))
                self._ep_return = 0.0
                self._ep_len = 0
                next_obs, _ = self.env.reset()

            self._next_obs = np.asarray(next_obs, dtype=np.float32)
            self._next_done = done

        # bootstrap value (통계 갱신 없이 정규화)
        with torch.no_grad():
            last_value = float(
                self.model.get_value(
                    torch.as_tensor(
                        self._normalize_obs(self._next_obs, update=False),
                        dtype=torch.float32, device=device,
                    ).unsqueeze(0)
                ).item()
            )

        adv_buf, ret_buf = self._compute_gae(rew_buf, val_buf, done_buf, last_value, self._next_done)

        batch = {
            "obs": torch.as_tensor(obs_buf, device=device),
            "actions": torch.as_tensor(act_buf, device=device),
            "logp": torch.as_tensor(logp_buf, device=device),
            "advantages": torch.as_tensor(adv_buf, device=device),
            "returns": torch.as_tensor(ret_buf, device=device),
            "values": torch.as_tensor(val_buf, device=device),
        }
        return batch, ep_returns, ep_lengths, ep_components

    def _compute_gae(self, rewards, values, dones, last_value, last_done):
        cfg = self.cfg
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    # ── 정책 업데이트 ────────────────────────────────────────────────────────
    def update(self, batch):
        cfg = self.cfg
        T = batch["obs"].shape[0]
        idx = np.arange(T)

        advantages = batch["advantages"]
        returns = batch["returns"]
        old_logp = batch["logp"]
        old_values = batch["values"]

        clip = cfg.clip_coef
        last_pl = last_vl = last_ent = last_kl = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            approx_kls = []
            for start in range(0, T, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_obs = batch["obs"][mb]
                mb_act = batch["actions"][mb]

                _, new_logp, entropy, new_value = self.model.get_action_and_value(mb_obs, mb_act)
                log_ratio = new_logp - old_logp[mb]
                ratio = log_ratio.exp()

                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # clipped value loss
                v_clipped = old_values[mb] + torch.clamp(
                    new_value - old_values[mb], -clip, clip
                )
                vf1 = (new_value - returns[mb]) ** 2
                vf2 = (v_clipped - returns[mb]) ** 2
                value_loss = 0.5 * torch.max(vf1, vf2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())

                last_pl = float(policy_loss.item())
                last_vl = float(value_loss.item())
                last_ent = float(entropy_loss.item())

            last_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
            if cfg.target_kl is not None and last_kl > cfg.target_kl:
                break

        # explained variance
        y_pred = old_values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

        return last_pl, last_vl, last_ent, last_kl, explained_var

    # ── 메인 루프 ────────────────────────────────────────────────────────────
    def train(self, on_iteration: Optional[Callable[[IterationStats], None]] = None):
        history: list[IterationStats] = []
        for it in range(1, self.cfg.total_iterations + 1):
            t0 = time.time()
            if self.cfg.anneal_lr:
                frac = 1.0 - (it - 1) / self.cfg.total_iterations
                for group in self.optimizer.param_groups:
                    group["lr"] = frac * self.cfg.lr
            batch, ep_returns, ep_lengths, ep_components = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            comp_means: dict = {}
            if ep_components:
                for key in ("pursuit", "damage", "terminal", "safety", "step"):
                    vals = [c.get(key, 0.0) for c in ep_components]
                    comp_means[key] = float(np.mean(vals))
            stats = IterationStats(
                iteration=it,
                global_step=self.global_step,
                mean_return=mean_ret,
                mean_length=mean_len,
                completed_episodes=len(ep_returns),
                policy_loss=pl,
                value_loss=vl,
                entropy=ent,
                approx_kl=kl,
                explained_variance=ev,
                elapsed_sec=time.time() - t0,
                extra=comp_means,
            )
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats)
        return history


__all__ = ["PPOConfig", "PPOTrainer", "IterationStats"]
