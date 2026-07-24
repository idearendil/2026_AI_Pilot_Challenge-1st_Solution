# -*- coding: utf-8 -*-
"""Discrete SAC + REDQ 학습 코어 (factorized categorical).

정책·Q 모두 채널별 독립(factorized) 구조를 따른다. 채널 c 의 Q head Q_c(s,·) 는
그 채널 행동만의 함수로 보고, discrete SAC 의 채널별 기대값과 맞물린다.

수식 (채널 c, 배치 s):
  soft value    V_c(s') = Σ_{a} π_c(a|s') ( minQ^tgt_c(s',a) − α logπ_c(a|s') )
  Bellman target y_c    = r + γ (1 − terminated) V_c(s')          # r 은 채널 공유
  critic loss           = mean_{i,c} ( Q^i_c(s, a_c) − y_c )²      # a_c = 취한 index
  actor loss            = mean_s Σ_c Σ_a π_c(a|s) ( α logπ_c(a|s) − meanQ_c(s,a) )
  alpha loss            = mean_s log α · ( H_total(s) − H̄ )        # H̄ = target entropy

REDQ: 매 update 마다 N개 앙상블 중 랜덤 M개를 골라 그 elementwise min 으로 target 을
만든다. actor 는 앙상블 전체 평균 Q 를 쓴다. N=M=1 이면 표준 single-Q discrete SAC.

reparameterization trick 없음(이산). 전체 카테고리 기대값을 정확히 계산한다.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from claude_code.model import MLPDiscreteActor
from claude_code.redq.networks import make_q_ensemble


class RedqLearner:
    """actor + Q 앙상블 + target + log_alpha 를 소유하고 gradient update 를 수행한다.

    driver(trainer.py)가 replay 에서 raw batch 를 뽑아 update() 에 넘긴다. 관측 정규화는
    현재 obs_rms 통계로 이 안에서 수행한다(raw 저장 → 최신 통계 정규화).
    """

    def __init__(self, cfg, obs_dim: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available()
                                   or cfg.device == "cpu" else "cpu")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(cfg.act_dim)
        self.num_bins = int(cfg.num_bins)

        self.actor = MLPDiscreteActor(
            obs_dim, cfg.act_dim, num_bins=cfg.num_bins,
            hidden=cfg.actor_hidden, activation=cfg.actor_activation).to(self.device)

        self.q = make_q_ensemble(cfg, obs_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.n_members = self.q.ensemble_size

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)

        self.target_entropy = cfg.target_entropy()
        self.autotune = bool(cfg.autotune_alpha)
        init_log_alpha = float(np.log(max(cfg.init_alpha, 1e-8)))
        self.log_alpha = torch.tensor(init_log_alpha, dtype=torch.float32,
                                      device=self.device, requires_grad=self.autotune)
        if self.autotune:
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self._rng = np.random.default_rng(int(cfg.seed) + 7)

    # ── 관측 정규화 (raw → 정규화, 현재 통계) ──────────────────────────────────
    def set_obs_stats(self, mean, var):
        m = np.asarray(mean, dtype=np.float32)
        v = np.asarray(var, dtype=np.float32)
        self._obs_mean = torch.as_tensor(m, device=self.device)
        self._obs_inv_std = torch.as_tensor(1.0 / np.sqrt(v + 1e-8), device=self.device)

    def _normalize(self, raw_obs_t: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_obs or not hasattr(self, "_obs_mean"):
            return raw_obs_t
        n = (raw_obs_t - self._obs_mean) * self._obs_inv_std
        return torch.clamp(n, -self.cfg.obs_clip, self.cfg.obs_clip)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    # ── 한 번의 gradient update (UTD 는 trainer 가 반복 호출) ────────────────────
    def update(self, batch) -> dict:
        dev = self.device
        raw_obs = torch.as_tensor(batch["obs"], device=dev)
        raw_next = torch.as_tensor(batch["next_obs"], device=dev)
        actions = torch.as_tensor(batch["actions"], device=dev, dtype=torch.long)  # (B,C)
        rewards = torch.as_tensor(batch["rewards"], device=dev).unsqueeze(-1)       # (B,1)
        terminated = torch.as_tensor(batch["terminated"], device=dev).unsqueeze(-1)  # (B,1)

        obs = self._normalize(raw_obs)
        next_obs = self._normalize(raw_next)
        alpha = self.log_alpha.exp().detach()

        # ── critic target ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_probs, next_logp = self.actor.probs_log_probs(next_obs)   # (B,C,K)
            # REDQ: N개 중 랜덤 M개 subset 의 elementwise min.
            m = min(self.cfg.subset_size, self.n_members)
            subset = self._rng.choice(self.n_members, size=m, replace=False)
            q_next = self.q_target.forward_subset(next_obs, subset)         # (M,B,C,K)
            min_q_next = q_next.min(dim=0).values                          # (B,C,K)
            # 채널별 soft value V_c(s') = Σ_a π( minQ − α logπ )
            v_next = (next_probs * (min_q_next - alpha * next_logp)).sum(-1)  # (B,C)
            y = rewards + self.cfg.gamma * (1.0 - terminated) * v_next        # (B,C)

        # ── critic loss (앙상블 전 멤버, 취한 action index gather) ───────────────
        q_all = self.q(obs)                                                # (N,B,C,K)
        a_idx = actions.unsqueeze(0).unsqueeze(-1).expand(self.n_members, -1, -1, 1)
        q_taken = q_all.gather(-1, a_idx).squeeze(-1)                       # (N,B,C)
        y_b = y.unsqueeze(0).expand_as(q_taken)                            # (N,B,C)
        q_loss = F.mse_loss(q_taken, y_b)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.q_opt.step()

        # ── actor loss (앙상블 평균 Q, 전체 카테고리 기대값) ─────────────────────
        probs, logp = self.actor.probs_log_probs(obs)                      # (B,C,K)
        with torch.no_grad():
            mean_q = self.q(obs).mean(dim=0)                               # (B,C,K)
        # Σ_c Σ_a π (α logπ − Q)
        actor_loss = (probs * (alpha * logp - mean_q)).sum(-1).sum(-1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        # 전체(채널 합) 엔트로피 H_total(s) = −Σ_c Σ_a π logπ
        entropy_total = -(probs.detach() * logp.detach()).sum(-1).sum(-1)   # (B,)

        # ── alpha auto-tune ───────────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.autotune:
            alpha_loss = (self.log_alpha * (entropy_total - self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            # runaway 방지: log_alpha 를 [ln(alpha_min), ln(alpha_max)] 로 clamp.
            with torch.no_grad():
                self.log_alpha.clamp_(float(np.log(self.cfg.alpha_min)),
                                      float(np.log(self.cfg.alpha_max)))
            alpha_loss_val = float(alpha_loss.item())

        # ── target network Polyak ─────────────────────────────────────────────
        with torch.no_grad():
            tau = self.cfg.tau
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.mul_(1.0 - tau).add_(tau * p)

        # ── 진단 지표 ─────────────────────────────────────────────────────────
        with torch.no_grad():
            # 앙상블 disagreement: 취한 action 위치에서 멤버 간 표준편차 평균
            q_std = q_taken.std(dim=0).mean().item() if self.n_members > 1 else 0.0
            # REDQ pessimism gap: 앙상블 평균 Q − 앙상블 min Q (전 멤버 기준). subset-min 이
            # 과대추정을 얼마나 끌어내리는지의 대리 지표(M 스윕 비교용). N=1 이면 0.
            if self.n_members > 1:
                q_pessimism = (q_taken.mean(dim=0) - q_taken.min(dim=0).values).mean().item()
            else:
                q_pessimism = 0.0
            # explained variance 대응: Q(s,a) vs target y
            qf = q_taken.mean(dim=0).reshape(-1).cpu().numpy()
            yt = y.reshape(-1).cpu().numpy()
            var_y = float(np.var(yt))
            ev = float("nan") if var_y == 0 else float(1.0 - np.var(yt - qf) / var_y)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
            "entropy": float(entropy_total.mean().item()),
            "q_mean": float(q_taken.mean().item()),
            "q_ensemble_std": float(q_std),
            "q_pessimism": float(q_pessimism),
            "td_target_mean": float(y.mean().item()),
            "explained_variance": ev,
        }

    # ── rollout 용 action 샘플링 (worker 로 weight 만 보내므로 실제로는 worker 가 함) ─
    @torch.no_grad()
    def act(self, raw_obs_np, deterministic: bool = False):
        raw = torch.as_tensor(np.asarray(raw_obs_np, dtype=np.float32),
                              device=self.device).unsqueeze(0)
        obs = self._normalize(raw)
        if deterministic:
            return self.actor.act_deterministic(obs).squeeze(0).cpu().numpy()
        return self.actor.act_stochastic(obs).squeeze(0).cpu().numpy()

    def actor_state_dict_cpu(self):
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}

    # ── 전체 학습 상태 저장/복원 (crash 자동 재시작용) ─────────────────────────
    def full_state(self) -> dict:
        """actor/Q/Q_target/optimizer/log_alpha 전부를 CPU dict 로 반환."""
        def _cpu(sd):
            return {k: v.detach().cpu() for k, v in sd.items()}
        st = {
            "actor": _cpu(self.actor.state_dict()),
            "q": _cpu(self.q.state_dict()),
            "q_target": _cpu(self.q_target.state_dict()),
            "actor_opt": self.actor_opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
        }
        if self.autotune:
            st["alpha_opt"] = self.alpha_opt.state_dict()
        return st

    def load_full_state(self, st: dict) -> None:
        self.actor.load_state_dict({k: torch.as_tensor(v) for k, v in st["actor"].items()})
        self.q.load_state_dict({k: torch.as_tensor(v) for k, v in st["q"].items()})
        self.q_target.load_state_dict({k: torch.as_tensor(v) for k, v in st["q_target"].items()})
        self.actor_opt.load_state_dict(st["actor_opt"])
        self.q_opt.load_state_dict(st["q_opt"])
        with torch.no_grad():
            self.log_alpha.fill_(float(st["log_alpha"]))
        if self.autotune and "alpha_opt" in st:
            self.alpha_opt.load_state_dict(st["alpha_opt"])


__all__ = ["RedqLearner"]
