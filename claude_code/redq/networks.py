# -*- coding: utf-8 -*-
"""REDQ Q 앙상블 네트워크 (factorized discrete Q).

정책이 채널별 독립 Categorical 이므로 Q 도 같은 독립성 가정을 따른다: joint action
Q(s,a1,a2,a3,a4) 하나로 만들지 않고, **채널별 Q head** 로 둔다.

  Q_θ(s) : obs(obs_dim) → (act_dim, num_bins) 실수값

한 채널의 Q(s, a_c) 는 그 채널 행동 a_c 만의 함수로 본다. discrete SAC 의 채널별
기대값 계산(Σ_{a_c} π(a_c|s)·Q(s,a_c))과 정확히 맞물린다.

앙상블은 N 개의 독립 Q 망을 갖고, 각 망은 (act_dim, num_bins) 를 출력한다.
forward 는 (N, B, act_dim, num_bins) 를 반환한다. REDQ 는 매 업데이트마다 랜덤
subset M 개를 골라 그 min 으로 Bellman target 을 만든다(sac.py).

DroQ 변형: dropout + LayerNorm 을 넣은 Q 망 2개(기본) — 앙상블 대신 정규화로
분산을 잡는 Plan B. use_droq 로 선택.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

_ACTS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _q_mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str,
           dropout: float = 0.0, layernorm: bool = False) -> nn.Sequential:
    act = _ACTS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(act())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(module.bias)


class FactorizedQ(nn.Module):
    """단일 Q 망: obs → (act_dim, num_bins)."""

    def __init__(self, obs_dim: int, act_dim: int, num_bins: int,
                 hidden: Sequence[int], activation: str,
                 dropout: float = 0.0, layernorm: bool = False):
        super().__init__()
        self.act_dim = int(act_dim)
        self.num_bins = int(num_bins)
        self.net = _q_mlp(obs_dim, hidden, self.act_dim * self.num_bins, activation,
                          dropout=dropout, layernorm=layernorm)
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).view(-1, self.act_dim, self.num_bins)


class QEnsemble(nn.Module):
    """N 개의 독립 FactorizedQ 앙상블.

    forward(obs) → (N, B, act_dim, num_bins).
    subset(indices) 로 특정 멤버만 골라 stack 한 결과를 얻는다(REDQ target 용).
    """

    def __init__(self, obs_dim: int, act_dim: int, num_bins: int,
                 hidden: Sequence[int], activation: str, ensemble_size: int,
                 dropout: float = 0.0, layernorm: bool = False):
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.act_dim = int(act_dim)
        self.num_bins = int(num_bins)
        self.members = nn.ModuleList([
            FactorizedQ(obs_dim, act_dim, num_bins, hidden, activation,
                        dropout=dropout, layernorm=layernorm)
            for _ in range(self.ensemble_size)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(obs) for m in self.members], dim=0)

    def forward_subset(self, obs: torch.Tensor, indices) -> torch.Tensor:
        """indices 로 지정한 멤버만 forward 해 (len(indices), B, act_dim, num_bins) 반환."""
        return torch.stack([self.members[i](obs) for i in indices], dim=0)


def make_q_ensemble(cfg, obs_dim: int) -> QEnsemble:
    """RedqConfig 로 Q 앙상블 생성.

    DroQ 옵션이면 dropout+LayerNorm+작은 앙상블. 기본 REDQ 앙상블에도 LayerNorm 을
    적용한다(critic_layernorm, 기본 True) — 높은 UTD 에서 Q 발산을 막는 핵심 장치.
    """
    if cfg.use_droq:
        return QEnsemble(
            obs_dim, cfg.act_dim, cfg.num_bins, cfg.critic_hidden, cfg.critic_activation,
            ensemble_size=cfg.droq_ensemble_size,
            dropout=cfg.droq_dropout, layernorm=True)
    return QEnsemble(
        obs_dim, cfg.act_dim, cfg.num_bins, cfg.critic_hidden, cfg.critic_activation,
        ensemble_size=cfg.ensemble_size, dropout=0.0,
        layernorm=bool(cfg.critic_layernorm))


__all__ = ["FactorizedQ", "QEnsemble", "make_q_ensemble"]
