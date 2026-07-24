# -*- coding: utf-8 -*-
"""REDQ / discrete SAC 하이퍼파라미터 설정.

모든 스윕 대상(앙상블 크기 N, subset M, UTD ratio, target entropy 비율, buffer
크기·eviction, Polyak τ 등)을 여기 한 곳에 모은다. train_redq.py 의 CLI 가 이 값을
채운다. 기존 PPO 의 PPOConfig 와 대응되는 위치.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedqConfig:
    # ── 학습 규모 ────────────────────────────────────────────────────────────
    total_env_steps: int = 5_000_000     # 전체 수집 env-step(비교 기준 축)
    warmup_steps: int = 10_000           # 이 step 까지는 랜덤 정책으로 buffer 만 채움
    collect_steps_per_cycle: int = 1000  # driver 한 사이클에 수집할 env-step(전 worker 합)
    # ── discount / horizon ──────────────────────────────────────────────────
    gamma: float = 0.99
    # ── replay buffer ────────────────────────────────────────────────────────
    buffer_size: int = 1_000_000         # 전체 transition 상한(보수적으로 시작)
    batch_size: int = 256
    min_buffer_for_update: int = 5_000   # 이 이상 쌓이면 gradient step 시작
    # ── 네트워크 구조 ────────────────────────────────────────────────────────
    actor_hidden: tuple = (512, 512, 512)
    actor_activation: str = "relu"
    critic_hidden: tuple = (512, 512, 512)
    critic_activation: str = "relu"
    # critic LayerNorm: 높은 UTD 에서 Q 발산을 막는 사실상 필수 장치(DroQ/CrossQ/BRO).
    # 기본 켜짐. 앙상블 Q 에도 적용된다.
    critic_layernorm: bool = True
    num_bins: int = 21                   # 채널당 이산 카테고리 수(정책 구조와 일치)
    act_dim: int = 4
    # ── 최적화 ──────────────────────────────────────────────────────────────
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    max_grad_norm: float = 10.0
    # ── SAC entropy temperature (채널별 auto-tune) ───────────────────────────
    # target entropy = target_entropy_ratio * ln(num_bins) * act_dim.
    # discrete SAC 표준은 채널별 목표를 ln(bins) 의 일정 비율로 둔다. 전체(4채널) 합으로
    # 관리하되, 아래 비율로 목표를 잡는다. 0.0~1.0.
    target_entropy_ratio: float = 0.5
    init_alpha: float = 0.1              # 초기 온도(자동 조정되므로 시작값일 뿐)
    autotune_alpha: bool = True
    # alpha runaway 방지 rail. 높은 UTD 에선 alpha 가 env-step 당 utd 배 갱신되므로
    # 엔트로피가 목표를 따라오기 전에 alpha 가 과도하게 치솟아 Q target 을 부풀리고
    # Q 를 발산시킬 수 있다. log_alpha 를 [ln(alpha_min), ln(alpha_max)] 로 clamp 한다.
    alpha_min: float = 1e-4
    alpha_max: float = 2.0
    # ── REDQ ────────────────────────────────────────────────────────────────
    ensemble_size: int = 10              # N: Q 앙상블 크기 (Phase 0 은 1로 강제)
    subset_size: int = 2                 # M: Bellman target 에 쓸 랜덤 subset 크기(min)
    utd_ratio: int = 10                  # UTD: env-step 당 gradient update 횟수
    tau: float = 0.005                   # target network Polyak 계수
    # ── DroQ (Plan B) ────────────────────────────────────────────────────────
    use_droq: bool = False               # True 면 앙상블 대신 dropout+LayerNorm Q 2개
    droq_dropout: float = 0.01
    droq_ensemble_size: int = 2
    # ── self-play ────────────────────────────────────────────────────────────
    self_play: bool = True
    # replay eviction: pool 에서 snapshot 후보가 evict 되면 그 상대(gen)로 모은
    # transition 도 buffer 에서 제거한다. BT(gen=-1) transition 은 만료 없음.
    purge_evicted_opponents: bool = True
    # ── 관측 정규화 ──────────────────────────────────────────────────────────
    normalize_obs: bool = True
    obs_clip: float = 10.0
    reconstruct_state: bool = True       # claude_code.my_observation HP 재구성
    # ── 인프라 ──────────────────────────────────────────────────────────────
    num_workers: int = 6
    device: str = "cuda"
    seed: int = 0

    def target_entropy(self) -> float:
        """전체(4채널 합) 목표 엔트로피 [nats]."""
        import math
        return float(self.target_entropy_ratio) * math.log(self.num_bins) * self.act_dim

    def effective_ensemble_size(self) -> int:
        return self.droq_ensemble_size if self.use_droq else self.ensemble_size


__all__ = ["RedqConfig"]
