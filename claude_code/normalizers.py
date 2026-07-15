"""관측/보상 정규화 유틸 (PPO 수렴 속도 향상용).

표준 PPO 구현(CleanRL / SB3 VecNormalize)이 쓰는 두 가지 정규화를 제공한다:

  - RunningMeanStd: Welford 류 온라인 평균/분산 추정
  - 관측 정규화: obs → (obs - mean) / sqrt(var + eps)
  - 보상 스케일링: reward → reward / sqrt(Var[discounted_return] + eps)

관측 통계는 추론(submission)에서도 동일하게 적용해야 하므로 번들에 저장한다.
보상 스케일링은 학습에만 쓰이며 추론에는 영향이 없다.
"""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:  # single sample
            x = x[None, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningMeanStd":
        mean = np.asarray(state["mean"], dtype=np.float64)
        rms = cls(shape=mean.shape)
        rms.mean = mean
        rms.var = np.asarray(state["var"], dtype=np.float64)
        rms.count = float(state["count"])
        return rms


__all__ = ["RunningMeanStd"]
