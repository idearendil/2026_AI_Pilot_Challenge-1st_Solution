from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_ACTION_LOW = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
DEFAULT_ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


@dataclass
class ActionContext:
    sim: Any
    opponent_sim: Any
    ownship_state: np.ndarray | None = None
    target_state: np.ndarray | None = None
    observation: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    action: np.ndarray
    source: str
    confidence: float = 1.0
    info: dict[str, Any] = field(default_factory=dict)


class ActionProvider(ABC):
    def reset(self, context: ActionContext | None = None) -> None:
        return None

    @abstractmethod
    def compute_action(self, context: ActionContext) -> ActionResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


def clip_action(action, low=None, high=None) -> np.ndarray:
    low = DEFAULT_ACTION_LOW if low is None else np.asarray(low, dtype=np.float32)
    high = DEFAULT_ACTION_HIGH if high is None else np.asarray(high, dtype=np.float32)
    action_array = np.asarray(action, dtype=np.float32)
    # 2026-05-26: Guard native DLL outputs before they can propagate NaN/Inf.
    action_array = np.nan_to_num(action_array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(action_array, low, high)
