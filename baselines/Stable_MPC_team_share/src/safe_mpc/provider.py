from __future__ import annotations

from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    clip_action,
)
from mpc.config import MPCConfig
from mpc.native import PublicState
from mpc.target_prediction import TargetMotionPredictor, TargetObservation
from mpc.transforms import body_velocity_to_ned

from .config import SafeMPCConfig
from .planner import SafeMPCPlanner, SafePlanResult


class SafeMPCActionProvider(ActionProvider):
    def __init__(
        self,
        root: str | Path,
        mpc_config: MPCConfig,
        safety_config: SafeMPCConfig,
    ):
        self.root = Path(root).resolve()
        self.mpc_config = mpc_config
        self.safety_config = safety_config
        self.planner = SafeMPCPlanner(
            self.root,
            mpc_config,
            safety_config,
        )
        self.target_predictor = TargetMotionPredictor(
            mpc_config.prediction,
            mpc_config.simulation_hz,
        )
        self._tick = 0
        self._cached_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.last_plan: SafePlanResult | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self.planner.reset()
        self.target_predictor.reset()
        self._tick = 0
        self._cached_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.last_plan = None

    @staticmethod
    def _time_from_state(state: np.ndarray, tick: int, simulation_hz: int) -> float:
        if state.size > 41 and np.isfinite(state[41]) and state[41] >= 0.0:
            return float(state[41])
        return tick / float(simulation_hz)

    @staticmethod
    def _target_observation(state: np.ndarray, time_s: float) -> TargetObservation:
        euler = np.asarray(state[3:6], dtype=np.float64)
        return TargetObservation(
            time_s=time_s,
            position_ned_m=np.asarray(state[0:3], dtype=np.float64),
            velocity_ned_mps=body_velocity_to_ned(state[6:9], euler),
            euler_deg=euler,
        )

    def _public_state(self, state: np.ndarray, time_s: float) -> PublicState:
        rates = (
            np.radians(np.asarray(state[9:12], dtype=np.float64))
            if state.size >= 12
            else np.zeros(3)
        )
        return PublicState(
            *np.asarray(state[0:9], dtype=np.float64).tolist(),
            *rates.tolist(),
            float(time_s),
            *self._cached_action.astype(np.float64).tolist(),
        )

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            return ActionResult(
                self._cached_action.copy(),
                "safe-mpc-fallback",
                0.0,
                {"error": "missing public state"},
            )
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        time_s = self._time_from_state(
            own,
            self._tick,
            self.mpc_config.simulation_hz,
        )
        target_observation = self._target_observation(target, time_s)
        self.target_predictor.update(target_observation)
        updated = self._tick % self.mpc_config.action_repeat == 0
        if updated:
            public_state = self._public_state(own, time_s)
            trajectory_steps = (
                self.mpc_config.knot_count * self.mpc_config.steps_per_knot
            )
            target_trajectory = self.target_predictor.predict(trajectory_steps)
            self.last_plan = self.planner.plan(
                public_state,
                target_trajectory,
            )
            self._cached_action = clip_action(self.last_plan.action)
        self._tick += 1
        info = {
            "policy_updated": updated,
            "tick": self._tick,
            "action_repeat": self.mpc_config.action_repeat,
        }
        if self.last_plan is not None:
            info.update(vars(self.last_plan))
            info["action"] = self._cached_action.copy()
        fallback = self.last_plan is not None and self.last_plan.fallback
        safety = self.last_plan is not None and self.last_plan.safety_override
        return ActionResult(
            action=self._cached_action.copy(),
            source=(
                "safe-mpc-fallback"
                if fallback
                else "safe-mpc-recovery" if safety else "safe-mpc"
            ),
            confidence=0.0 if fallback else 1.0,
            info=info,
        )

    def close(self) -> None:
        self.planner.close()
