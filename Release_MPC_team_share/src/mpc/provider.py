from __future__ import annotations

from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action

from .config import MPCConfig
from .native import PublicState
from .planner import MPCPlanner, PlanResult
from .target_prediction import TargetMotionPredictor, TargetObservation
from .transforms import body_velocity_to_ned


class MPCActionProvider(ActionProvider):
    def __init__(self, root: str | Path, config: MPCConfig):
        self.root = Path(root).resolve()
        self.config = config
        self.planner = MPCPlanner(self.root, config)
        self.target_predictor = TargetMotionPredictor(config.prediction, config.simulation_hz)
        self._tick = 0
        self._cached_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.last_plan: PlanResult | None = None

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

    def _target_observation(self, state: np.ndarray, time_s: float) -> TargetObservation:
        euler = np.asarray(state[3:6], dtype=np.float64)
        velocity_ned = body_velocity_to_ned(state[6:9], euler)
        return TargetObservation(
            time_s=time_s,
            position_ned_m=np.asarray(state[0:3], dtype=np.float64),
            velocity_ned_mps=velocity_ned,
            euler_deg=euler,
        )

    def _public_state(self, state: np.ndarray, time_s: float) -> PublicState:
        pqr = np.radians(np.asarray(state[9:12], dtype=np.float64)) if state.size >= 12 else np.zeros(3)
        return PublicState(
            *np.asarray(state[0:9], dtype=np.float64).tolist(),
            *pqr.tolist(),
            float(time_s),
            *self._cached_action.astype(np.float64).tolist(),
        )

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            return ActionResult(self._cached_action.copy(), "mpc-fallback", 0.0, {"error": "missing public state"})
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        time_s = self._time_from_state(own, self._tick, self.config.simulation_hz)
        self.target_predictor.update(self._target_observation(target, time_s))
        updated = self._tick % self.config.action_repeat == 0
        if updated:
            trajectory_steps = self.config.knot_count * self.config.steps_per_knot
            target_trajectory = self.target_predictor.predict(trajectory_steps)
            self.last_plan = self.planner.plan(self._public_state(own, time_s), target_trajectory)
            self._cached_action = clip_action(self.last_plan.action)
        self._tick += 1
        plan = self.last_plan
        info = {
            "policy_updated": updated,
            "tick": self._tick,
            "action_repeat": self.config.action_repeat,
        }
        if plan is not None:
            info.update(vars(plan))
            info["action"] = self._cached_action.copy()
        return ActionResult(
            action=self._cached_action.copy(),
            source="mpc-fallback" if plan is not None and plan.fallback else "mpc",
            confidence=0.0 if plan is not None and plan.fallback else 1.0,
            info=info,
        )

    def close(self) -> None:
        self.planner.close()
