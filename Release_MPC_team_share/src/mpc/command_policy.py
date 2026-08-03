from __future__ import annotations

from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import CMD

from .config import MPCConfig
from .provider import MPCActionProvider
from .transforms import AngularRateEstimator


class MPCCommandPolicy:
    """Competition UDP adapter.  The protocol's yaw_cmd field is rudder command."""

    def __init__(self, root: str | Path, config: MPCConfig):
        self.provider = MPCActionProvider(root, config)
        self.config = config
        self._first_frame: int | None = None
        self._own_rates = AngularRateEstimator()
        self._target_rates = AngularRateEstimator()

    def reset(self, context: RemoteClientContext) -> None:
        self.provider.reset(None)
        self._first_frame = None
        self._own_rates.reset()
        self._target_rates.reset()

    def _to_state(self, plane, time_s: float, rate_estimator: AngularRateEstimator) -> np.ndarray:
        state = np.zeros(51, dtype=np.float64)
        state[0:3] = [plane.position.x, plane.position.y, plane.position.z]
        state[3:6] = [plane.rotation.roll, plane.rotation.pitch, plane.rotation.yaw]
        state[6:9] = [plane.velocity.x, plane.velocity.y, plane.velocity.z]
        state[9:12] = np.degrees(rate_estimator.update(state[3:6], time_s))
        state[41] = time_s
        return state

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            if self._first_frame is None:
                self._first_frame = int(context.frame_index)
            time_s = max(0.0, (int(context.frame_index) - self._first_frame) / self.config.simulation_hz)
            own = self._to_state(context.own_plane.plane_info, time_s, self._own_rates)
            target = self._to_state(context.enemy_plane.plane_info, time_s, self._target_rates)
            result = self.provider.compute_action(
                ActionContext(
                    sim=None,
                    opponent_sim=None,
                    ownship_state=own,
                    target_state=target,
                    info={"frame_index": context.frame_index},
                )
            )
            action = np.asarray(result.action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        action[3] = np.clip(action[3], 0.0, 1.0)
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def close(self) -> None:
        self.provider.close()
