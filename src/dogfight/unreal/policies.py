from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dogfight.ai.action_provider import ActionProvider
from dogfight.ai.native_bt import AIPilot
from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.envs.observation import build_observation
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import CMD


@dataclass
class ConstantCommandPolicy:
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    throttle_cmd: float = 1.0

    def reset(self, context: RemoteClientContext) -> None:
        return None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=self.roll_cmd,
            pitch_cmd=self.pitch_cmd,
            yaw_cmd=self.yaw_cmd,
            throttle_cmd=self.throttle_cmd,
        )


class RLLightweightCommandPolicy:
    def __init__(
        self,
        action_provider: RLActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.geometry = GeometryInfo()

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        ownship_state = plane_info_to_state(context.own_plane.plane_info)
        target_state = plane_info_to_state(context.enemy_plane.plane_info)
        observation = self._build_observation(ownship_state, target_state)

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={"frame_index": context.frame_index},
            )
        )
        action = np.asarray(action_result.action, dtype=np.float32)

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
        )


class ProviderCommandPolicy:
    def __init__(
        self,
        action_provider: ActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
        ownship_force_side: int = 1,
        target_force_side: int = 2,
        action_repeat: int = 1,
        debug_action_repeat: bool = False,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.ownship_force_side = ownship_force_side
        self.target_force_side = target_force_side
        self.action_repeat = max(1, int(action_repeat))
        self.debug_action_repeat = debug_action_repeat
        self.geometry = GeometryInfo()
        self._state_pair_count = 0
        self._cached_action: np.ndarray | None = None
        self._last_policy_count: int | None = None
        self._last_policy_frame_index: int | None = None

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)
        self._state_pair_count = 0
        self._cached_action = None
        self._last_policy_count = None
        self._last_policy_frame_index = None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        own_plane = context.own_plane.plane_info
        enemy_plane = context.enemy_plane.plane_info
        pair_count = self._state_pair_count
        self._state_pair_count += 1

        policy_updated = (
            self._cached_action is None
            or pair_count % self.action_repeat == 0
        )
        if policy_updated:
            action = self._compute_provider_action(context, own_plane, enemy_plane)
            self._cached_action = action
            self._last_policy_count = pair_count
            self._last_policy_frame_index = context.frame_index
        else:
            action = np.asarray(self._cached_action, dtype=np.float32)

        if self.debug_action_repeat:
            self._print_action_repeat_debug(
                context=context,
                pair_count=pair_count,
                policy_updated=policy_updated,
                action=action,
            )

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _compute_provider_action(self, context, own_plane, enemy_plane) -> np.ndarray:
        ownship_state = plane_info_to_state(own_plane)
        target_state = plane_info_to_state(enemy_plane)
        observation = self._build_observation(ownship_state, target_state)

        own_speed = float(
            np.linalg.norm([own_plane.velocity.x, own_plane.velocity.y, own_plane.velocity.z])
        )
        target_speed = float(
            np.linalg.norm([enemy_plane.velocity.x, enemy_plane.velocity.y, enemy_plane.velocity.z])
        )
        my_plane_data = AIPilot.BuildPlaneData(
            [own_plane.position.x, own_plane.position.y, own_plane.position.z],
            [own_plane.rotation.roll, own_plane.rotation.pitch, own_plane.rotation.yaw],
            own_speed,
            self.ownship_force_side,
        )
        target_plane_data = AIPilot.BuildPlaneData(
            [enemy_plane.position.x, enemy_plane.position.y, enemy_plane.position.z],
            [enemy_plane.rotation.roll, enemy_plane.rotation.pitch, enemy_plane.rotation.yaw],
            target_speed,
            self.target_force_side,
        )

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={
                    "frame_index": context.frame_index,
                    "my_plane_id": context.plane_id,
                    "target_plane_id": enemy_plane.plane_id,
                    "my_force_side": self.ownship_force_side,
                    "target_force_side": self.target_force_side,
                    "my_plane_data": my_plane_data,
                    "target_plane_data": target_plane_data,
                },
            )
        )
        return np.asarray(action_result.action, dtype=np.float32)

    def _print_action_repeat_debug(
        self,
        context: RemoteClientContext,
        pair_count: int,
        policy_updated: bool,
        action: np.ndarray,
    ) -> None:
        repeat_offset = pair_count % self.action_repeat
        print(
            "[DogFightEnv][Unreal][ACTION_REPEAT] "
            f"pair_count={pair_count} repeat={self.action_repeat} "
            f"repeat_offset={repeat_offset} policy_updated={policy_updated} "
            f"cmd_frame={context.frame_index} "
            f"own_frame={context.own_plane.frame_index} "
            f"enemy_frame={context.enemy_plane.frame_index} "
            f"policy_frame={self._last_policy_frame_index} "
            f"policy_count={self._last_policy_count} "
            f"action={np.asarray(action, dtype=np.float32).tolist()}"
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
        )


def plane_info_to_state(plane_info) -> np.ndarray:
    state = np.zeros(51, dtype=np.float32)
    state[0] = plane_info.position.x
    state[1] = plane_info.position.y
    state[2] = plane_info.position.z
    state[3] = plane_info.rotation.roll
    state[4] = plane_info.rotation.pitch
    state[5] = plane_info.rotation.yaw
    state[6] = plane_info.velocity.x
    state[7] = plane_info.velocity.y
    state[8] = plane_info.velocity.z
    return state
