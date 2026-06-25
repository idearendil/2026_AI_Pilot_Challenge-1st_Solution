from __future__ import annotations

import copy
import datetime
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymap3d as pm
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback for local setups
    import gym as gym

import FighterSim
import JSBSimWrapper
from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.ai.native_bt import AIPilot
from dogfight.config import FEET_TO_METER, METER_TO_FEET, merge_env_config
from dogfight.envs.observation import (
    build_observation,
    normalize,
    observation_size as builtin_observation_size,
)
from dogfight.envs.reward import compute_reward
from dogfight.envs.termination import evaluate_termination
from dogfight.sim.state_schema import StateIndex


REF_OLD_RANDOM_SCENARIOS = {
    0: (
        [1000.0, 0.0, -8100.0, 0.0, 0.0, 0.0, 250.0, 2],
        [3000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
    ),
    1: (
        [1500.0, 500.0, -8500.0, 0.0, 0.0, 0.0, 250.0, 2],
        [3000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
    ),
    2: (
        [3000.0, 1000.0, -8000.0, 0.0, 0.0, 270.0, 250.0, 2],
        [3000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
    ),
    3: (
        [3000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
        [2000.0, 1000.0, -8000.0, 0.0, 0.0, 315.0, 250.0, 2],
    ),
    4: (
        [3000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
        [2000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 2],
    ),
    5: (
        [3000.0, 1000.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
        [4000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
    ),
    6: (
        [3000.0, 1000.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
        [4000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
    ),
    7: (
        [3000.0, 1000.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
        [4000.0, 0.0, -8000.0, 0.0, 0.0, 0.0, 250.0, 1],
    ),
}


class DogFightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[dict] = None,
        AIP_ownship=None,
        AIP_target=None,
        ownship_action_provider=None,
        target_action_provider=None,
        reward_fn=None,  # 학생 정의 보상 함수; None이면 reward.py 기본값 사용
        observation_fn=None,
        observation_size=None,
        observation_low=None,
        observation_high=None,
    ):
        super().__init__()
        self._closed = False
        self.battle_space_id: int | None = None
        self._sim = None
        self._target_sim = None
        self._ownship_ai = None
        self._target_ai = None
        self._ownship_action_provider = ownship_action_provider
        self._target_action_provider = target_action_provider

        self.config = merge_env_config(env_config)
        self._runner_index = self.config.get("_runner_index", "local")
        self._env_index = self.config.get("_env_index", 0)
        self._geo_info = GeometryInfo()
        self._sim_hz = int(self.config["sim_hz"])
        self._step_ratio = self._resolve_step_ratio(self.config)
        self._delta_t = 1.0 / self._sim_hz
        self._max_engage_time = float(self.config["max_engage_time"])
        self._min_altitude = float(self.config["min_altitude"])
        self._observation_mode = self.config["observation_mode"]
        self._episode_step_limit = self.config.get("episode_step_limit")
        self._geometry_guard = self.config.get("geometry_guard", {})
        self._wez = self.config["wez"]
        self._reward_config = self.config["reward"]
        self._artifacts_dir = self.config["artifacts_dir"]
        self._reward_fn = reward_fn  # None → compute_reward from reward.py
        self._observation_fn = observation_fn

        self.battle_space_id = JSBSimWrapper.create_battleSpace()
        self._ownship_ai = (
            AIP_ownship if AIP_ownship is not None else self._build_ownship_ai()
        )
        self._target_ai = (
            AIP_target if AIP_target is not None else self._build_target_ai()
        )

        self.num_action = 4
        self.num_observation = (
            int(observation_size)
            if observation_fn is not None
            else builtin_observation_size(self._observation_mode)
        )
        if observation_fn is not None:
            low = -1.0 if observation_low is None else observation_low
            high = 1.0 if observation_high is None else observation_high
            self.observation_space = gym.spaces.Box(
                low=np.full(self.num_observation, low, dtype=np.float32)
                if np.isscalar(low) else np.asarray(low, dtype=np.float32),
                high=np.full(self.num_observation, high, dtype=np.float32)
                if np.isscalar(high) else np.asarray(high, dtype=np.float32),
                shape=(self.num_observation,),
                dtype=np.float32,
            )
        elif self._observation_mode == "tactical16":
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_observation,), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_observation,), dtype=np.float32
            )
        # All 4 action dims in [-1, 1] (throttle remapped to [0,1] internally).
        # Untrained networks output ≈0 → throttle = (0+1)/2 = 0.5 → no stall.
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.num_action, dtype=np.float32),
            high=np.ones(self.num_action, dtype=np.float32),
            shape=(self.num_action,),
            dtype=np.float32,
        )

        self._sim = FighterSim.JSBSim(
            [
                1,
                1,
                self.config["ownship"][0],
                self.config["ownship"][1],
                self.config["ownship"][2],
                self.config["ownship"][3],
                self.config["ownship"][4],
                self.config["ownship"][5],
                self.config["ownship"][6],
            ],
            self._ownship_ai,
            self._sim_hz,
            self.battle_space_id,
        )
        self._target_sim = FighterSim.JSBSim(
            [
                1,
                2,
                self.config["target"][0],
                self.config["target"][1],
                self.config["target"][2],
                self.config["target"][3],
                self.config["target"][4],
                self.config["target"][5],
                self.config["target"][6],
            ],
            self._target_ai,
            self._sim_hz,
            self.battle_space_id,
        )

        self.ownship_log: List[List[float]] = []
        self.target_log: List[List[float]] = []
        self.info: Dict[str, object] = {"end_condition": ""}
        self.ownship_damage = 0.0
        self.target_damage = 0.0
        self.num_engage = 0
        self.current_timestep = 0
        self.pre_obs = np.zeros(self.num_observation, dtype=np.float32)
        # Episode-level accumulators (reset in reset())
        self._in_wez = False
        self._ep_wez_steps = 0
        self._ep_step_count = 0
        self._ep_distance_sum = 0.0
        self._ep_distance_min = float("inf")
        self._ep_altitude_penalty_steps = 0
        self._ep_total_reward = 0.0
        self._ep_reward_components: Dict[str, float] = {}
        self._ep_action_sum = np.zeros(self.num_action, dtype=np.float64)
        self._ep_action_sq_sum = np.zeros(self.num_action, dtype=np.float64)
        self._initial_scenario_metrics: Dict[str, float] = {}

    # Sim expects throttle in [0, 1]; RL policy outputs throttle in [-1, 1].
    _SIM_ACTION_LOW  = np.array([-1., -1., -1., 0.], dtype=np.float32)
    _SIM_ACTION_HIGH = np.ones(4, dtype=np.float32)

    def _to_sim_action(self, rl_action: np.ndarray) -> np.ndarray:
        """Convert RL action space [-1,1]^4 → simulator format (throttle [0,1])."""
        a = np.clip(rl_action, -1.0, 1.0).astype(np.float32)
        a[3] = (a[3] + 1.0) / 2.0                        # [-1,1] → [0,1]
        return np.clip(a, self._SIM_ACTION_LOW, self._SIM_ACTION_HIGH)

    def _build_ai(self, dll_name):
        if not dll_name:
            return None
        return AIPilot(dll_name)

    def _build_ownship_ai(self):
        if self._ownship_action_provider is not None:
            return None
        if self.config["ownship_control_mode"] != "behavior_tree":
            return None
        return self._build_ai(self.config["ownship_behavior_dll"])

    def _build_target_ai(self):
        if self._target_action_provider is not None:
            return None
        if self.config["target_mode"] != "behavior_tree":
            return None
        return self._build_ai(self.config["target_behavior_dll"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options:
            self.config = merge_env_config({**self.config, **options})
            self._episode_step_limit = self.config.get("episode_step_limit")
            self._geometry_guard = self.config.get("geometry_guard", {})
            self._reward_config = self.config["reward"]

        scenario = self.config.get("initial_scenario", {})
        scenario_mode = scenario.get("mode", "default")
        if scenario_mode == "two_circle_headon":
            self._apply_two_circle_headon_initial_scenario(scenario)
        elif scenario_mode == "ref_old_random":
            self._apply_ref_old_random_initial_scenario(scenario)
        else:
            self._initial_scenario_metrics = {}

        # Apply per-episode position randomization if configured
        rand = self.config.get("ownship_randomization", {})
        if scenario_mode != "two_circle_headon" and rand.get("enabled", False):
            self.add_random_init_position(
                "ownship",
                radius=float(rand.get("radius", 0)),
                r_roll=float(rand.get("r_roll", 0)),
                r_pitch=float(rand.get("r_pitch", 0)),
                r_heading=float(rand.get("r_heading", 0)),
            )

        JSBSimWrapper.Reset(self.battle_space_id)
        self._ownship_state = self._sim.reset()
        self._target_state = self._target_sim.reset()
        self._update_initial_geometry_metrics(scenario_mode)
        self.pre_obs = self.get_observation()
        self.info = {"end_condition": "", **self._initial_scenario_metrics}
        self.ownship_damage = 0.0
        self.target_damage = 0.0
        self.num_engage += 1
        self.current_timestep = 0
        self._reset_action_providers()
        # Reset episode accumulators
        self._in_wez = False
        self._ep_wez_steps = 0
        self._ep_step_count = 0
        self._ep_distance_sum = 0.0
        self._ep_distance_min = float("inf")
        self._ep_altitude_penalty_steps = 0
        self._ep_total_reward = 0.0
        self._ep_reward_components = {}
        self._ep_action_sum = np.zeros(self.num_action, dtype=np.float64)
        self._ep_action_sq_sum = np.zeros(self.num_action, dtype=np.float64)
        return np.array(self.pre_obs, dtype=np.float32), dict(self.info)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        failure = self._advance_simulation_step_ratio(action)
        if failure is not None:
            return failure
        cur_obs = self.get_observation()

        terminated, truncated, end_condition = evaluate_termination(
            self._ownship_state,
            self._target_state,
            self._sim,
            self._target_sim,
            self._max_engage_time,
            self._min_altitude,
            self.current_timestep,
            self._episode_step_limit,
            self._geo_info,
            self._geometry_guard,
        )

        ownship_health = float(self._ownship_state[StateIndex.HEALTH])
        target_health = float(self._target_state[StateIndex.HEALTH])
        outcome = self._classify_outcome(
            terminated,
            truncated,
            end_condition,
            ownship_health,
            target_health,
        )
        reward, components = self._compute_step_reward(
            terminated,
            truncated,
            end_condition,
        )

        ep_mean_dist, ep_min_dist = self._update_episode_metrics(
            action,
            reward,
            components,
        )

        self.info = {
            "end_condition": end_condition,
            "outcome": outcome,
            "ownship_damage": self.ownship_damage,
            "target_damage": self.target_damage,
            "ownship_health": ownship_health,
            "target_health": target_health,
            "reward_components": components,
            "ep_reward_components": dict(self._ep_reward_components),
            "ep_wez_steps": self._ep_wez_steps,
            "ep_step_count": self._ep_step_count,
            "ep_mean_distance": ep_mean_dist,
            "ep_min_distance": ep_min_dist,
            "ep_altitude_penalty_steps": self._ep_altitude_penalty_steps,
            "final_ata_deg": abs(float(self._geo_info._get_antenna_train_angle(
                self._ownship_state, self._target_state, True
            ))),
            "final_aa_deg": abs(float(self._geo_info._get_aspect_angle(
                self._ownship_state, self._target_state, True
            ))),
            "headon_guard_fail": end_condition == "two circle headon guard fail",
            **self._initial_scenario_metrics,
        }

        self._append_logs()
        self.pre_obs = copy.deepcopy(cur_obs)
        self.current_timestep += 1
        if terminated or truncated:
            self._print_episode_termination(end_condition)
        return np.array(cur_obs, dtype=np.float32), reward, terminated, truncated, dict(self.info)

    def _advance_simulation_step_ratio(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, Dict] | None:
        ownship_damage_total = 0.0
        target_damage_total = 0.0
        in_wez_any = False
        for _ in range(int(self._step_ratio)):
            self._step_controlled_aircraft(action)
            if np.isnan(self._sim.get_state()).any():
                info = {"end_condition": "Ownship FDM output Fall", "outcome": "crash"}
                self._ep_step_count += 1
                self._print_episode_termination(info["end_condition"])
                return np.array(self.pre_obs, dtype=np.float32), 0.0, True, False, info

            self._step_target_aircraft()
            if np.isnan(self._target_sim.get_state()).any():
                info = {"end_condition": "Target FDM output Fall", "outcome": "other"}
                self._ep_step_count += 1
                self._print_episode_termination(info["end_condition"])
                return np.array(self.pre_obs, dtype=np.float32), 0.0, True, False, info

            self.update_damage()
            ownship_damage_total += float(self.ownship_damage)
            target_damage_total += float(self.target_damage)
            in_wez_any = in_wez_any or self._in_wez
            self._ownship_state = self._sim.get_state()
            self._target_state = self._target_sim.get_state()

        self.ownship_damage = ownship_damage_total
        self.target_damage = target_damage_total
        self._in_wez = in_wez_any
        return None

    @staticmethod
    def _classify_outcome(
        terminated: bool,
        truncated: bool,
        end_condition: str,
        ownship_health: float,
        target_health: float,
    ) -> str:
        if terminated:
            if target_health <= 0.0 < ownship_health:
                return "win"
            if ownship_health <= 0.0 < target_health:
                return "loss"
            if end_condition in ("ownship altitude below min", "FDM Update Fail"):
                return "crash"
            if end_condition == "two circle headon guard fail":
                return "loss"
            return "draw"
        if truncated:
            return "timeout"
        return "ongoing"

    def _compute_step_reward(
        self,
        terminated: bool,
        truncated: bool,
        end_condition: str,
    ) -> tuple[float, dict]:
        _reward_fn = self._reward_fn if self._reward_fn is not None else compute_reward
        return _reward_fn(
            self._ownship_state,
            self._target_state,
            self.ownship_damage,
            self.target_damage,
            self._geo_info,
            self._wez,
            self._reward_config,
            terminated,
            truncated,
            end_condition,
        )

    def _update_episode_metrics(
        self,
        action: np.ndarray,
        reward: float,
        components: dict,
    ) -> tuple[float, float]:
        distance = self._geo_info._get_distance(self._ownship_state, self._target_state)
        self._ep_step_count += 1
        self._ep_total_reward += float(reward)
        self._ep_distance_sum += distance
        self._ep_distance_min = min(self._ep_distance_min, distance)
        if self._in_wez:
            self._ep_wez_steps += 1
        if components.get("safety", 0.0) < 0.0:
            self._ep_altitude_penalty_steps += 1
        for k, v in components.items():
            self._ep_reward_components[k] = self._ep_reward_components.get(k, 0.0) + v
        self._ep_action_sum += action.astype(np.float64)
        self._ep_action_sq_sum += action.astype(np.float64) ** 2

        ep_mean_dist = self._ep_distance_sum / self._ep_step_count
        ep_min_dist = self._ep_distance_min if self._ep_step_count > 0 else 0.0
        return ep_mean_dist, ep_min_dist

    @staticmethod
    def _resolve_step_ratio(config: dict) -> float:
        """Resolve RL-action hold ratio from explicit or legacy timing config."""
        ratio = config.get("step_ratio")
        if ratio is None:
            delta = config.get("delta")
            time_step = config.get("time_step")
            if delta is not None and time_step is not None:
                ratio = float(delta) / float(time_step)
            else:
                ratio = 1.0
        ratio = float(ratio)
        if int(ratio) <= 0:
            raise ValueError(f"step_ratio must resolve to at least 1, got {ratio}")
        return ratio

    def _step_controlled_aircraft(self, action: np.ndarray) -> None:
        if self._ownship_action_provider is not None:
            context = self._build_action_context(
                self._sim,
                self._target_sim,
                self._ownship_state,
                self._target_state,
                self.pre_obs,
            )
            result = self._ownship_action_provider.compute_action(context)
            self._sim.step(result.action)
            return

        control_mode = self.config["ownship_control_mode"]
        if control_mode == "behavior_tree":
            self._sim.step_behavior(self._target_sim.get_model())
        elif control_mode == "fixed":
            self._sim.step_fix()
        elif control_mode == "loiter":
            loiter = self.config["target_loiter"]
            self._sim.step_loiter(loiter["enabled"], loiter["bank"], loiter["pitch"])
        else:
            self._sim.step(self._to_sim_action(action))

    def _step_target_aircraft(self) -> None:
        if self._target_action_provider is not None:
            context = self._build_action_context(
                self._target_sim,
                self._sim,
                self._target_state,
                self._ownship_state,
                self.pre_obs,
            )
            result = self._target_action_provider.compute_action(context)
            self._target_sim.step(result.action)
            return

        target_mode = self.config["target_mode"]
        if target_mode == "behavior_tree":
            self._target_sim.step_behavior(self._sim.get_model())
        elif target_mode == "fixed":
            self._target_sim.step_fix()
        elif target_mode == "loiter":
            loiter = self.config["target_loiter"]
            self._target_sim.step_loiter(loiter["enabled"], loiter["bank"], loiter["pitch"])
        elif target_mode == "autopilot":
            autopilot = self.config["target_autopilot"]
            self._target_sim.step_autopilot(
                autopilot["heading_cmd"],
                autopilot["altitude_cmd"],
                autopilot["speed_cmd"],
            )
        else:
            self._target_sim.step_fix()

    def get_observation(self):
        return self._build_observation_for(self._ownship_state, self._target_state)

    def _build_observation_for(self, ownship_state, target_state) -> np.ndarray:
        if self._observation_fn is not None:
            observation = self._observation_fn(
                np.array(ownship_state, copy=True),
                np.array(target_state, copy=True),
                self._geo_info,
                self._wez,
            )
            observation = np.asarray(observation, dtype=np.float32)
            if observation.shape != (self.num_observation,):
                raise ValueError(
                    "custom observation_fn returned shape "
                    f"{observation.shape}, expected {(self.num_observation,)}"
                )
            return observation

        wez_cfg = self._wez if self._observation_mode == "tactical16" else None
        return build_observation(
            self._observation_mode,
            ownship_state,
            target_state,
            self._geo_info,
            wez_cfg,
        )

    def get_reward(self):
        terminated, truncated, end_condition = evaluate_termination(
            self._ownship_state,
            self._target_state,
            self._sim,
            self._target_sim,
            self._max_engage_time,
            self._min_altitude,
            self.current_timestep,
            self._episode_step_limit,
            self._geo_info,
            self._geometry_guard,
        )
        _reward_fn = self._reward_fn if self._reward_fn is not None else compute_reward
        result = _reward_fn(
            self._ownship_state,
            self._target_state,
            self.ownship_damage,
            self.target_damage,
            self._geo_info,
            self._wez,
            self._reward_config,
            terminated,
            truncated,
            end_condition,
        )
        return result[0] if isinstance(result, tuple) else result

    def get_done(self):
        return evaluate_termination(
            self._ownship_state,
            self._target_state,
            self._sim,
            self._target_sim,
            self._max_engage_time,
            self._min_altitude,
            self.current_timestep,
            self._episode_step_limit,
            self._geo_info,
            self._geometry_guard,
        )[:2]

    def _print_episode_termination(self, end_condition: str) -> None:
        print(
            f"runner:{self._runner_index}/env:{self._env_index}\t"
            f"num_steps=[{self._ep_step_count}] | "
            f"total rewards=[{self._ep_total_reward:.4f}] | "
            f"termination= [{end_condition}]",
            flush=True,
        )

    def update_damage(self):
        sim_state = self._sim.get_state()
        target_sim_state = self._target_sim.get_state()
        dis_m = self._geo_info._get_distance(sim_state, target_sim_state)
        ownship_ata_deg = self._geo_info._get_antenna_train_angle(sim_state, target_sim_state, False)
        target_ata_deg = self._geo_info._get_antenna_train_angle(target_sim_state, sim_state, False)

        max_range_m = self._wez["max_range_m"]
        min_range_m = self._wez["min_range_m"]
        base_range_m = max_range_m - min_range_m
        half_wez_angle_deg = self._wez["angle_deg"] / 2.0

        target_damage = 0.0
        ownship_damage = 0.0
        if base_range_m != 0 and min_range_m <= dis_m <= max_range_m:
            if half_wez_angle_deg >= abs(ownship_ata_deg):
                target_damage = ((max_range_m - dis_m) / base_range_m) * self._delta_t
            if half_wez_angle_deg >= abs(target_ata_deg):
                ownship_damage = ((max_range_m - dis_m) / base_range_m) * self._delta_t

        self.ownship_damage = ownship_damage
        self.target_damage = target_damage
        self._in_wez = target_damage > 0.0  # True when ownship is inside WEZ toward target
        self._sim.deduct_health(ownship_damage)
        self._target_sim.deduct_health(target_damage)

    def change_init_position(
        self,
        flight="ownship",
        init_n=0,
        init_e=0,
        init_d=-8000,
        init_roll=0,
        init_pitch=0,
        init_heading=0,
        init_speed=250,
        target_type=2,
    ):
        fighter = self._sim if flight == "ownship" else self._target_sim
        self._target_type = target_type
        fighter._init_pos_n = init_n
        fighter._init_pos_e = init_e
        fighter._init_pos_d = init_d
        fighter._init_roll = init_roll
        fighter._init_pitch = init_pitch
        fighter._init_heading = init_heading
        fighter._init_speed = init_speed
        lla = pm.ned2geodetic(
            fighter._init_pos_n,
            fighter._init_pos_e,
            fighter._init_pos_d,
            fighter._origin_lat,
            fighter._origin_lon,
            fighter._origin_alt,
        )
        fighter._init_pos_lat = lla[0]
        fighter._init_pos_lon = lla[1]
        fighter._init_pos_alt = lla[2]

    def _apply_two_circle_headon_initial_scenario(self, scenario: dict) -> None:
        """Place both aircraft on the paper's two-circle head-on curriculum."""
        alpha_deg = float(scenario.get("alpha_deg", 0.0))
        alpha_rad = math.radians(alpha_deg)
        turn_diameter_ft = float(scenario.get("turn_diameter_ft", 6000.0))
        jitter_min_ft, jitter_max_ft = self._range_values(
            scenario.get("separation_jitter_ft", [3000.0, 6000.0])
        )
        jitter_ft = float(self.np_random.uniform(jitter_min_ft, jitter_max_ft))
        separation_m = (
            2.0 * turn_diameter_ft * math.sin(alpha_rad) + jitter_ft
        ) * FEET_TO_METER

        center_n = float(scenario.get("center_n_m", 3500.0))
        center_e = float(scenario.get("center_e_m", 0.0))
        altitude_m = float(scenario.get("altitude_m", 7000.0))
        half_sep = separation_m / 2.0

        speed_min, speed_max = self._range_values(
            scenario.get("speed_mps_range", [250.0, 300.0])
        )
        roll_min, roll_max = self._range_values(
            scenario.get("roll_range_deg", [0.0, 180.0])
        )
        side = float(self.np_random.choice(scenario.get("side_choices", [-1.0, 1.0])))
        pitch = float(self.np_random.choice(
            scenario.get("vertical_pitch_choices_deg", [0.0, 10.0, -10.0])
        ))

        own_heading = self._wrap_heading(side * alpha_deg)
        target_heading = self._wrap_heading(180.0 + side * alpha_deg)
        own_roll = float(self.np_random.uniform(roll_min, roll_max))
        target_roll = float(self.np_random.uniform(roll_min, roll_max))
        own_speed = float(self.np_random.uniform(speed_min, speed_max))
        target_speed = float(self.np_random.uniform(speed_min, speed_max))

        self.change_init_position(
            "ownship",
            init_n=center_n - half_sep,
            init_e=center_e,
            init_d=-altitude_m,
            init_roll=own_roll,
            init_pitch=pitch,
            init_heading=own_heading,
            init_speed=own_speed,
        )
        self.change_init_position(
            "target",
            init_n=center_n + half_sep,
            init_e=center_e,
            init_d=-altitude_m,
            init_roll=target_roll,
            init_pitch=-pitch,
            init_heading=target_heading,
            init_speed=target_speed,
        )

        self._initial_scenario_metrics = {
            "initial_alpha_deg": alpha_deg,
            "initial_distance_m": separation_m,
        }

    def _apply_ref_old_random_initial_scenario(self, scenario: dict) -> None:
        """Apply ref_oldDogFightEnv 1vs1 mixed BT/loiter initial scenarios."""
        indices = list(scenario.get("legacy_scenario_indices", [0]))
        if scenario.get("legacy_use_first_scenario_only", False):
            scenario_index = int(indices[0])
        elif scenario.get("legacy_use_random_scenario", True):
            scenario_index = int(self.np_random.choice(indices))
        else:
            scenario_index = int(scenario.get("legacy_scenario_index", indices[0]))
        ownship, target = REF_OLD_RANDOM_SCENARIOS[scenario_index]

        self.change_init_position(
            "ownship",
            init_n=ownship[0],
            init_e=ownship[1],
            init_d=ownship[2],
            init_roll=ownship[3],
            init_pitch=ownship[4],
            init_heading=ownship[5],
            init_speed=ownship[6],
            target_type=ownship[7],
        )
        self.change_init_position(
            "target",
            init_n=target[0],
            init_e=target[1],
            init_d=target[2],
            init_roll=target[3],
            init_pitch=target[4],
            init_heading=target[5],
            init_speed=target[6],
            target_type=target[7],
        )

        randomization = scenario.get("legacy_randomization", {})
        self.add_random_init_position(
            "ownship",
            radius=float(randomization.get("aircraft_radius_m", 100.0)),
            r_roll=float(randomization.get("roll_deg", 5.0)),
            r_pitch=float(randomization.get("pitch_deg", 5.0)),
            r_heading=float(randomization.get("heading_deg", 5.0)),
        )
        self.add_random_init_position(
            "target",
            radius=float(randomization.get("aircraft_radius_m", 100.0)),
            r_roll=float(randomization.get("roll_deg", 5.0)),
            r_pitch=float(randomization.get("pitch_deg", 5.0)),
            r_heading=float(randomization.get("heading_deg", 5.0)),
        )
        self._add_ref_old_shared_random_starting_position(randomization)

        if int(target[7]) == 1:
            bank_min, bank_max = self._range_values(
                randomization.get("loiter_bank_deg_range", [40.0, 70.0])
            )
            bank = float(
                self.np_random.choice([-1.0, 1.0])
                * self.np_random.uniform(bank_min, bank_max)
            )
            self.config["target_mode"] = "loiter"
            self.config["target_loiter"] = {
                "enabled": True,
                "bank": bank,
                "pitch": 0.0,
            }
        else:
            self.config["target_mode"] = "behavior_tree"

        self._initial_scenario_metrics = {
            "legacy_scenario_index": float(scenario_index),
        }

    def _add_ref_old_shared_random_starting_position(self, randomization: dict) -> None:
        sign = np.array([-1.0, 1.0])
        rand_n = float(
            self.np_random.choice(sign)
            * self.np_random.uniform(
                0.0, float(randomization.get("shared_n_m", 4000.0))
            )
        )
        rand_e = float(
            self.np_random.choice(sign)
            * self.np_random.uniform(
                0.0, float(randomization.get("shared_e_m", 4000.0))
            )
        )
        rand_d = float(
            self.np_random.choice(sign)
            * self.np_random.uniform(
                0.0, float(randomization.get("shared_d_m", 4000.0))
            )
        )
        rand_distance_n = float(
            self.np_random.choice(sign)
            * self.np_random.uniform(
                0.0, float(randomization.get("target_distance_n_m", 300.0))
            )
        )
        rand_speed = float(
            self.np_random.choice(sign)
            * self.np_random.uniform(
                0.0, float(randomization.get("speed_mps", 50.0))
            )
        )

        self._sim._init_pos_n += rand_n
        self._sim._init_pos_e += rand_e
        self._sim._init_pos_d += rand_d
        self._sim._init_speed += rand_speed
        self._target_sim._init_pos_n += rand_n + rand_distance_n
        self._target_sim._init_pos_e += rand_e
        self._target_sim._init_pos_d += rand_d
        self._target_sim._init_speed += rand_speed

        for fighter in (self._sim, self._target_sim):
            lla = pm.ned2geodetic(
                fighter._init_pos_n,
                fighter._init_pos_e,
                fighter._init_pos_d,
                fighter._origin_lat,
                fighter._origin_lon,
                fighter._origin_alt,
            )
            fighter._init_pos_lat = lla[0]
            fighter._init_pos_lon = lla[1]
            fighter._init_pos_alt = lla[2]

    def _update_initial_geometry_metrics(self, scenario_mode: str) -> None:
        if scenario_mode not in ("two_circle_headon", "ref_old_random"):
            return
        ata = abs(float(self._geo_info._get_antenna_train_angle(
            self._ownship_state, self._target_state, True
        )))
        aa = abs(float(self._geo_info._get_aspect_angle(
            self._ownship_state, self._target_state, True
        )))
        distance = float(self._geo_info._get_distance(
            self._ownship_state, self._target_state
        ))
        self._initial_scenario_metrics.update({
            "initial_ata_deg": ata,
            "initial_aa_deg": aa,
            "initial_distance_m": distance,
        })

    @staticmethod
    def _range_values(values) -> tuple[float, float]:
        if isinstance(values, (int, float)):
            value = float(values)
            return value, value
        if len(values) != 2:
            raise ValueError(f"Expected two range values, got {values!r}")
        return float(values[0]), float(values[1])

    @staticmethod
    def _wrap_heading(value: float) -> float:
        return float(value % 360.0)

    def add_random_init_position(self, flight="ownship", radius=500.0, r_roll=5, r_pitch=5, r_heading=5):
        fighter = self._sim if flight == "ownship" else self._target_sim
        sign = np.array([-1.0, 1.0])
        fighter._init_pos_n += float(self.np_random.choice(sign) * self.np_random.integers(0, radius))
        fighter._init_pos_e += float(self.np_random.choice(sign) * self.np_random.integers(0, radius))
        fighter._init_pos_d += float(self.np_random.choice(sign) * self.np_random.integers(0, radius))
        fighter._init_roll += float(self.np_random.choice(sign) * self.np_random.integers(0, r_roll))
        fighter._init_pitch += float(self.np_random.choice(sign) * self.np_random.integers(0, r_pitch))
        fighter._init_heading += float(self.np_random.choice(sign) * self.np_random.integers(0, r_heading))

        if fighter._init_roll > 180:
            fighter._init_roll -= 360
        if fighter._init_roll < -180:
            fighter._init_roll += 360
        if fighter._init_pitch > 180:
            fighter._init_pitch -= 360
        if fighter._init_pitch < -180:
            fighter._init_pitch += 360
        if fighter._init_heading > 360:
            fighter._init_heading -= 360
        if fighter._init_heading < 0:
            fighter._init_heading += 360

        lla = pm.ned2geodetic(
            fighter._init_pos_n,
            fighter._init_pos_e,
            fighter._init_pos_d,
            fighter._origin_lat,
            fighter._origin_lon,
            fighter._origin_alt,
        )
        fighter._init_pos_lat = lla[0]
        fighter._init_pos_lon = lla[1]
        fighter._init_pos_alt = lla[2]

    def _append_logs(self):
        self.ownship_log.append(
            [
                self._ownship_state[StateIndex.LAT],
                self._ownship_state[StateIndex.LON],
                self._ownship_state[StateIndex.ALT],
                self._ownship_state[StateIndex.ROLL],
                self._ownship_state[StateIndex.PITCH],
                self._ownship_state[StateIndex.YAW],
                self._ownship_state[StateIndex.HEALTH],
            ]
        )
        self.target_log.append(
            [
                self._target_state[StateIndex.LAT],
                self._target_state[StateIndex.LON],
                self._target_state[StateIndex.ALT],
                self._target_state[StateIndex.ROLL],
                self._target_state[StateIndex.PITCH],
                self._target_state[StateIndex.YAW],
                self._target_state[StateIndex.HEALTH],
            ]
        )

    def _build_action_context(self, sim, opponent_sim, ownship_state, target_state, observation) -> ActionContext:
        if ownship_state is not None and target_state is not None:
            observation = self._build_observation_for(
                np.array(ownship_state, copy=True),
                np.array(target_state, copy=True),
            )
        return ActionContext(
            sim=sim,
            opponent_sim=opponent_sim,
            ownship_state=np.array(ownship_state, copy=True) if ownship_state is not None else None,
            target_state=np.array(target_state, copy=True) if target_state is not None else None,
            observation=np.array(observation, copy=True) if observation is not None else None,
            info={"timestep": self.current_timestep},
        )

    def _reset_action_providers(self) -> None:
        for provider, sim, opponent, ownship_state, target_state in (
            (self._ownship_action_provider, self._sim, self._target_sim, self._ownship_state, self._target_state),
            (self._target_action_provider, self._target_sim, self._sim, self._target_state, self._ownship_state),
        ):
            if provider is None:
                continue
            provider.reset(self._build_action_context(sim, opponent, ownship_state, target_state, self.pre_obs))

    def get_ownship_sim(self):
        return self._sim

    def get_target_sim(self):
        return self._target_sim

    def get_ownship_action(self):
        return np.array(self._sim.action, dtype=np.float32)

    def get_target_action(self):
        return np.array(self._target_sim.action, dtype=np.float32)

    def get_ownship_VP(self):
        return np.array(self._sim.VP, dtype=np.float32)

    def get_target_VP(self):
        return np.array(self._target_sim.VP, dtype=np.float32)

    def get_ownship_state(self) -> np.ndarray:
        return self._ownship_state

    def get_target_state(self) -> np.ndarray:
        return self._target_state

    def get_damage(self):
        return self.ownship_damage, self.target_damage

    def get_ownship_state_for_udp(self) -> List:
        return [
            self._ownship_state[StateIndex.N],
            self._ownship_state[StateIndex.E],
            -self._ownship_state[StateIndex.D],
            self._ownship_state[StateIndex.ROLL],
            self._ownship_state[StateIndex.PITCH],
            self._ownship_state[StateIndex.YAW],
            self._ownship_state[StateIndex.HEALTH],
        ]

    def get_target_state_for_udp(self) -> List:
        return [
            self._target_state[StateIndex.N],
            self._target_state[StateIndex.E],
            -self._target_state[StateIndex.D],
            self._target_state[StateIndex.ROLL],
            self._target_state[StateIndex.PITCH],
            self._target_state[StateIndex.YAW],
            self._target_state[StateIndex.HEALTH],
        ]

    def make_tacviewLog(self):
        os.makedirs(self._artifacts_dir, exist_ok=True)
        timestamp = datetime.datetime.today()
        ownship_filename = (
            f"{timestamp.year}_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}_{timestamp.second}_ownship_(F-16)[Blue].csv"
        )
        target_filename = (
            f"{timestamp.year}_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}_{timestamp.second}_target_(F-16)[Red].csv"
        )
        summary_filename = (
            f"{timestamp.year}_{timestamp.month}_{timestamp.day}_{timestamp.hour}_{timestamp.minute}_{timestamp.second}_summary.json"
        )
        self._write_log(os.path.join(self._artifacts_dir, ownship_filename), self.ownship_log)
        self._write_log(os.path.join(self._artifacts_dir, target_filename), self.target_log)
        self._write_log_summary(os.path.join(self._artifacts_dir, summary_filename))

    def _write_log(self, path: str, entries: List[List[float]]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(
                "Time,Longitude,Latitude,Altitude,Roll (deg),Pitch (deg),"
                "Yaw (deg),Health\n"
            )
            for step, item in enumerate(entries):
                time_value = np.floor(self._delta_t * step * 10000) / 10000
                health = item[6] if len(item) > 6 else ""
                file.write(
                    f"{time_value},{item[1]},{item[0]},{item[2]},"
                    f"{item[3]},{item[4]},{item[5]},{health}\n"
                )

    def _write_log_summary(self, path: str) -> None:
        summary = {
            "end_condition": self.info.get("end_condition", ""),
            "outcome": self.info.get("outcome", ""),
            "ownship_health": self.info.get("ownship_health", None),
            "target_health": self.info.get("target_health", None),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    def render(self):
        return None

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        providers = (
            getattr(self, "_ownship_action_provider", None),
            getattr(self, "_target_action_provider", None),
        )
        for provider in providers:
            if provider is not None:
                try:
                    provider.close()
                except Exception:
                    pass
        sim_ai_pairs = (
            (getattr(self, "_sim", None), getattr(self, "_ownship_ai", None)),
            (getattr(self, "_target_sim", None), getattr(self, "_target_ai", None)),
        )
        for sim, ai in sim_ai_pairs:
            if (
                ai is not None
                and sim is not None
                and getattr(sim, "_model", None) is not None
            ):
                try:
                    ai.RemoveBT(sim.get_model().fighterID)
                except Exception:
                    pass
        battle_space_id = getattr(self, "battle_space_id", None)
        if battle_space_id is not None:
            try:
                JSBSimWrapper.RemoveSpace(battle_space_id)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["DogFightEnv", "normalize", "FEET_TO_METER", "METER_TO_FEET"]
