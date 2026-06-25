from __future__ import annotations

import copy

FEET_TO_METER = 0.30480
METER_TO_FEET = 3.28084
KNOT_TO_METER_SEC = 0.51444

DEFAULT_ENV_CONFIG = {
    "sim_hz": 60,
    "step_ratio": None,
    "delta": None,
    "time_step": None,
    "max_engage_time": 300.0,
    "episode_step_limit": 18000,
    "min_altitude": 300.0,
    "observation_mode": "classic12",
    "ownship_control_mode": "rl",
    "target_mode": "behavior_tree",
    "ownship_behavior_dll": None,
    "target_behavior_dll": "AIP_BASE_target.dll",
    "target_loiter": {"enabled": True, "bank": 30.0, "pitch": 0.0},
    "target_autopilot": {"heading_cmd": 180.0, "altitude_cmd": 7000.0, "speed_cmd": 250.0},
    "reward": {
        "step_penalty": -0.01,
        "damage_scale": 20.0,
        "pursuit_scale": 0.3,
        "pursuit_half_angle_deg": 30.0,
        "pursuit_range_m": 3000.0,
        "low_altitude_penalty": 0.1,
        "win_reward": 100.0,
        "loss_reward": -100.0,
        "draw_reward": -30.0,
        "guard_fail_penalty": -50.0,
    },
    "wez": {
        "angle_deg": 2.0,
        "min_range_m": 500 * FEET_TO_METER,
        "max_range_m": 3000 * FEET_TO_METER,
    },
    "ownship": [1000.0, 0.0, -7000.0, 0.0, 0.0, 0.0, 300.0],
    "target": [6000.0, 0.0, -7000.0, 0.0, 0.0, 180.0, 300.0],
    "artifacts_dir": "artifacts/logs",
    # Per-episode position randomization (used by curriculum; disabled by default)
    "ownship_randomization": {
        "enabled": False,
        "radius": 0.0,      # NED position scatter radius (meters)
        "r_roll": 0.0,      # roll scatter (degrees)
        "r_pitch": 0.0,     # pitch scatter (degrees)
        "r_heading": 0.0,   # heading scatter (degrees)
    },
    "initial_scenario": {
        "mode": "default",
        "legacy_use_random_scenario": True,
        "legacy_use_first_scenario_only": False,
        "legacy_scenario_indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "legacy_randomization": {
            "aircraft_radius_m": 100.0,
            "roll_deg": 5.0,
            "pitch_deg": 5.0,
            "heading_deg": 5.0,
            "shared_n_m": 4000.0,
            "shared_e_m": 4000.0,
            "shared_d_m": 4000.0,
            "target_distance_n_m": 300.0,
            "speed_mps": 50.0,
            "loiter_bank_deg_range": [40.0, 70.0],
        },
        "alpha_deg": 0.0,
        "turn_diameter_ft": 6000.0,
        "separation_jitter_ft": [3000.0, 6000.0],
        "center_n_m": 3500.0,
        "center_e_m": 0.0,
        "altitude_m": 7000.0,
        "speed_mps_range": [250.0, 300.0],
        "vertical_pitch_choices_deg": [0.0, 10.0, -10.0],
        "roll_range_deg": [0.0, 180.0],
        "side_choices": [-1.0, 1.0],
    },
    "geometry_guard": {
        "enabled": False,
        "mode": "two_circle_headon",
        "alpha_deg": 0.0,
        "early_alpha_max_deg": 80.0,
        "mid_alpha_max_deg": 140.0,
        "crossed_ata_limit_deg": 90.0,
        "turn_margin_deg": 10.0,
    },
}


def merge_env_config(env_config: dict | None) -> dict:
    merged = copy.deepcopy(DEFAULT_ENV_CONFIG)
    if not env_config:
        return merged

    _deep_update(merged, env_config)
    return merged


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge env config dictionaries in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
