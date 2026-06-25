"""Curriculum stage definitions and advancement logic for dogfight RL training.

Stages (in order):
  0  flight_survival     — stay airborne, throttle control
  1  target_pursuit      — orient and close on fixed target
  2  wez_approach        — enter WEZ against loitering target
  3  autopilot_pursuit   — pursue a moving autopilot target
  4+ two_circle_headon   — alpha-based head-on curriculum stages
  N  full_dogfight       — full engagement vs. behavior-tree opponent
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


# ── Stage definition ──────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    index: int
    name: str
    description: str
    target_mode: str
    episode_step_limit: int     # steps per episode
    max_iterations: int         # hard cap; auto-advance may exit earlier
    checkpoint_interval: int    # save native checkpoint every N iterations
    reward_overrides: dict[str, Any]
    randomization: dict[str, Any]   # merged into env_config["ownship_randomization"]
    advance_conditions: dict[str, Any]  # metric key → threshold
    advance_window: int = 10    # rolling average window for advancement check
    env_overrides: dict[str, Any] = field(default_factory=dict)


# ── Stage catalogue ───────────────────────────────────────────────────────────

def get_stages() -> list[CurriculumStage]:
    """Return the ordered list of curriculum stages."""
    stages = [
        # ── Stage 0: Flight Survival ──────────────────────────────────────
        CurriculumStage(
            index=0,
            name="flight_survival",
            description="Stay airborne and develop throttle control. Target is stationary.",
            target_mode="fixed",
            episode_step_limit=3600,    # 60 s
            max_iterations=200,
            checkpoint_interval=10,
            reward_overrides={
                "survival_bonus": 0.05,         # +0.05 every step alive
                "pursuit_scale": 0.0,            # no pursuit required
                "damage_scale": 0.0,             # no damage reward
                "low_altitude_penalty": 1.0,     # 10× normal → hard boundary
                "win_reward": 0.0,
                "loss_reward": -50.0,            # crash penalty only
                "draw_reward": 0.0,
            },
            randomization={
                "enabled": True,
                "radius": 0.0,
                "r_roll": 5.0,
                "r_pitch": 5.0,
                "r_heading": 15.0,              # mild heading variation
            },
            advance_conditions={
                "crash_rate_max": 0.20,         # crash < 20 %
            },
        ),

        # ── Stage 1: Target Pursuit ───────────────────────────────────────
        CurriculumStage(
            index=1,
            name="target_pursuit",
            description="Orient nose toward and close on a fixed target.",
            target_mode="fixed",
            episode_step_limit=7200,    # 120 s
            max_iterations=300,
            checkpoint_interval=10,
            reward_overrides={
                "survival_bonus": 0.01,
                "pursuit_scale": 0.5,
                "pursuit_half_angle_deg": 60.0,  # wider gradient
                "pursuit_range_m": 8000.0,        # activates earlier
                "damage_scale": 0.0,
                "low_altitude_penalty": 0.3,
                "win_reward": 20.0,
                "loss_reward": -50.0,
                "draw_reward": 0.0,
            },
            randomization={
                "enabled": True,
                "radius": 500.0,
                "r_roll": 10.0,
                "r_pitch": 5.0,
                "r_heading": 60.0,
            },
            advance_conditions={
                "ep_min_distance_max": 2000.0,  # closes within 2 km
                "crash_rate_max": 0.15,
            },
        ),

        # ── Stage 2: WEZ Approach ─────────────────────────────────────────
        CurriculumStage(
            index=2,
            name="wez_approach",
            description="Enter WEZ against a loitering (non-threatening) target.",
            target_mode="loiter",
            episode_step_limit=10800,   # 180 s
            max_iterations=400,
            checkpoint_interval=10,
            reward_overrides={
                "survival_bonus": 0.0,
                "pursuit_scale": 0.3,
                "pursuit_half_angle_deg": 30.0,
                "pursuit_range_m": 3000.0,
                "damage_scale": 20.0,
                "low_altitude_penalty": 0.1,
                "win_reward": 100.0,
                "loss_reward": -100.0,
                "draw_reward": -30.0,
            },
            randomization={
                "enabled": True,
                "radius": 1000.0,
                "r_roll": 10.0,
                "r_pitch": 5.0,
                "r_heading": 120.0,
            },
            advance_conditions={
                # Either win rate meets threshold OR consistent WEZ contact
                "win_rate_min": 0.10,
                "ep_wez_steps_min": 10.0,      # OR-condition handled in check fn
            },
        ),

        # ── Stage 3: Autopilot Pursuit ────────────────────────────────────
        CurriculumStage(
            index=3,
            name="autopilot_pursuit",
            description="Pursue a moving target with fixed heading, altitude, and speed.",
            target_mode="autopilot",
            episode_step_limit=14400,   # 240 s
            max_iterations=500,
            checkpoint_interval=10,
            reward_overrides={
                "survival_bonus": 0.0,
                "pursuit_scale": 0.25,
                "pursuit_half_angle_deg": 25.0,
                "pursuit_range_m": 3500.0,
                "damage_scale": 20.0,
                "low_altitude_penalty": 0.1,
                "win_reward": 100.0,
                "loss_reward": -100.0,
                "draw_reward": -30.0,
            },
            randomization={
                "enabled": True,
                "radius": 1500.0,
                "r_roll": 12.0,
                "r_pitch": 8.0,
                "r_heading": 150.0,
            },
            advance_conditions={
                "ep_min_distance_max": 1500.0,
                "ep_wez_steps_min": 5.0,
                "crash_rate_max": 0.20,
            },
        ),

    ]
    two_circle_stages = _build_two_circle_headon_stages(start_index=len(stages))
    return stages + two_circle_stages + [
        CurriculumStage(
            index=len(stages) + len(two_circle_stages),
            name="full_dogfight",
            description="Full engagement against active behavior-tree opponent.",
            target_mode="behavior_tree",
            episode_step_limit=18000,   # 300 s
            max_iterations=1000,
            checkpoint_interval=10,
            reward_overrides={},        # use default reward (no overrides)
            randomization={
                "enabled": True,
                "radius": 2000.0,
                "r_roll": 15.0,
                "r_pitch": 10.0,
                "r_heading": 180.0,
            },
            advance_conditions={},      # no automatic advancement (final stage)
        ),
    ]


def _build_two_circle_headon_stages(start_index: int) -> list[CurriculumStage]:
    """Build alpha-based two-circle head-on curriculum stages."""
    stages = []
    for offset, alpha_deg in enumerate((0, 20, 40, 60, 80, 100, 120, 140, 160, 180)):
        stages.append(
            CurriculumStage(
                index=start_index + offset,
                name=f"two_circle_headon_a{alpha_deg:03d}",
                description=(
                    f"Two-circle head-on curriculum at alpha={alpha_deg} deg."
                ),
                target_mode="behavior_tree",
                episode_step_limit=18000,
                max_iterations=200,
                checkpoint_interval=10,
                reward_overrides={},
                randomization={"enabled": False},
                advance_conditions={
                    "win_rate_min": 0.70,
                    "crash_rate_max": 0.30,
                },
                advance_window=10,
                env_overrides={
                    "initial_scenario": {
                        "mode": "two_circle_headon",
                        "alpha_deg": float(alpha_deg),
                    },
                    "geometry_guard": {
                        "enabled": True,
                        "mode": "two_circle_headon",
                        "alpha_deg": float(alpha_deg),
                    },
                },
            )
        )
    return stages


# ── Config builder ────────────────────────────────────────────────────────────

def build_stage_env_config(base_config: dict, stage: CurriculumStage) -> dict:
    """Merge stage-specific overrides into a deep copy of base_config."""
    cfg = copy.deepcopy(base_config)
    cfg["target_mode"] = stage.target_mode
    cfg["episode_step_limit"] = stage.episode_step_limit
    # Reward overrides (only the keys listed; others keep base values)
    for k, v in stage.reward_overrides.items():
        cfg["reward"][k] = v
    # Position randomization
    if stage.randomization:
        cfg["ownship_randomization"] = {
            **cfg.get("ownship_randomization", {}),
            **stage.randomization,
        }
    if stage.env_overrides:
        _deep_update(cfg, stage.env_overrides)
    return cfg


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge stage env overrides in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


# ── Advancement logic ─────────────────────────────────────────────────────────

def check_advancement(stage: CurriculumStage, metric_window: list[dict]) -> tuple[bool, str]:
    """Evaluate whether the stage's advance_conditions are met.

    Args:
        stage: The current CurriculumStage.
        metric_window: List of per-iteration metric dicts (most recent last).
            Dicts may contain "n/a" strings for unavailable metrics.

    Returns:
        (should_advance, reason_string)
    """
    if not stage.advance_conditions or not metric_window:
        return False, ""

    # Compute rolling averages, skipping "n/a" entries
    averages: dict[str, float] = {}
    for key in stage.advance_conditions:
        metric_key = _condition_key_to_metric(key)
        values = [
            m[metric_key]
            for m in metric_window
            if isinstance(m.get(metric_key), (int, float))
        ]
        if values:
            averages[metric_key] = sum(values) / len(values)

    # WEZ approach special: win_rate_min OR ep_wez_steps_min (either condition)
    if stage.name == "wez_approach":
        win_avg   = averages.get("win_rate", None)
        wez_avg   = averages.get("ep_wez_steps", None)
        win_thr   = stage.advance_conditions.get("win_rate_min")
        wez_thr   = stage.advance_conditions.get("ep_wez_steps_min")
        if win_thr is not None and win_avg is not None and win_avg >= win_thr:
            return True, f"win_rate={win_avg:.3f} >= {win_thr}"
        if wez_thr is not None and wez_avg is not None and wez_avg >= wez_thr:
            return True, f"ep_wez_steps={wez_avg:.1f} >= {wez_thr}"
        return False, ""

    # General: ALL conditions must be met simultaneously
    reasons = []
    for cond_key, threshold in stage.advance_conditions.items():
        metric_key = _condition_key_to_metric(cond_key)
        avg = averages.get(metric_key)
        if avg is None:
            return False, ""   # metric not yet available → wait

        if cond_key.endswith("_max") and avg > threshold:
            return False, f"{metric_key}={avg:.4f} still above {threshold}"
        if cond_key.endswith("_min") and avg < threshold:
            return False, f"{metric_key}={avg:.4f} still below {threshold}"
        reasons.append(f"{metric_key}={avg:.4f}")

    reason_str = ", ".join(reasons)
    return True, reason_str


def _condition_key_to_metric(cond_key: str) -> str:
    """Strip _max/_min suffix → metric key used in metric_history dicts."""
    for suffix in ("_max", "_min"):
        if cond_key.endswith(suffix):
            return cond_key[: -len(suffix)]
    return cond_key
