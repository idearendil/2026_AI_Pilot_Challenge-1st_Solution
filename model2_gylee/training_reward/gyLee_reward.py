"""Outcome-dominant reward profiles for the ver06 Phase1-iter75 branch.

Three named profiles are kept in this module:

``phase1``
    Historical tmLee terminal/target-damage/distance terms plus the small,
    symmetric broad-ATA occupancy term used by the completed Phase 1 run.

``phase2``
    HP destruction gives +5/-5, timeout HP comparison gives +5/-5 and an
    exact HP tie gives -4, altitude exits are asymmetric (-20 ownship / +5
    target), and symmetric net-damage reward is combined with a symmetric
    broad ATA-control occupancy term. Lethal-step damage is included.
    Distance shaping is disabled. The actual time-gated
    1/2/3-degree weapon damage remains in ``my_observation.py``; it is not
    duplicated as a reward.

``phase3``
    Outcome-dominant ver09 profile. Every win is +100 regardless of whether it
    came from one HP at timeout, destruction, or an opponent simulator/ground
    loss. Every loss is -100, and a draw is -80. Raw damage and distance reward
    are disabled. A bounded HP-state term (maximum +/-5 over 200 seconds)
    rewards maintaining any strict lead equally: a one-point and a 99-point
    lead receive the same rate, while a deficit penalty smoothly saturates.
    Symmetric 0--60 degree ATA guidance is gamma-correct potential shaping,
    range-gated by the real weapon schedule, so holding or oscillating geometry
    cannot accumulate positive reward. Predictive ground risk is capped at -30;
    an altitude termination pays only the remainder required to make the
    safety-plus-terminal consequence exactly -100.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dogfight.sim.state_schema import StateIndex
from claude_code.my_observation import (
    MIN_DAMAGE_RANGE_FT,
    TIER1_MAX_RANGE_FT,
    TIER2_MAX_RANGE_FT,
    TIER2_START_SEC,
    TIER3_MAX_RANGE_FT,
    TIER3_START_SEC,
)


_FEET_TO_METER = 0.3048
_SAFETY_START_ALTITUDE_M = 1750.0 * _FEET_TO_METER

_COMMON_CONFIG = {
    "win_reward": 10.0,
    "loss_reward": -10.0,
    "ownship_alt_reward": -10.0,
    "target_alt_reward": 10.0,
    "damage_scale": 10.0,
    "target_damage_weight": 1.0,
    "include_terminal_damage": False,
    "timeout_reward": 0.0,
    "timeout_win_reward": 0.0,
    "timeout_loss_reward": 0.0,
    "geometry_reference_duration_sec": 200.0,
    "simulation_hz": 60,
    "step_ratio": 6,
    "dt_per_step": 0.1,
}

PHASE1_REWARD_CONFIG = {
    **_COMMON_CONFIG,
    "reward_profile": "phase1",
    "mode": "gyLee_tmlee_phase1",
    "ownship_damage_weight": 0.0,
    "geometry_reward_variant": "broad_ata_control",
    "geometry_reward_component": "ata",
    "geometry_episode_budget": 1.0,
    "broad_ata_full_control_angle_deg": 30.0,
    "distance_reward_variant": "legacy_linear",
    "distance_reward_scale": 0.001,
}

PHASE2_REWARD_CONFIG = {
    **_COMMON_CONFIG,
    "reward_profile": "phase2",
    "mode": "gyLee_tmlee_phase2",
    # Phase 2 only: HP destruction and timeout HP comparison use symmetric
    # +/-5 outcome rewards. Low-altitude exits retain dedicated rewards and
    # take precedence, so they do not stack with HP-based outcome rewards.
    "win_reward": 5.0,
    "loss_reward": -5.0,
    "ownship_alt_reward": -20.0,
    "target_alt_reward": 5.0,
    "timeout_win_reward": 5.0,
    "timeout_loss_reward": -5.0,
    "timeout_reward": -4.0,
    # Net HP exchange is symmetric and the lethal damage step is included.
    # A full enemy/own HP loss contributes +10/-10 dense reward.
    "ownship_damage_weight": 1.0,
    "include_terminal_damage": True,
    # Signed broad-control occupancy: own ATA control minus enemy ATA control.
    # Each side receives full control score at 0--30 degrees; the score then
    # falls linearly over the remaining 150 degrees and reaches zero at 180.
    # Therefore reciprocal/equal geometry cancels, being attacked is negative,
    # and 200 seconds of maximum advantage sums to +/-25.
    "geometry_reward_variant": "broad_ata_control",
    "geometry_reward_component": "ata",
    "geometry_episode_budget": 25.0,
    "broad_ata_full_control_angle_deg": 30.0,
    "distance_reward_variant": "disabled",
    "distance_reward_scale": 0.0,
}

PHASE3_REWARD_CONFIG = {
    **PHASE2_REWARD_CONFIG,
    "reward_profile": "phase3",
    "mode": (
        "ver09_outcome100_draw80_leadhold5_"
        "symmetric_linearata60_potential5_safety30"
    ),
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "ownship_alt_reward": -100.0,
    "target_alt_reward": 100.0,
    "timeout_win_reward": 100.0,
    "timeout_loss_reward": -100.0,
    # Keep the generic draw field aligned with the custom timeout/terminal
    # draw value so run metadata cannot report a stale environment default.
    "draw_reward": -80.0,
    "timeout_reward": -80.0,
    "damage_scale": 0.0,
    "target_damage_weight": 0.0,
    "ownship_damage_weight": 0.0,
    "include_terminal_damage": False,
    # Bounded HP-state occupancy. Any strict lead gets the same +1 score,
    # independent of margin. A deficit is -tanh(deficit_points/15), a tie is 0.
    # The maximum 200-second contribution is only +/-5, so no losing episode
    # can offset the -100 outcome with a transient or narrow lead.
    "hp_advantage_reward_variant": "bounded_lead_hold",
    "hp_lead_episode_budget": 5.0,
    "hp_deficit_scale_points": 15.0,
    "hp_health_to_points_scale": 100.0,
    "hp_advantage_reference_duration_sec": 200.0,
    "hp_advantage_tie_tolerance": 1.0e-9,
    # Symmetric linear 60-degree ATA potential. The score is one only at 0 deg,
    # zero at/beyond 60 deg, and is multiplied by the actual weapon-range gate.
    # r = gamma * Phi(next) - Phi(previous), with terminal Phi=0. Therefore it
    # accelerates credit assignment without changing the outcome-optimal policy
    # or paying every step merely for holding favorable geometry.
    "geometry_reward_variant": "symmetric_cone_range_gated_potential",
    "geometry_reward_component": "ata",
    "geometry_episode_budget": 5.0,
    "geometry_potential_gamma": 0.999,
    # Legacy broad-ATA key retained for CLI/config compatibility; this variant
    # uses geometry_cone_deg and has no nonzero full-credit plateau.
    "broad_ata_full_control_angle_deg": 0.0,
    "geometry_cone_deg": 60.0,
    "own_control_full_angle_deg": 5.0,
    "own_control_outer_angle_deg": 30.0,
    "enemy_threat_full_angle_deg": 3.0,
    "enemy_threat_outer_angle_deg": 10.0,
    "enemy_threat_multiplier": 1.0,
    "geometry_min_range_ft": MIN_DAMAGE_RANGE_FT,
    "geometry_range_buffer_ft": 1000.0,
    "distance_reward_variant": "disabled",
    "distance_reward_scale": 0.0,
    # Predictive bounded safety potential:
    #   h_projected = h - clamp(v_descent, 0, cap) * horizon
    #   phi = -budget * clamp((start-h_projected)/(start-hard_deck), 0, 1)^2
    #   r_safety = min(0, phi(current) - phi(previous))
    # It starts with zero slope, produces no positive recovery reward, returns
    # zero while risk is unchanged, and is robust to one-step FDM spikes
    # through EMA filtering and a descent-rate cap. Episode safety penalties
    # are capped by low_altitude_potential_budget.
    "low_altitude_penalty_variant": (
        "one_sided_predictive_quadratic_potential"
    ),
    "low_altitude_penalty_start_m": _SAFETY_START_ALTITUDE_M,
    "low_altitude_min_m": 300.0,
    "low_altitude_potential_budget": 30.0,
    "low_altitude_prediction_horizon_sec": 3.0,
    "low_altitude_descent_rate_cap_mps": 150.0,
    "low_altitude_descent_rate_ema_alpha": 0.25,
}

REWARD_PROFILES = {
    "phase1": PHASE1_REWARD_CONFIG,
    "phase2": PHASE2_REWARD_CONFIG,
    "phase3": PHASE3_REWARD_CONFIG,
}

# ``load_reward_hook`` discovers this name.  Keeping Phase 1 as the module
# default preserves existing launch commands and makes future Phase 1 retrains
# reproduce the original reward without Phase 2 terms leaking into them.
MY_REWARD_CONFIG = copy.deepcopy(PHASE1_REWARD_CONFIG)

_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"
_TARGET_FORCED_LOSS_ENDS = {
    _TARGET_ALT_END,
    "Target FDM output Fall",
}
_OWNSHIP_FORCED_LOSS_ENDS = {
    _OWNSHIP_ALT_END,
    "FDM Update Fail",
    "Ownship FDM output Fall",
    "fuel fail",
    "two circle headon guard fail",
}


@dataclass
class RewardState:
    """Per-environment state for distance and duration-based shaping."""

    previous_distance_m: float | None = None
    previous_sim_time: float | None = None
    previous_altitude_m: float | None = None
    filtered_descent_rate_mps: float = 0.0
    previous_safety_potential: float | None = None
    previous_geometry_potential: float | None = None
    safety_penalty_total: float = 0.0
    hp_deficit_debt: float = 0.0
    opponent_kind: str = ""
    step_count: int = 0


def get_reward_config(profile: str) -> dict:
    """Return an isolated, validated reward config for a named phase."""

    profile_name = str(profile).strip().lower()
    if profile_name not in REWARD_PROFILES:
        raise ValueError(
            f"unknown gyLee reward profile {profile!r}; "
            f"expected one of {sorted(REWARD_PROFILES)}"
        )
    config = copy.deepcopy(REWARD_PROFILES[profile_name])
    _validate_config(config)
    return config


def _merged_config(overrides: dict | None = None) -> dict:
    override_values = dict(overrides or {})
    profile = str(
        override_values.get(
            "reward_profile",
            MY_REWARD_CONFIG["reward_profile"],
        )
    )
    config = get_reward_config(profile)
    config.update(override_values)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    profile = str(config["reward_profile"])
    variant = str(config["geometry_reward_variant"])
    component = str(config["geometry_reward_component"])
    if profile not in REWARD_PROFILES:
        raise ValueError(f"unsupported reward_profile: {profile!r}")
    supported_geometry_variants = {
        "broad_ata_control",
        "broad_ata_control_range_gated",
        "cone30_range_gated_control",
        "hp_conditioned_cone_range_gated_control",
        "outcome_control_threat_range_gated",
        "symmetric_cone_range_gated_potential",
    }
    if variant not in supported_geometry_variants:
        raise ValueError(f"unsupported geometry_reward_variant: {variant!r}")
    expected_component = (
        "weapon"
        if variant == "outcome_control_threat_range_gated"
        else "ata"
    )
    if component != expected_component:
        raise ValueError(
            f"{variant} must use geometry component {expected_component!r}"
        )

    budget = float(config["geometry_episode_budget"])
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError(
            "geometry_episode_budget must be finite and non-negative"
        )
    reference_duration = float(config["geometry_reference_duration_sec"])
    if not math.isfinite(reference_duration) or reference_duration <= 0.0:
        raise ValueError(
            "geometry_reference_duration_sec must be finite and positive"
        )
    ownship_damage_weight = float(config["ownship_damage_weight"])
    if not math.isfinite(ownship_damage_weight) or ownship_damage_weight < 0.0:
        raise ValueError(
            "ownship_damage_weight must be finite and non-negative"
        )
    if not isinstance(config["include_terminal_damage"], bool):
        raise ValueError("include_terminal_damage must be a bool")
    for reward_key in (
        "win_reward",
        "loss_reward",
        "ownship_alt_reward",
        "target_alt_reward",
        "timeout_reward",
        "timeout_win_reward",
        "timeout_loss_reward",
    ):
        if not math.isfinite(float(config[reward_key])):
            raise ValueError(f"{reward_key} must be finite")
    full_control = float(config["broad_ata_full_control_angle_deg"])
    if not math.isfinite(full_control) or not 0.0 <= full_control < 180.0:
        raise ValueError(
            "broad_ata_full_control_angle_deg must be in [0, 180)"
        )
    if variant in {
        "broad_ata_control_range_gated",
        "cone30_range_gated_control",
        "hp_conditioned_cone_range_gated_control",
        "symmetric_cone_range_gated_potential",
    }:
        min_range_ft = float(config["geometry_min_range_ft"])
        range_buffer_ft = float(config["geometry_range_buffer_ft"])
        if variant in {
            "cone30_range_gated_control",
            "hp_conditioned_cone_range_gated_control",
            "symmetric_cone_range_gated_potential",
        }:
            cone_deg = float(config["geometry_cone_deg"])
            if not math.isfinite(cone_deg) or not 0.0 < cone_deg < 180.0:
                raise ValueError("geometry_cone_deg must be in (0, 180)")
        if not math.isfinite(min_range_ft) or min_range_ft < 0.0:
            raise ValueError(
                "geometry_min_range_ft must be finite and non-negative"
            )
        if not math.isfinite(range_buffer_ft) or range_buffer_ft <= 0.0:
            raise ValueError(
                "geometry_range_buffer_ft must be finite and positive"
            )
    if variant == "hp_conditioned_cone_range_gated_control":
        nonleading_scale = float(
            config["geometry_nonleading_attack_scale"]
        )
        taper_points = float(config["geometry_lead_taper_hp_points"])
        minimum_scale = float(config["geometry_lead_min_attack_scale"])
        if not math.isfinite(nonleading_scale) or nonleading_scale <= 0.0:
            raise ValueError(
                "geometry_nonleading_attack_scale must be finite and positive"
            )
        if not math.isfinite(taper_points) or taper_points <= 0.0:
            raise ValueError(
                "geometry_lead_taper_hp_points must be finite and positive"
            )
        if (
            not math.isfinite(minimum_scale)
            or not 0.0 <= minimum_scale <= 1.0
        ):
            raise ValueError(
                "geometry_lead_min_attack_scale must be in [0, 1]"
            )
    if variant == "symmetric_cone_range_gated_potential":
        potential_gamma = float(config["geometry_potential_gamma"])
        if (
            not math.isfinite(potential_gamma)
            or not 0.0 < potential_gamma <= 1.0
        ):
            raise ValueError("geometry_potential_gamma must be in (0, 1]")
    if variant == "outcome_control_threat_range_gated":
        own_full = float(config["own_control_full_angle_deg"])
        own_outer = float(config["own_control_outer_angle_deg"])
        enemy_full = float(config["enemy_threat_full_angle_deg"])
        enemy_outer = float(config["enemy_threat_outer_angle_deg"])
        multiplier = float(config["enemy_threat_multiplier"])
        if not 0.0 <= own_full < own_outer < 180.0:
            raise ValueError(
                "own control angles must satisfy 0 <= full < outer < 180"
            )
        if not 0.0 <= enemy_full < enemy_outer < 180.0:
            raise ValueError(
                "enemy threat angles must satisfy 0 <= full < outer < 180"
            )
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("enemy_threat_multiplier must be positive")
        min_range_ft = float(config["geometry_min_range_ft"])
        range_buffer_ft = float(config["geometry_range_buffer_ft"])
        if not math.isfinite(min_range_ft) or min_range_ft < 0.0:
            raise ValueError(
                "geometry_min_range_ft must be finite and non-negative"
            )
        if not math.isfinite(range_buffer_ft) or range_buffer_ft <= 0.0:
            raise ValueError(
                "geometry_range_buffer_ft must be finite and positive"
            )

    distance_variant = str(config["distance_reward_variant"])
    if distance_variant not in {"legacy_linear", "disabled"}:
        raise ValueError(
            "distance_reward_variant must be 'legacy_linear' or 'disabled'"
        )
    distance_scale = float(config["distance_reward_scale"])
    if not math.isfinite(distance_scale) or distance_scale < 0.0:
        raise ValueError(
            "distance_reward_scale must be finite and non-negative"
        )
    if distance_variant == "disabled" and distance_scale != 0.0:
        raise ValueError(
            "disabled distance reward must use distance_reward_scale=0"
        )
    if profile == "phase3":
        if (
            str(config["hp_advantage_reward_variant"])
            != "bounded_lead_hold"
        ):
            raise ValueError(
                "phase3 hp_advantage_reward_variant must be "
                "'bounded_lead_hold'"
            )
        hp_lead_budget = float(config["hp_lead_episode_budget"])
        hp_deficit_scale = float(config["hp_deficit_scale_points"])
        hp_points_scale = float(config["hp_health_to_points_scale"])
        hp_duration = float(config["hp_advantage_reference_duration_sec"])
        hp_tolerance = float(config["hp_advantage_tie_tolerance"])
        if not math.isfinite(hp_lead_budget) or hp_lead_budget <= 0.0:
            raise ValueError(
                "hp_lead_episode_budget must be finite and positive"
            )
        if not math.isfinite(hp_deficit_scale) or hp_deficit_scale <= 0.0:
            raise ValueError(
                "hp_deficit_scale_points must be finite and positive"
            )
        if not math.isfinite(hp_points_scale) or hp_points_scale <= 0.0:
            raise ValueError(
                "hp_health_to_points_scale must be finite and positive"
            )
        if not math.isfinite(hp_duration) or hp_duration <= 0.0:
            raise ValueError(
                "hp_advantage_reference_duration_sec must be finite and positive"
            )
        if not math.isfinite(hp_tolerance) or hp_tolerance < 0.0:
            raise ValueError(
                "hp_advantage_tie_tolerance must be finite and non-negative"
            )
        if (
            str(config["low_altitude_penalty_variant"])
            != "one_sided_predictive_quadratic_potential"
        ):
            raise ValueError(
                "phase3 low_altitude_penalty_variant must be "
                "'one_sided_predictive_quadratic_potential'"
            )
        safety_start = float(config["low_altitude_penalty_start_m"])
        safety_min = float(config["low_altitude_min_m"])
        safety_budget = float(config["low_altitude_potential_budget"])
        if (
            not math.isfinite(safety_start)
            or not math.isfinite(safety_min)
            or safety_start <= safety_min
        ):
            raise ValueError(
                "low_altitude_penalty_start_m must be finite and greater "
                "than low_altitude_min_m"
            )
        if not math.isfinite(safety_budget) or safety_budget < 0.0:
            raise ValueError(
                "low_altitude_potential_budget must be finite and "
                "non-negative"
            )
        horizon = float(config["low_altitude_prediction_horizon_sec"])
        descent_cap = float(config["low_altitude_descent_rate_cap_mps"])
        ema_alpha = float(config["low_altitude_descent_rate_ema_alpha"])
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(
                "low_altitude_prediction_horizon_sec must be finite and "
                "non-negative"
            )
        if not math.isfinite(descent_cap) or descent_cap <= 0.0:
            raise ValueError(
                "low_altitude_descent_rate_cap_mps must be finite and "
                "positive"
            )
        if not math.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError(
                "low_altitude_descent_rate_ema_alpha must be in (0, 1]"
            )


def _state_value(state, index: StateIndex, default: float) -> float:
    """Read extended simulator fields while keeping unit tests state-minimal."""

    try:
        value = float(state[int(index)])
    except (IndexError, TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _ata_control_score(
    ata_deg: float,
    full_reward_angle_deg: float,
) -> float:
    """Symmetric broad-control score in ``[0, 1]``."""

    angle = abs(float(ata_deg))
    full_reward = float(full_reward_angle_deg)
    if not math.isfinite(angle):
        raise ValueError("ATA angle must be finite")
    if not math.isfinite(full_reward) or not 0.0 <= full_reward < 180.0:
        raise ValueError("full ATA reward angle must be in [0, 180)")
    if angle <= full_reward:
        return 1.0
    return max(
        0.0,
        min(1.0, (180.0 - angle) / (180.0 - full_reward)),
    )


def _ata_control_advantage(
    own_ata_deg: float,
    enemy_ata_deg: float,
    config: dict,
) -> tuple[float, dict]:
    """Signed, own/enemy-symmetric broad ATA advantage."""

    full_control_angle = float(
        config["broad_ata_full_control_angle_deg"]
    )
    own_control = _ata_control_score(own_ata_deg, full_control_angle)
    enemy_control = _ata_control_score(enemy_ata_deg, full_control_angle)
    advantage = own_control - enemy_control
    diagnostics = {
        "own_ata_control": float(own_control),
        "enemy_ata_control": float(enemy_control),
        "ata_control_advantage": float(advantage),
    }
    return float(advantage), diagnostics


def _cone_control_score(ata_deg: float, cone_deg: float) -> float:
    """Linear ATA score: one at 0 deg, zero at and beyond ``cone_deg``."""

    angle = abs(float(ata_deg))
    cone = float(cone_deg)
    if not math.isfinite(angle):
        raise ValueError("ATA angle must be finite")
    if not math.isfinite(cone) or not 0.0 < cone < 180.0:
        raise ValueError("cone_deg must be in (0, 180)")
    return float(max(0.0, min(1.0, (cone - angle) / cone)))


def _hp_conditioned_attack_scale(
    own_hp: float,
    target_hp: float,
    config: dict,
) -> tuple[float, float]:
    """Return smooth own-attack weight and HP gap in 100-point units.

    A tie or deficit uses the configured acquisition multiplier. Any strict
    positive lead uses the original 1.0-to-floor linear taper. This keeps the
    requested 2x boost strictly on the non-leading side and leaves post-lead
    behavior unchanged.
    """

    hp_gap_points = (
        float(own_hp) - float(target_hp)
    ) * float(config["hp_health_to_points_scale"])
    taper_points = float(config["geometry_lead_taper_hp_points"])
    minimum_scale = float(config["geometry_lead_min_attack_scale"])
    if hp_gap_points <= 0.0:
        return float(
            config["geometry_nonleading_attack_scale"]
        ), float(hp_gap_points)
    progress = min(1.0, hp_gap_points / taper_points)
    attack_scale = 1.0 - (1.0 - minimum_scale) * progress
    return float(attack_scale), float(hp_gap_points)


def _plateau_cone_score(
    ata_deg: float,
    full_angle_deg: float,
    outer_angle_deg: float,
) -> float:
    """Return one inside ``full`` and taper linearly to zero at ``outer``."""

    angle = abs(float(ata_deg))
    full = float(full_angle_deg)
    outer = float(outer_angle_deg)
    if not math.isfinite(angle):
        raise ValueError("ATA angle must be finite")
    if not 0.0 <= full < outer < 180.0:
        raise ValueError("cone angles must satisfy 0 <= full < outer < 180")
    if angle <= full:
        return 1.0
    if angle >= outer:
        return 0.0
    return float((outer - angle) / (outer - full))


def _active_weapon_max_range_ft(sim_time_sec: float) -> float:
    """Return the same time-gated maximum range used by actual HP damage."""

    elapsed = float(sim_time_sec)
    if elapsed >= TIER3_START_SEC:
        return float(TIER3_MAX_RANGE_FT)
    if elapsed >= TIER2_START_SEC:
        return float(TIER2_MAX_RANGE_FT)
    return float(TIER1_MAX_RANGE_FT)


def _cone_range_gate(
    distance_ft: float,
    sim_time_sec: float,
    config: dict,
) -> tuple[float, dict]:
    """Gate ATA shaping to the weapon range plus a linear approach buffer."""

    distance = float(distance_ft)
    minimum = float(config["geometry_min_range_ft"])
    active_max = _active_weapon_max_range_ft(sim_time_sec)
    buffer_ft = float(config["geometry_range_buffer_ft"])
    outer = active_max + buffer_ft
    if distance < minimum or distance >= outer:
        gate = 0.0
    elif distance <= active_max:
        gate = 1.0
    else:
        gate = (outer - distance) / buffer_ft
    return float(gate), {
        "geometry_distance_ft": distance,
        "geometry_min_range_ft": minimum,
        "geometry_active_max_range_ft": active_max,
        "geometry_outer_range_ft": outer,
        "geometry_range_gate": float(gate),
    }


def _geometry_snapshot(
    ownship_state,
    target_state,
    geo_info,
    sim_time_sec: float,
    config: dict,
    own_attack_scale: float = 1.0,
):
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    own_ata = abs(float(
        geo_info._get_antenna_train_angle(
            ownship_state, target_state, False
        )
    ))
    enemy_ata = abs(float(
        geo_info._get_antenna_train_angle(
            target_state, ownship_state, False
        )
    ))
    variant = str(config["geometry_reward_variant"])
    if variant == "broad_ata_control":
        advantage, diagnostics = _ata_control_advantage(
            own_ata,
            enemy_ata,
            config,
        )
    elif variant == "broad_ata_control_range_gated":
        ungated_advantage, diagnostics = _ata_control_advantage(
            own_ata,
            enemy_ata,
            config,
        )
        range_gate, range_diagnostics = _cone_range_gate(
            distance_m / _FEET_TO_METER,
            sim_time_sec,
            config,
        )
        advantage = range_gate * ungated_advantage
        diagnostics = {
            **diagnostics,
            "ata_control_advantage_ungated": float(ungated_advantage),
            "ata_control_advantage": float(advantage),
            **range_diagnostics,
        }
    elif variant in {
        "cone30_range_gated_control",
        "symmetric_cone_range_gated_potential",
    }:
        cone_deg = float(config["geometry_cone_deg"])
        own_control = _cone_control_score(own_ata, cone_deg)
        enemy_control = _cone_control_score(enemy_ata, cone_deg)
        range_gate, range_diagnostics = _cone_range_gate(
            distance_m / _FEET_TO_METER,
            sim_time_sec,
            config,
        )
        advantage = range_gate * (own_control - enemy_control)
        diagnostics = {
            "own_ata_control": float(own_control),
            "enemy_ata_control": float(enemy_control),
            "ata_control_advantage_ungated": float(
                own_control - enemy_control
            ),
            "ata_control_advantage": float(advantage),
            "geometry_cone_deg": cone_deg,
            **range_diagnostics,
        }
    elif variant == "hp_conditioned_cone_range_gated_control":
        cone_deg = float(config["geometry_cone_deg"])
        own_control = _cone_control_score(own_ata, cone_deg)
        enemy_control = _cone_control_score(enemy_ata, cone_deg)
        attack_scale = float(own_attack_scale)
        max_attack_scale = max(
            1.0,
            float(config["geometry_nonleading_attack_scale"]),
        )
        if (
            not math.isfinite(attack_scale)
            or not 0.0 <= attack_scale <= max_attack_scale
        ):
            raise ValueError(
                "own_attack_scale must be finite and within the configured "
                "non-leading maximum"
            )
        ungated_advantage = attack_scale * own_control - enemy_control
        range_gate, range_diagnostics = _cone_range_gate(
            distance_m / _FEET_TO_METER,
            sim_time_sec,
            config,
        )
        advantage = range_gate * ungated_advantage
        diagnostics = {
            "own_ata_control": float(own_control),
            "enemy_ata_control": float(enemy_control),
            "own_ata_attack_scale": attack_scale,
            "ata_control_advantage_ungated": float(ungated_advantage),
            "ata_control_advantage": float(advantage),
            "geometry_cone_deg": cone_deg,
            **range_diagnostics,
        }
    elif variant == "outcome_control_threat_range_gated":
        own_control = _plateau_cone_score(
            own_ata,
            float(config["own_control_full_angle_deg"]),
            float(config["own_control_outer_angle_deg"]),
        )
        enemy_threat = _plateau_cone_score(
            enemy_ata,
            float(config["enemy_threat_full_angle_deg"]),
            float(config["enemy_threat_outer_angle_deg"]),
        )
        range_gate, range_diagnostics = _cone_range_gate(
            distance_m / _FEET_TO_METER,
            sim_time_sec,
            config,
        )
        threat_multiplier = float(config["enemy_threat_multiplier"])
        advantage = range_gate * (
            own_control - threat_multiplier * enemy_threat
        )
        diagnostics = {
            "own_control_score": float(own_control),
            "enemy_threat_score": float(enemy_threat),
            "enemy_threat_multiplier": threat_multiplier,
            "geometry_control_threat_ungated": float(
                own_control - threat_multiplier * enemy_threat
            ),
            "ata_control_advantage": float(advantage),
            "own_control_full_angle_deg": float(
                config["own_control_full_angle_deg"]
            ),
            "own_control_outer_angle_deg": float(
                config["own_control_outer_angle_deg"]
            ),
            "enemy_threat_full_angle_deg": float(
                config["enemy_threat_full_angle_deg"]
            ),
            "enemy_threat_outer_angle_deg": float(
                config["enemy_threat_outer_angle_deg"]
            ),
            **range_diagnostics,
        }
    else:  # Validated configs cannot reach this branch.
        raise ValueError(f"unsupported geometry_reward_variant: {variant!r}")
    diagnostics.update({
        "own_ata_deg": float(own_ata),
        "enemy_ata_deg": float(enemy_ata),
    })
    return distance_m, advantage, diagnostics


def _distance_reward(
    previous_distance_m: float,
    distance_m: float,
    config: dict,
) -> tuple[float, dict]:
    """Return the selected distance delta and compact diagnostics."""

    previous = float(previous_distance_m)
    current = float(distance_m)
    variant = str(config["distance_reward_variant"])
    if variant == "legacy_linear":
        scale = float(config["distance_reward_scale"])
        return (previous - current) * scale, {
            "distance_reward_variant_legacy_linear": 1.0,
            "distance_reward_scale": scale,
        }
    if variant == "disabled":
        return 0.0, {"distance_reward_variant_disabled": 1.0}
    raise ValueError(f"unsupported distance_reward_variant: {variant!r}")


def _hp_deficit_snapshot(
    own_hp: float,
    target_hp: float,
    config: dict,
) -> tuple[float, float, int]:
    """Return signed HP-point gap, deficit severity, and state code.

    State code is +1 for a strict HP lead, -1 for a strict deficit, and 0 for
    a tie. Severity is ``tanh(deficit_points / scale_points)`` and is therefore
    zero outside a deficit.
    """

    advantage = float(own_hp) - float(target_hp)
    tolerance = float(config["hp_advantage_tie_tolerance"])
    hp_gap_points = advantage * float(config["hp_health_to_points_scale"])
    if advantage > tolerance:
        return float(hp_gap_points), 0.0, 1
    if advantage < -tolerance:
        deficit_points = -hp_gap_points
        severity = math.tanh(
            deficit_points / float(config["hp_deficit_scale_points"])
        )
        return float(hp_gap_points), float(severity), -1
    return float(hp_gap_points), 0.0, 0


def _low_altitude_safety_potential(
    altitude_m: float,
    descent_rate_mps: float,
    config: dict,
) -> tuple[float, dict]:
    """Return the rate-aware bounded hard-deck potential and diagnostics."""

    if str(config["reward_profile"]) != "phase3":
        return 0.0, {}
    start_m = float(config["low_altitude_penalty_start_m"])
    min_m = float(config["low_altitude_min_m"])
    budget = float(config["low_altitude_potential_budget"])
    horizon_sec = float(config["low_altitude_prediction_horizon_sec"])
    descent_cap_mps = float(config["low_altitude_descent_rate_cap_mps"])
    bounded_descent_rate = min(
        descent_cap_mps,
        max(0.0, float(descent_rate_mps)),
    )
    projected_altitude_m = (
        float(altitude_m) - bounded_descent_rate * horizon_sec
    )
    severity = max(
        0.0,
        min(
            1.0,
            (start_m - projected_altitude_m) / (start_m - min_m),
        ),
    )
    potential = -budget * severity * severity
    return float(potential), {
        "ownship_altitude_m": float(altitude_m),
        "ownship_descent_rate_mps": float(bounded_descent_rate),
        "projected_ownship_altitude_m": float(projected_altitude_m),
        "low_altitude_prediction_horizon_sec": horizon_sec,
        "low_altitude_penalty_start_m": start_m,
        "low_altitude_min_m": min_m,
        "low_altitude_severity": float(severity),
        "low_altitude_safety_potential": float(potential),
    }


def _penalty_only_safety_delta(
    current_potential: float,
    previous_potential: float,
    episode_penalty_total: float,
    budget: float,
) -> float:
    """Return a bounded, never-positive safety shaping delta.

    Worsening risk is penalized, while holding or recovering produces zero.
    The cumulative penalty is bounded below by ``-budget`` per episode, so
    repeated danger/recovery cycles cannot farm reward or grow without bound.
    """

    raw_penalty = min(
        0.0,
        float(current_potential) - float(previous_potential),
    )
    penalty_spent = max(0.0, -float(episode_penalty_total))
    remaining_budget = max(0.0, float(budget) - penalty_spent)
    return float(max(raw_penalty, -remaining_budget))


def _compute_reward(
    state: RewardState,
    config: dict,
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    own_hp = _state_value(ownship_state, StateIndex.HEALTH, 1.0)
    target_hp = _state_value(target_state, StateIndex.HEALTH, 1.0)
    sim_time = _state_value(
        ownship_state,
        StateIndex.SIM_TIME,
        state.step_count * float(config["dt_per_step"]),
    )
    ownship_altitude_m = _state_value(
        ownship_state,
        StateIndex.ALT,
        float(config.get(
            "low_altitude_penalty_start_m",
            _SAFETY_START_ALTITUDE_M,
        )),
    )
    own_attack_scale = 1.0
    geometry_hp_gap_points = (
        (own_hp - target_hp)
        * float(config.get("hp_health_to_points_scale", 100.0))
    )
    if (
        str(config["geometry_reward_variant"])
        == "hp_conditioned_cone_range_gated_control"
    ):
        own_attack_scale, geometry_hp_gap_points = (
            _hp_conditioned_attack_scale(own_hp, target_hp, config)
        )
    distance_m, geometry_advantage, diagnostics = _geometry_snapshot(
        ownship_state,
        target_state,
        geo_info,
        sim_time,
        config,
        own_attack_scale=own_attack_scale,
    )
    diagnostics["geometry_hp_gap_points"] = float(geometry_hp_gap_points)
    new_episode = (
        state.previous_distance_m is None
        or state.previous_sim_time is None
        or sim_time <= state.previous_sim_time
    )
    if new_episode:
        # Explicit env reset is the normal path. This guard also prevents debt
        # leaking across episodes in compatibility/direct-call paths.
        state.hp_deficit_debt = 0.0

    hp_gap_points = 0.0
    hp_deficit_severity = 0.0
    hp_outcome_code = 0
    if str(config["reward_profile"]) == "phase3":
        (
            hp_gap_points,
            hp_deficit_severity,
            hp_outcome_code,
        ) = _hp_deficit_snapshot(own_hp, target_hp, config)
    filtered_descent_rate_mps = 0.0
    if not new_episode and state.previous_altitude_m is not None:
        altitude_step_dt_sec = sim_time - float(state.previous_sim_time)
        raw_descent_rate_mps = max(
            0.0,
            (float(state.previous_altitude_m) - ownship_altitude_m)
            / altitude_step_dt_sec,
        )
        descent_cap_mps = float(
            config["low_altitude_descent_rate_cap_mps"]
        )
        ema_alpha = float(
            config["low_altitude_descent_rate_ema_alpha"]
        )
        filtered_descent_rate_mps = (
            ema_alpha * min(raw_descent_rate_mps, descent_cap_mps)
            + (1.0 - ema_alpha)
            * float(state.filtered_descent_rate_mps)
        )
    safety_potential, safety_diagnostics = (
        _low_altitude_safety_potential(
            ownship_altitude_m,
            filtered_descent_rate_mps,
            config,
        )
    )

    r_damage = 0.0
    damage_is_eligible = (
        bool(config["include_terminal_damage"])
        or (own_hp > 0.0 and target_hp > 0.0)
    )
    if damage_is_eligible:
        damage_signal = (
            float(config["target_damage_weight"]) * float(target_damage)
            - float(config["ownship_damage_weight"]) * float(ownship_damage)
        )
        r_damage = float(config["damage_scale"]) * damage_signal

    r_distance = 0.0
    r_hp_advantage = 0.0
    r_hp_deficit_penalty = 0.0
    r_hp_deficit_refund = 0.0
    r_geometry = 0.0
    r_safety = 0.0
    distance_diagnostics: dict = {}
    geometry_step_dt_sec = 0.0
    geometry_variant = str(config["geometry_reward_variant"])
    geometry_potential = (
        float(config["geometry_episode_budget"]) * geometry_advantage
        if geometry_variant == "symmetric_cone_range_gated_potential"
        else 0.0
    )
    if not new_episode:
        geometry_step_dt_sec = sim_time - float(state.previous_sim_time)
        if str(config["reward_profile"]) == "phase3":
            live_nonterminal = (
                own_hp > 0.0
                and target_hp > 0.0
                and not terminated
                and not truncated
            )
            if live_nonterminal:
                hp_state_score = (
                    1.0
                    if hp_outcome_code > 0
                    else -hp_deficit_severity
                    if hp_outcome_code < 0
                    else 0.0
                )
                r_hp_advantage = (
                    float(config["hp_lead_episode_budget"])
                    / float(config["hp_advantage_reference_duration_sec"])
                    * geometry_step_dt_sec
                    * hp_state_score
                )
                r_hp_deficit_penalty = min(0.0, r_hp_advantage)
        r_distance, distance_diagnostics = _distance_reward(
            float(state.previous_distance_m),
            distance_m,
            config,
        )
        if geometry_variant == "symmetric_cone_range_gated_potential":
            previous_geometry_potential = (
                geometry_potential
                if state.previous_geometry_potential is None
                else float(state.previous_geometry_potential)
            )
            terminal_geometry_potential = (
                0.0 if terminated or truncated else geometry_potential
            )
            r_geometry = (
                float(config["geometry_potential_gamma"])
                * terminal_geometry_potential
                - previous_geometry_potential
            )
            geometry_potential = terminal_geometry_potential
        elif own_hp > 0.0 and target_hp > 0.0:
            r_geometry = (
                float(config["geometry_episode_budget"])
                / float(config["geometry_reference_duration_sec"])
                * geometry_step_dt_sec
                * geometry_advantage
            )
        previous_safety_potential = (
            safety_potential
            if state.previous_safety_potential is None
            else float(state.previous_safety_potential)
        )
        if str(config["reward_profile"]) == "phase3":
            r_safety = _penalty_only_safety_delta(
                safety_potential,
                previous_safety_potential,
                state.safety_penalty_total,
                float(config["low_altitude_potential_budget"]),
            )
            state.safety_penalty_total += r_safety

    state.previous_distance_m = distance_m
    state.previous_sim_time = sim_time
    state.previous_altitude_m = ownship_altitude_m
    state.filtered_descent_rate_mps = filtered_descent_rate_mps
    state.previous_safety_potential = safety_potential
    state.previous_geometry_potential = geometry_potential
    state.step_count += 1

    r_terminal = 0.0
    if terminated:
        # Ground loss is split without changing its final category value.
        # Safety has already emitted a bounded negative prefix, so the terminal
        # pays exactly the remainder needed for safety + terminal == -100.
        if end_condition == _OWNSHIP_ALT_END:
            r_terminal = (
                float(config["ownship_alt_reward"])
                - float(state.safety_penalty_total)
            )
        elif end_condition in _TARGET_FORCED_LOSS_ENDS:
            r_terminal = float(config["target_alt_reward"])
        elif end_condition in _OWNSHIP_FORCED_LOSS_ENDS:
            r_terminal = float(config["loss_reward"])
        else:
            own_dead = own_hp <= 0.0
            target_dead = target_hp <= 0.0
            if target_dead and not own_dead:
                r_terminal = float(config["win_reward"])
            elif own_dead and not target_dead:
                r_terminal = float(config["loss_reward"])
            else:
                # Simultaneous destruction and any otherwise-unclassified
                # terminal outcome are draws.  The scalar penalty discourages
                # draw-seeking, but outcome logs/PFSP retain the draw label.
                r_terminal = float(config["timeout_reward"])

    # Phase 2 timeout follows the competition HP tie-break. Phase 1 keeps all
    # timeout outcome values at zero through its profile configuration.
    r_timeout = 0.0
    # Defensive precedence: Gymnasium normally guarantees that terminated and
    # truncated are mutually exclusive, but never allow an upstream double
    # flag to stack two outcome rewards.
    if truncated and not terminated:
        hp_advantage = own_hp - target_hp
        if hp_advantage > 1e-9:
            r_timeout = float(config["timeout_win_reward"])
        elif hp_advantage < -1e-9:
            r_timeout = float(config["timeout_loss_reward"])
        else:
            r_timeout = float(config["timeout_reward"])

    # HP and ATA guidance are deliberately bounded. No terminal label creates
    # an extra damage, destruction, debt-refund, or geometry bonus.
    component_name = str(config["geometry_reward_component"])
    r_ata = r_geometry if component_name == "ata" else 0.0
    r_weapon = r_geometry if component_name == "weapon" else 0.0
    total = (
        r_damage
        + r_distance
        + r_hp_advantage
        + r_ata
        + r_weapon
        + r_safety
        + r_terminal
        + r_timeout
    )
    components = {
        "damage": float(r_damage),
        "distance": float(r_distance),
        "hp_advantage": float(r_hp_advantage),
        "hp_deficit_penalty": float(r_hp_deficit_penalty),
        "hp_deficit_refund": float(r_hp_deficit_refund),
        "hp_deficit_debt": float(state.hp_deficit_debt),
        "ata": float(r_ata),
        "weapon": float(r_weapon),
        "safety": float(r_safety),
        "safety_penalty_episode_total": float(
            state.safety_penalty_total
        ),
        "terminal": float(r_terminal),
        "timeout": float(r_timeout),
        "geometry_step_dt_sec": float(geometry_step_dt_sec),
        "geometry_reward_rate_per_sec": float(
            r_geometry / geometry_step_dt_sec
            if geometry_step_dt_sec > 0.0
            else 0.0
        ),
        "geometry_potential": float(geometry_potential),
        **distance_diagnostics,
        **safety_diagnostics,
        **diagnostics,
    }
    components["health_advantage"] = float(own_hp - target_hp)
    if str(config["reward_profile"]) == "phase3":
        hp_state_score = (
            1.0
            if hp_outcome_code > 0
            else -hp_deficit_severity
            if hp_outcome_code < 0
            else 0.0
        )
        hp_reward_rate = (
            float(config["hp_lead_episode_budget"])
            / float(config["hp_advantage_reference_duration_sec"])
            * hp_state_score
        )
    else:
        hp_outcome_code = 0
        hp_reward_rate = 0.0
    components["hp_outcome_state_code"] = float(hp_outcome_code)
    components["hp_gap_points"] = float(hp_gap_points)
    components["hp_deficit_severity"] = float(hp_deficit_severity)
    components["hp_deficit_debt_cap"] = 0.0
    components["hp_advantage_reward_rate_per_sec"] = float(hp_reward_rate)
    components["ata_step_dt_sec"] = float(geometry_step_dt_sec)
    components["weapon_step_dt_sec"] = float(geometry_step_dt_sec)
    components["ata_reward_rate_per_sec"] = (
        components["geometry_reward_rate_per_sec"]
        if component_name == "ata"
        else 0.0
    )
    components["weapon_reward_rate_per_sec"] = (
        components["geometry_reward_rate_per_sec"]
        if component_name == "weapon"
        else 0.0
    )
    return float(total), components


class _RewardFunction:
    """Callable reward hook with isolated state for each environment."""

    def __init__(self, overrides: dict | None = None):
        self.config = _merged_config(overrides)
        self._pending_config: dict | None = None
        self.reward_state = RewardState()

    def queue_overrides(self, overrides: dict) -> None:
        """Apply validated overrides at the next episode reset."""

        candidate = copy.deepcopy(
            self._pending_config if self._pending_config is not None else self.config
        )
        candidate.update(dict(overrides))
        _validate_config(candidate)
        self._pending_config = candidate

    def reset_episode(self, opponent_kind: str = "") -> None:
        if self._pending_config is not None:
            self.config = self._pending_config
            self._pending_config = None
        self.reward_state = RewardState(opponent_kind=str(opponent_kind or ""))

    def prime_episode(
        self,
        ownship_state,
        target_state,
        geo_info,
        _reward_config: dict | None = None,
    ) -> None:
        sim_time = _state_value(
            ownship_state,
            StateIndex.SIM_TIME,
            0.0,
        )
        distance_m, geometry_advantage, _ = _geometry_snapshot(
            ownship_state,
            target_state,
            geo_info,
            sim_time,
            self.config,
        )
        self.reward_state.previous_distance_m = distance_m
        self.reward_state.previous_sim_time = sim_time
        self.reward_state.previous_geometry_potential = (
            float(self.config["geometry_episode_budget"])
            * float(geometry_advantage)
            if str(self.config["geometry_reward_variant"])
            == "symmetric_cone_range_gated_potential"
            else 0.0
        )
        ownship_altitude_m = _state_value(
            ownship_state,
            StateIndex.ALT,
            float(self.config.get(
                "low_altitude_penalty_start_m",
                _SAFETY_START_ALTITUDE_M,
            )),
        )
        safety_potential, _ = _low_altitude_safety_potential(
            ownship_altitude_m,
            0.0,
            self.config,
        )
        self.reward_state.previous_altitude_m = ownship_altitude_m
        self.reward_state.filtered_descent_rate_mps = 0.0
        self.reward_state.previous_safety_potential = safety_potential
        self.reward_state.hp_deficit_debt = 0.0
        self.reward_state.step_count = 0

    def __call__(
        self,
        ownship_state,
        target_state,
        ownship_damage: float,
        target_damage: float,
        geo_info,
        wez_config: dict,
        reward_config: dict,
        terminated: bool,
        truncated: bool,
        end_condition: str,
    ) -> tuple[float, dict]:
        del wez_config, reward_config
        return _compute_reward(
            self.reward_state,
            self.config,
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            geo_info,
            bool(terminated),
            bool(truncated),
            str(end_condition),
        )

    def settle_abnormal_terminal(
        self,
        end_condition: str,
    ) -> tuple[float, dict]:
        """Score an FDM-NaN terminal that bypasses normal geometry reward.

        ``DogFightWrapper`` returns immediately when either simulator emits
        NaN, before calling the configured reward hook. The isolated ver06 env
        calls this method so an ownship numerical fall cannot become a zero-
        reward escape from a loss and a target fall remains a proper win.
        Geometry and safety cannot be evaluated safely from NaN state, so only
        the canonical terminal is emitted. FDM labels never refund HP debt.
        """

        reason = str(end_condition)
        if reason in _TARGET_FORCED_LOSS_ENDS:
            terminal = float(self.config["target_alt_reward"])
        elif reason in _OWNSHIP_FORCED_LOSS_ENDS:
            terminal = float(self.config["loss_reward"])
        else:
            raise ValueError(
                f"unsupported abnormal terminal condition: {reason!r}"
            )

        # FDM outcome labels never refund HP debt. No trustworthy combat HP
        # transition can be inferred from a numerical simulator failure.
        total = terminal
        return float(total), {
            "damage": 0.0,
            "distance": 0.0,
            "hp_advantage": 0.0,
            "hp_deficit_penalty": 0.0,
            "hp_deficit_refund": 0.0,
            "hp_deficit_debt": float(
                self.reward_state.hp_deficit_debt
            ),
            "ata": 0.0,
            "weapon": 0.0,
            "safety": 0.0,
            "terminal": float(terminal),
            "timeout": 0.0,
            "abnormal_fdm_terminal": 1.0,
        }


def make_reward_fn(overrides: dict | None = None) -> _RewardFunction:
    """Create an isolated reward hook for one train or evaluation env."""

    return _RewardFunction(overrides)


_FALLBACK_REWARD = _RewardFunction()


def reset_distance_tracker() -> None:
    """Compatibility reset for direct calls to ``compute_reward``."""

    _FALLBACK_REWARD.reset_episode()


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Statically discoverable hook; normal env creation uses the factory."""

    del wez_config
    config = (
        _FALLBACK_REWARD.config
        if not reward_config
        else _merged_config(reward_config)
    )
    return _compute_reward(
        _FALLBACK_REWARD.reward_state,
        config,
        ownship_state,
        target_state,
        ownship_damage,
        target_damage,
        geo_info,
        bool(terminated),
        bool(truncated),
        str(end_condition),
    )


__all__ = [
    "MY_REWARD_CONFIG",
    "PHASE1_REWARD_CONFIG",
    "PHASE2_REWARD_CONFIG",
    "PHASE3_REWARD_CONFIG",
    "REWARD_PROFILES",
    "RewardState",
    "compute_reward",
    "get_reward_config",
    "make_reward_fn",
    "reset_distance_tracker",
]
