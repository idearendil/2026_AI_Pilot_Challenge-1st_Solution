from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SafeMPCConfig:
    """Safety and optional robust-selection settings layered over native MPC."""

    safety_clearance_m: float = 380.0
    safety_pullout_accel_mps2: float = 28.0
    safety_roll_rate_radps: float = 1.6
    safety_level_roll_deg: float = 45.0
    safety_check_mode: str = "lazy"
    safety_policy: str = "predictive"
    emergency_trigger_margin_m: float = 50.0
    safety_risk_penalty_per_m: float = 0.01
    robust_enabled: bool = False
    robust_top_k: int = 8
    robust_native_loss_std: float = 0.20
    robust_worst_case_weight: float = 0.50
    response_turn_rate_radps: float = 0.30
    response_entry_time_constant_s: float = 0.35
    response_decay_time_constant_s: float = 2.50
    candidate_selection: str = "all_samples"
    selection_blend_fraction: float = 0.50


def load_safe_mpc_config(path: str | Path | None = None) -> SafeMPCConfig:
    if path is None:
        config = SafeMPCConfig()
    else:
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"safe MPC config must be a mapping: {config_path}")
        config = SafeMPCConfig(**raw)
    if config.safety_clearance_m < 0.0:
        raise ValueError("safety_clearance_m must be non-negative")
    if config.safety_pullout_accel_mps2 <= 0.0:
        raise ValueError("safety_pullout_accel_mps2 must be positive")
    if config.safety_roll_rate_radps <= 0.0:
        raise ValueError("safety_roll_rate_radps must be positive")
    if not 0.0 < config.safety_level_roll_deg < 180.0:
        raise ValueError("safety_level_roll_deg must be between zero and 180")
    if config.safety_check_mode not in {"full", "lazy"}:
        raise ValueError("safety_check_mode must be full or lazy")
    if config.safety_policy not in {
        "predictive",
        "emergency",
        "altitude",
        "risk_priced",
    }:
        raise ValueError(
            "safety_policy must be predictive, emergency, altitude, or risk_priced"
        )
    if config.emergency_trigger_margin_m < 0.0:
        raise ValueError("emergency_trigger_margin_m must be non-negative")
    if config.safety_risk_penalty_per_m < 0.0:
        raise ValueError("safety_risk_penalty_per_m must be non-negative")
    if config.robust_top_k < 2:
        raise ValueError("robust_top_k must be at least two")
    if config.robust_native_loss_std < 0.0:
        raise ValueError("robust_native_loss_std must be non-negative")
    if not 0.0 <= config.robust_worst_case_weight <= 1.0:
        raise ValueError("robust_worst_case_weight must be between zero and one")
    if config.response_turn_rate_radps <= 0.0:
        raise ValueError("response_turn_rate_radps must be positive")
    if (
        config.response_entry_time_constant_s <= 0.0
        or config.response_decay_time_constant_s <= 0.0
    ):
        raise ValueError("response time constants must be positive")
    if config.candidate_selection not in {
        "all_samples",
        "last_iteration",
        "smoothed_mean",
        "elite_mean",
        "winner_mean_blend",
    }:
        raise ValueError(
            "candidate_selection must be all_samples, last_iteration, "
            "smoothed_mean, elite_mean, or winner_mean_blend"
        )
    if not 0.0 <= config.selection_blend_fraction <= 1.0:
        raise ValueError("selection_blend_fraction must be between zero and one")
    return config
