from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CostWeights:
    damage_dealt: float = 140.0
    damage_taken: float = 180.0
    attack_geometry: float = 3.5
    control_zone: float = 9.0
    closure: float = 1.0
    threat_geometry: float = 12.0
    nose_advantage: float = 9.0
    far_range: float = 4.0
    overshoot: float = 6.0
    ground: float = 120.0
    envelope: float = 6.0
    terminal_geometry: float = 28.0
    control_slew: float = 0.03


@dataclass(frozen=True)
class CEMConfig:
    candidates: int = 48
    iterations: int = 2
    elite_fraction: float = 0.1666666667
    initial_std: tuple[float, float, float, float] = (0.65, 0.65, 0.40, 0.25)
    minimum_std: tuple[float, float, float, float] = (0.10, 0.10, 0.08, 0.04)
    maximum_std: tuple[float, float, float, float] = (0.80, 0.80, 0.55, 0.35)
    mean_smoothing: float = 0.25
    std_smoothing: float = 0.35
    seed: int = 20260802
    adaptive_candidate_counts: tuple[int, ...] = (48,)


@dataclass(frozen=True)
class PredictionConfig:
    acceleration_time_constant_s: float = 1.25
    turn_rate_time_constant_s: float = 1.5
    acceleration_smoothing: float = 0.35
    turn_rate_smoothing: float = 0.35
    max_acceleration_mps2: float = 80.0
    max_turn_rate_radps: float = 1.5
    min_speed_mps: float = 80.0
    max_speed_mps: float = 650.0


@dataclass(frozen=True)
class MPCConfig:
    simulation_hz: int = 60
    policy_hz: int = 10
    horizon_seconds: float = 2.0
    knot_seconds: float = 0.5
    compute_budget_ms: float = 80.0
    max_candidates: int = 48
    native_dll: str = "runtime/predictor/Release/MPCJSBSim.dll"
    asset_root: str = "."
    cem: CEMConfig = field(default_factory=CEMConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    weights: CostWeights = field(default_factory=CostWeights)

    @property
    def action_repeat(self) -> int:
        if self.policy_hz <= 0 or self.simulation_hz % self.policy_hz:
            raise ValueError("simulation_hz must be an integer multiple of policy_hz")
        return self.simulation_hz // self.policy_hz

    @property
    def steps_per_knot(self) -> int:
        value = int(round(self.knot_seconds * self.simulation_hz))
        if value < 1:
            raise ValueError("knot_seconds is shorter than one simulation step")
        return value

    @property
    def knot_count(self) -> int:
        value = int(round(self.horizon_seconds / self.knot_seconds))
        if value < 1:
            raise ValueError("horizon_seconds must contain at least one knot")
        return value


def _tuple4(value: Any, name: str) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in value)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values")
    return values  # type: ignore[return-value]


def _build_config(raw: dict[str, Any]) -> MPCConfig:
    cem_raw = dict(raw.get("cem", {}))
    pred_raw = dict(raw.get("prediction", {}))
    weights_raw = dict(raw.get("weights", {}))
    for name in ("initial_std", "minimum_std", "maximum_std"):
        if name in cem_raw:
            cem_raw[name] = _tuple4(cem_raw[name], f"cem.{name}")
    if "adaptive_candidate_counts" in cem_raw:
        cem_raw["adaptive_candidate_counts"] = tuple(
            int(x) for x in cem_raw["adaptive_candidate_counts"]
        )
    return MPCConfig(
        **{k: v for k, v in raw.items() if k not in {"cem", "prediction", "weights"}},
        cem=CEMConfig(**cem_raw),
        prediction=PredictionConfig(**pred_raw),
        weights=CostWeights(**weights_raw),
    )


def load_config(path: str | Path | None = None) -> MPCConfig:
    if path is None:
        return MPCConfig()
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"MPC config must be a mapping: {config_path}")
    config = _build_config(raw)
    if config.max_candidates < config.cem.candidates:
        raise ValueError("max_candidates must be >= cem.candidates")
    if config.horizon_seconds <= 0.0 or config.compute_budget_ms <= 0.0:
        raise ValueError("horizon_seconds and compute_budget_ms must be positive")
    _ = config.action_repeat, config.steps_per_knot, config.knot_count
    return config
