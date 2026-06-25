from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


class DashboardJsonlLogger:
    """Write RLlib training rows in the dashboard metrics.jsonl format."""

    def __init__(
        self,
        root_dir: Path | str,
        run_name: str,
        config: dict[str, Any] | None = None,
        append: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.run_name = _safe_run_name(run_name)
        self.run_dir = self.root_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        if not append:
            self.metrics_path.write_text("", encoding="utf-8")
        if config is not None:
            self.write_config(config)

    def write_config(self, config: dict[str, Any]) -> None:
        path = self.run_dir / "config.json"
        path.write_text(
            json.dumps(_json_safe(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_row(
        self,
        row: dict[str, Any],
        *,
        iteration_key: str = "iter",
        step_key: str = "sampled_steps",
        extra: dict[str, Any] | None = None,
    ) -> None:
        metrics = training_row_to_dashboard_metrics(
            row,
            iteration_key=iteration_key,
            step_key=step_key,
            extra=extra,
        )
        if len(metrics) <= 1:
            return
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def training_row_to_dashboard_metrics(
    row: dict[str, Any],
    *,
    iteration_key: str = "iter",
    step_key: str = "sampled_steps",
    extra: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Map DogFight RLlib summary rows to stable dashboard metric names."""

    step = _number(row.get(step_key))
    if step is None:
        step = _number(row.get("total_iter"))
    if step is None:
        step = _number(row.get(iteration_key))
    if step is None:
        step = 0

    mapped = {
        "step": int(step),
        "episode/score": row.get("reward_mean"),
        "episode/reward_mean": row.get("reward_mean"),
        "episode/length": row.get("ep_len_mean"),
        "episode/win_rate": row.get("win_rate"),
        "episode/loss_rate": row.get("loss_rate"),
        "episode/timeout_rate": row.get("timeout_rate"),
        "episode/crash_rate": row.get("crash_rate"),
        "episode/count": row.get("episodes"),
        "dogfight/wez_steps": row.get("ep_wez_steps"),
        "dogfight/distance_mean": row.get("ep_mean_distance"),
        "dogfight/distance_min": row.get("ep_min_distance"),
        "dogfight/initial_alpha_deg": row.get("initial_alpha_deg"),
        "dogfight/initial_ata_deg": row.get("initial_ata_deg"),
        "dogfight/initial_aa_deg": row.get("initial_aa_deg"),
        "dogfight/initial_distance_m": row.get("initial_distance_m"),
        "dogfight/final_ata_deg": row.get("final_ata_deg"),
        "dogfight/final_aa_deg": row.get("final_aa_deg"),
        "dogfight/headon_guard_fail": row.get("headon_guard_fail"),
        "dogfight/altitude_penalty_steps": row.get("ep_altitude_penalty_steps"),
        "reward/pursuit": row.get("ep_reward_pursuit"),
        "reward/damage": row.get("ep_reward_damage"),
        "reward/safety": row.get("ep_reward_safety"),
        "reward/survival": row.get("ep_reward_survival"),
        "action/saturation_rate": row.get("action_sat_rate"),
        "action/roll_mean": row.get("action_roll_mean"),
        "action/pitch_mean": row.get("action_pitch_mean"),
        "action/rudder_mean": row.get("action_rudder_mean"),
        "action/throttle_mean": row.get("action_throttle_mean"),
        "action/roll_std": row.get("action_roll_std"),
        "action/pitch_std": row.get("action_pitch_std"),
        "action/rudder_std": row.get("action_rudder_std"),
        "action/throttle_std": row.get("action_throttle_std"),
        "train/loss/policy": row.get("policy_loss"),
        "train/loss/value": row.get("vf_loss"),
        "train/loss/actor": row.get("actor_loss"),
        "train/loss/critic": row.get("critic_loss"),
        "train/loss/alpha": row.get("alpha_loss"),
        "train/alpha": row.get("alpha"),
        "train/target_entropy": row.get("target_entropy"),
        "train/entropy": row.get("entropy"),
        "train/kl": row.get("kl"),
        "train/clip_frac": row.get("clip_frac"),
        "train/explained_var": row.get("explained_var"),
        "replay/memory_mb": row.get("replay_buffer_memory_mb"),
        "perf/env_steps_per_sec": row.get("env_steps_per_sec"),
        "perf/learner_steps_per_sec": row.get("learner_steps_per_sec"),
        "perf/iteration_time_s": row.get("iteration_time_s"),
    }
    if extra:
        mapped.update(extra)
    return {
        key: value
        for key, value in (
            (name, _number(value)) for name, value in mapped.items()
        )
        if value is not None
    }


def load_experiment_metadata(
    yaml_path: str | Path | None,
    *,
    script_name: str,
    cli_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Load experiment YAML metadata for dashboard configs and records."""
    if not yaml_path:
        return {
            "script": script_name,
            "cli_argv": cli_argv or [],
            "experiment_yaml": "",
            "experiment": {},
        }

    path = Path(yaml_path)
    metadata: dict[str, Any] = {
        "script": script_name,
        "cli_argv": cli_argv or [],
        "experiment_yaml": str(path),
        "experiment": {},
    }
    if not path.exists():
        return metadata
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metadata["experiment"] = data if isinstance(data, dict) else {"raw": data}
    return metadata


def copy_experiment_yaml(yaml_path: str | Path | None, output_dir: str | Path) -> None:
    """Copy the source experiment YAML into a record directory when available."""
    if not yaml_path:
        return
    src = Path(yaml_path)
    if not src.exists():
        return
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "experiment.yaml")


def _safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe.strip("._") or "run"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
