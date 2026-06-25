"""Training metrics reader for the unified dashboard."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path


class MetricsReader:
    """Incrementally read dashboard metrics from run directories."""

    def __init__(self, logdir: Path) -> None:
        self.logdir = Path(logdir).resolve()
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}
        self._cache: dict[str, list[dict]] = {}

    def list_runs(self) -> list[dict]:
        """Return metric run directories sorted by latest modification time."""
        runs = []
        if not self.logdir.exists():
            return runs
        for run_dir in sorted(self.logdir.iterdir()):
            metrics_path = run_dir / "metrics.jsonl"
            if not run_dir.is_dir() or not metrics_path.exists():
                continue
            rows = self._read_rows(run_dir.name)
            runs.append(
                {
                    "name": run_dir.name,
                    "last_step": int(rows[-1].get("step", 0)) if rows else 0,
                    "last_modified": metrics_path.stat().st_mtime,
                    "has_config": (
                        (run_dir / "config.json").exists()
                        or (run_dir / "config.yaml").exists()
                    ),
                }
            )
        runs.sort(key=lambda item: item["last_modified"], reverse=True)
        return runs

    def read_metrics(
        self,
        run: str,
        since_step: int = 0,
        smooth: int = 1,
    ) -> dict:
        """Return scalar metric series for one training run."""
        rows = self._read_rows(run)
        full_series: dict[str, list[list[float]]] = {}
        for row in rows:
            step = _as_number(row.get("step"))
            if step is None:
                continue
            for key, value in row.items():
                if key == "step":
                    continue
                number = _as_number(value)
                if number is not None:
                    full_series.setdefault(key, []).append([step, number])
        if smooth > 1:
            full_series = {
                key: self._smooth_ema(points, smooth)
                for key, points in full_series.items()
            }
        series = {
            key: [point for point in points if point[0] > since_step]
            for key, points in full_series.items()
        }
        last_step = int(rows[-1].get("step", 0)) if rows else 0
        return {"last_step": last_step, "metrics": series}

    def get_latest(self, run: str) -> dict:
        """Return latest scalar values and alert subset for one run."""
        values = {}
        step = 0
        for row in self._read_rows(run):
            step = int(row.get("step", step) or step)
            for key, value in row.items():
                if key == "step":
                    continue
                number = _as_number(value)
                if number is not None:
                    values[key] = number
        alerts = {
            key: value
            for key, value in values.items()
            if self._is_alert(key, value)
        }
        return {"step": step, "values": values, "alerts": alerts}

    def get_config(self, run: str) -> dict:
        """Read sibling config JSON/YAML for one training run."""
        run_dir = self._run_dir(run)
        json_path = run_dir / "config.json"
        if json_path.exists():
            try:
                return json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"raw": json_path.read_text(encoding="utf-8")}
        yaml_path = run_dir / "config.yaml"
        if yaml_path.exists():
            return {"raw": yaml_path.read_text(encoding="utf-8")}
        return {}

    def _read_rows(self, run: str) -> list[dict]:
        metrics_path = self._run_dir(run) / "metrics.jsonl"
        if not metrics_path.exists():
            return self._cache.get(run, [])

        with self._lock:
            size = metrics_path.stat().st_size
            offset = self._offsets.get(run, 0)
            if offset > size:
                offset = 0
                self._cache[run] = []
            if offset == size:
                return self._cache.get(run, [])

            with metrics_path.open("rb") as file:
                file.seek(offset)
                chunk = file.read()
                self._offsets[run] = file.tell()

            rows = self._cache.setdefault(run, [])
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows

    def _run_dir(self, run: str) -> Path:
        candidate = (self.logdir / run).resolve()
        if self.logdir not in candidate.parents and candidate != self.logdir:
            raise ValueError("invalid run path")
        return candidate

    @staticmethod
    def _smooth_ema(points: list[list[float]], smooth: int) -> list[list[float]]:
        if not points:
            return points
        weight = 1.0 / (smooth + 1)
        value = points[0][1]
        result = []
        for step, raw in points:
            value = (1.0 - weight) * value + weight * raw
            result.append([step, round(value, 6)])
        return result

    @staticmethod
    def _is_alert(key: str, value: float) -> bool:
        thresholds = {
            "episode/crash_rate": (">", 0.3),
            "action/saturation_rate": (">", 0.6),
            "dogfight/headon_guard_fail": (">", 0.0),
            "dogfight/altitude_penalty_steps": (">", 0.0),
            "train/kl": (">", 1.0),
            "train/entropy": ("<", 0.01),
        }
        rule = thresholds.get(key)
        if not rule:
            return False
        op, threshold = rule
        return value > threshold if op == ">" else value < threshold


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None
