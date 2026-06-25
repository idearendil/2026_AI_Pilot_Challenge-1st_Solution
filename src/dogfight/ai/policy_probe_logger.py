from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class PolicyProbeLogger:
    """Log deterministic policy probes during training.

    The probes are intentionally synthetic and fixed. They are not an evaluation
    rollout; they are a cheap way to inspect how the current actor maps stable
    observation patterns to actions, and how recurrent state evolves over a few
    inference steps.
    """

    CSV_FIELDS = [
        "iteration",
        "sampled_steps",
        "stage",
        "probe",
        "probe_step",
        "obs_first4",
        "action_roll",
        "action_pitch",
        "action_rudder",
        "action_throttle",
        "action_norm",
        "action_delta_norm",
        "state_in_norm",
        "state_out_norm",
    ]

    def __init__(
        self,
        log_dir: Path | str,
        *,
        obs_dim: int,
        action_dim: int = 4,
        interval: int = 0,
        sequence_steps: int = 4,
        print_to_console: bool = True,
        append: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.interval = max(0, int(interval))
        self.sequence_steps = max(1, int(sequence_steps))
        self.print_to_console = print_to_console
        self.append = append
        self.csv_path = self.log_dir / "policy_probe.csv"
        self.jsonl_path = self.log_dir / "policy_probe.jsonl"
        self._csv_file = None
        self._csv_writer = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def __enter__(self) -> "PolicyProbeLogger":
        if not self.enabled:
            return self
        self.log_dir.mkdir(parents=True, exist_ok=True)
        csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        mode = "a" if self.append else "w"
        self._csv_file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        if not self.append or not csv_exists:
            self._csv_writer.writeheader()
        if not self.append:
            self.jsonl_path.write_text("", encoding="utf-8")
        else:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.touch(exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def maybe_log(
        self,
        algorithm: Any,
        *,
        iteration: int,
        sampled_steps: Any = None,
        stage: int | None = None,
    ) -> None:
        """Log probes for this iteration when the configured interval matches."""

        if not self.enabled:
            return
        if iteration % self.interval != 0:
            return

        records = self._run_probes(algorithm, iteration, sampled_steps, stage)
        if not records:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        assert self._csv_writer is not None
        for record in records:
            self._csv_writer.writerow({
                key: _csv_value(record.get(key)) for key in self.CSV_FIELDS
            })
        if self._csv_file is not None:
            self._csv_file.flush()
        if self.print_to_console:
            self._print_summary(iteration, records)

    def _run_probes(
        self,
        algorithm: Any,
        iteration: int,
        sampled_steps: Any,
        stage: int | None,
    ) -> list[dict[str, Any]]:
        try:
            actor = _PolicyActorView(algorithm)
        except Exception as exc:
            print(f"[DogFightEnv][PolicyProbe] disabled: {exc}")
            self.interval = 0
            return []

        records: list[dict[str, Any]] = []
        for probe_name, sequence in self._build_probe_sequences().items():
            actor.reset_state()
            previous_action = None
            for step, obs in enumerate(sequence):
                probe = actor.compute(obs)
                action = _fit_action(probe["action"], self.action_dim)
                action_delta = (
                    float(np.linalg.norm(action - previous_action))
                    if previous_action is not None
                    else 0.0
                )
                previous_action = action
                records.append({
                    "iteration": iteration,
                    "sampled_steps": sampled_steps,
                    "stage": stage,
                    "probe": probe_name,
                    "probe_step": step,
                    "obs_first4": [float(x) for x in obs[:4]],
                    "action_roll": float(action[0]) if action.size > 0 else 0.0,
                    "action_pitch": float(action[1]) if action.size > 1 else 0.0,
                    "action_rudder": float(action[2]) if action.size > 2 else 0.0,
                    "action_throttle": float(action[3]) if action.size > 3 else 0.0,
                    "action_norm": float(np.linalg.norm(action)),
                    "action_delta_norm": action_delta,
                    "state_in_norm": probe["state_in_norm"],
                    "state_out_norm": probe["state_out_norm"],
                })
        return records

    def _build_probe_sequences(self) -> dict[str, list[np.ndarray]]:
        base = np.zeros(self.obs_dim, dtype=np.float32)
        probes: dict[str, list[np.ndarray]] = {}

        probes["zero_hold"] = [base.copy() for _ in range(self.sequence_steps)]

        ahead = base.copy()
        if self.obs_dim >= 9:
            ahead[6] = 1.0
            ahead[7] = 0.0
            ahead[8] = 0.0
        elif self.obs_dim:
            ahead[0] = 1.0
        probes["target_ahead"] = [
            _with_time_drift(ahead, step) for step in range(self.sequence_steps)
        ]

        left = base.copy()
        if self.obs_dim >= 9:
            left[6] = 0.0
            left[7] = 1.0
            left[8] = 0.0
        elif self.obs_dim:
            left[0] = -1.0
        probes["target_left"] = [
            _with_time_drift(left, step) for step in range(self.sequence_steps)
        ]

        ramp = np.linspace(-0.5, 0.5, self.obs_dim, dtype=np.float32)
        probes["ramp"] = [
            np.clip(ramp + 0.02 * step, -1.0, 1.0).astype(np.float32)
            for step in range(self.sequence_steps)
        ]
        return probes

    def _print_summary(self, iteration: int, records: list[dict[str, Any]]) -> None:
        last_by_probe = {}
        for record in records:
            last_by_probe[record["probe"]] = record
        parts = []
        for probe_name in sorted(last_by_probe):
            record = last_by_probe[probe_name]
            action = [
                record["action_roll"],
                record["action_pitch"],
                record["action_rudder"],
                record["action_throttle"],
            ]
            parts.append(
                f"{probe_name}:a={_short_list(action)} "
                f"s={_fmt_float(record['state_out_norm'])}"
            )
        print(
            "[DogFightEnv][PolicyProbe] "
            f"iter={iteration} " + " | ".join(parts)
        )


class _PolicyActorView:
    def __init__(self, algorithm: Any) -> None:
        self.algorithm = algorithm
        self.module = None
        self.state = None
        self._torch = None
        self._columns = None
        if hasattr(algorithm, "get_module"):
            self.module = algorithm.get_module("default_policy")
            if self.module is None:
                self.module = algorithm.get_module()
        if self.module is not None:
            import torch
            from ray.rllib.core.columns import Columns

            self._torch = torch
            self._columns = Columns
            return
        if not hasattr(algorithm, "compute_single_action"):
            raise RuntimeError("algorithm exposes neither RLModule nor compute_single_action")

    def reset_state(self) -> None:
        self.state = None
        if self.module is None or not hasattr(self.module, "get_initial_state"):
            return
        initial_state = self.module.get_initial_state()
        if initial_state:
            self.state = _batch_state(initial_state, self._torch)

    def compute(self, obs: np.ndarray) -> dict[str, Any]:
        if self.module is None:
            action = self.algorithm.compute_single_action(obs, explore=False)
            return {
                "action": np.asarray(action, dtype=np.float32),
                "state_in_norm": None,
                "state_out_norm": None,
            }
        return self._compute_module(obs)

    def _compute_module(self, obs: np.ndarray) -> dict[str, Any]:
        assert self._torch is not None
        assert self._columns is not None
        state_in = self.state
        if state_in is not None:
            obs_batch = obs[None, None, :]
        else:
            obs_batch = obs[None, :]
        batch = {
            self._columns.OBS: self._torch.as_tensor(obs_batch, dtype=self._torch.float32)
        }
        if state_in is not None:
            batch[self._columns.STATE_IN] = state_in
            batch[self._columns.SEQ_LENS] = self._torch.as_tensor([1], dtype=self._torch.int32)
        with self._torch.no_grad():
            output = self.module.forward_inference(batch)
            if self._columns.STATE_OUT in output:
                self.state = _detach_state(output[self._columns.STATE_OUT])
            if self._columns.ACTIONS in output:
                action = output[self._columns.ACTIONS]
            else:
                logits = output[self._columns.ACTION_DIST_INPUTS]
                dist_class = self.module.get_inference_action_dist_cls()
                action_dist = dist_class.from_logits(logits).to_deterministic()
                action = action_dist.sample()
        return {
            "action": _to_numpy_action(action),
            "state_in_norm": _state_norm(state_in),
            "state_out_norm": _state_norm(self.state),
        }


def _with_time_drift(obs: np.ndarray, step: int) -> np.ndarray:
    value = obs.copy()
    if value.size:
        value[0] = np.clip(value[0] + 0.01 * step, -1.0, 1.0)
    return value


def _batch_state(state: Any, torch_module: Any) -> Any:
    if isinstance(state, dict):
        return {key: _batch_state(value, torch_module) for key, value in state.items()}
    tensor = state if hasattr(state, "detach") else torch_module.as_tensor(state)
    return tensor.detach().clone().float().unsqueeze(0)


def _detach_state(state: Any) -> Any:
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    return state.detach().clone() if hasattr(state, "detach") else state


def _state_norm(state: Any) -> float | None:
    if state is None:
        return None
    if isinstance(state, dict):
        total = 0.0
        found = False
        for value in state.values():
            norm = _state_norm(value)
            if norm is not None:
                total += norm * norm
                found = True
        return math.sqrt(total) if found else None
    if hasattr(state, "detach"):
        return float(state.detach().float().norm().cpu().item())
    try:
        return float(np.linalg.norm(np.asarray(state, dtype=np.float32)))
    except Exception:
        return None


def _to_numpy_action(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action_array = np.asarray(action, dtype=np.float32)
    while action_array.ndim > 1 and action_array.shape[0] == 1:
        action_array = action_array[0]
    return action_array


def _fit_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.size >= action_dim:
        return flat[:action_dim]
    padded = np.zeros(action_dim, dtype=np.float32)
    padded[: flat.size] = flat
    return padded


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _short_list(values: list[float]) -> str:
    return "[" + ",".join(_fmt_float(value) for value in values) + "]"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)
