from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dogfight.ai.policy_probe_logger import _PolicyActorView


class EngagementReplayLogger:
    """Periodically roll out the current actor and save Tacview CSV replays.

    This logger is intentionally separate from RLlib's training EnvRunners. It
    creates a short local evaluation episode, drives the ownship with the
    current actor, and lets the existing environment write the Blue/Red Tacview
    CSV pair. That keeps replay inspection opt-in and avoids touching replay
    buffer sampling or learner batches.
    """

    INDEX_FIELDS = [
        "iteration",
        "sampled_steps",
        "stage",
        "episode",
        "steps",
        "total_reward",
        "terminated",
        "truncated",
        "outcome",
        "end_condition",
        "ownship_health",
        "target_health",
        "ep_min_distance",
        "replay_dir",
        "ownship_log",
        "target_log",
        "summary_json",
    ]

    def __init__(
        self,
        log_dir: Path | str,
        *,
        env_factory: Callable[[dict], Any],
        env_config: dict,
        interval: int = 0,
        max_steps: int = 600,
        episodes: int = 1,
        print_to_console: bool = True,
        append: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.replay_root = self.log_dir / "engagement_replays"
        self.env_factory = env_factory
        self.env_config = deepcopy(env_config)
        self.interval = max(0, int(interval))
        self.max_steps = max(1, int(max_steps))
        self.episodes = max(1, int(episodes))
        self.print_to_console = print_to_console
        self.append = append
        self.csv_path = self.replay_root / "replay_index.csv"
        self.jsonl_path = self.replay_root / "replay_index.jsonl"
        self._csv_file = None
        self._csv_writer = None

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def __enter__(self) -> "EngagementReplayLogger":
        if not self.enabled:
            return self
        self.replay_root.mkdir(parents=True, exist_ok=True)
        csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        mode = "a" if self.append else "w"
        self._csv_file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=self.INDEX_FIELDS,
        )
        if not self.append or not csv_exists:
            self._csv_writer.writeheader()
        if not self.append:
            self.jsonl_path.write_text("", encoding="utf-8")
        else:
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
        """Save engagement replays when the configured interval matches."""

        if not self.enabled:
            return
        if iteration % self.interval != 0:
            return

        try:
            records = self._run_replays(
                algorithm,
                iteration=iteration,
                sampled_steps=sampled_steps,
                stage=stage,
            )
        except Exception as exc:
            print(f"[DogFightEnv][EngagementReplay] skipped iter={iteration}: {exc}")
            return

        if not records:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        assert self._csv_writer is not None
        for record in records:
            self._csv_writer.writerow({
                key: _csv_value(record.get(key)) for key in self.INDEX_FIELDS
            })
        if self._csv_file is not None:
            self._csv_file.flush()
        if self.print_to_console:
            self._print_summary(iteration, records)

    def _run_replays(
        self,
        algorithm: Any,
        *,
        iteration: int,
        sampled_steps: Any,
        stage: int | None,
    ) -> list[dict[str, Any]]:
        actor = _PolicyActorView(algorithm)
        records: list[dict[str, Any]] = []
        for episode in range(self.episodes):
            replay_dir = self._episode_dir(iteration, stage, episode)
            env = self._make_env(replay_dir)
            try:
                actor.reset_state()
                obs, _ = _normalize_reset(env.reset())
                total_reward = 0.0
                terminated = False
                truncated = False
                info: dict[str, Any] = {}

                for step in range(self.max_steps):
                    action = actor.compute(np.asarray(obs, dtype=np.float32))["action"]
                    obs, reward, terminated, truncated, info = _normalize_step(
                        env.step(action)
                    )
                    total_reward += float(reward)
                    if terminated or truncated:
                        break

                if not (terminated or truncated):
                    info = dict(info)
                    info.setdefault("end_condition", "engagement replay step limit")
                    info.setdefault("outcome", "preview")
                env.make_tacviewLog()
                records.append(
                    self._build_record(
                        replay_dir=replay_dir,
                        iteration=iteration,
                        sampled_steps=sampled_steps,
                        stage=stage,
                        episode=episode,
                        steps=step + 1,
                        total_reward=total_reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
            finally:
                if hasattr(env, "close"):
                    env.close()
        return records

    def _make_env(self, replay_dir: Path) -> Any:
        cfg = deepcopy(self.env_config)
        cfg["artifacts_dir"] = str(replay_dir)
        cfg["_runner_index"] = "engagement_replay"
        cfg["_env_index"] = 0
        replay_dir.mkdir(parents=True, exist_ok=True)
        return self.env_factory(cfg)

    def _episode_dir(
        self,
        iteration: int,
        stage: int | None,
        episode: int,
    ) -> Path:
        prefix = f"stage_{stage:02d}_" if stage is not None else ""
        return self.replay_root / f"{prefix}iter_{iteration:06d}" / f"episode_{episode:02d}"

    def _build_record(
        self,
        *,
        replay_dir: Path,
        iteration: int,
        sampled_steps: Any,
        stage: int | None,
        episode: int,
        steps: int,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        ownship_log = _first_match(replay_dir, "*_ownship_(F-16)[Blue].csv")
        target_log = _first_match(replay_dir, "*_target_(F-16)[Red].csv")
        summary_json = _first_match(replay_dir, "*_summary.json")
        return {
            "iteration": iteration,
            "sampled_steps": sampled_steps,
            "stage": stage,
            "episode": episode,
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "outcome": info.get("outcome", ""),
            "end_condition": info.get("end_condition", ""),
            "ownship_health": info.get("ownship_health", ""),
            "target_health": info.get("target_health", ""),
            "ep_min_distance": info.get("ep_min_distance", ""),
            "replay_dir": str(replay_dir),
            "ownship_log": str(ownship_log) if ownship_log else "",
            "target_log": str(target_log) if target_log else "",
            "summary_json": str(summary_json) if summary_json else "",
        }

    def _print_summary(self, iteration: int, records: list[dict[str, Any]]) -> None:
        parts = []
        for record in records:
            parts.append(
                "ep={episode} steps={steps} outcome={outcome} "
                "min_dist={min_dist}".format(
                    episode=record["episode"],
                    steps=record["steps"],
                    outcome=record["outcome"] or "n/a",
                    min_dist=_fmt_float(record["ep_min_distance"]),
                )
            )
        print(
            "[DogFightEnv][EngagementReplay] "
            f"iter={iteration} " + " | ".join(parts)
        )


def _normalize_reset(result: Any) -> tuple[Any, dict]:
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _normalize_step(result: Any) -> tuple[Any, float, bool, bool, dict]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated), bool(truncated), dict(info)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, dict(info)
    raise ValueError(f"Unexpected env.step result shape: {type(result).__name__}")


def _first_match(root: Path, pattern: str) -> Path | None:
    suffix = pattern[1:] if pattern.startswith("*") else pattern
    for path in sorted(root.iterdir()):
        if path.name.endswith(suffix):
            return path
    return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)
