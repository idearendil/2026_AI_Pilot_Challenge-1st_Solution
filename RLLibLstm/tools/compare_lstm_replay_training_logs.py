"""Compare EpisodeReplayBuffer vs PrioritizedEpisodeReplayBuffer training logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


METRICS = (
    "reward_mean",
    "win_rate",
    "crash_rate",
    "ep_wez_steps",
    "ep_min_distance",
    "actor_loss",
    "critic_loss",
    "alpha",
    "replay_buffer_memory_mb",
    "iteration_time_s",
)


def _to_float(value: str):
    if value in {"", "n/a", "nan", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _summarize(path: Path, tail: int) -> dict[str, object]:
    rows = _load_rows(path)
    selected = rows[-tail:] if tail > 0 else rows
    summary: dict[str, object] = {
        "path": str(path),
        "rows": len(rows),
        "tail": len(selected),
    }
    for metric in METRICS:
        values = [_to_float(row.get(metric, "")) for row in selected]
        values = [value for value in values if value is not None]
        summary[metric] = mean(values) if values else "n/a"
    return summary


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_log", type=Path)
    parser.add_argument("prioritized_log", type=Path)
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Average over this many final iterations. Use 0 for all rows.",
    )
    args = parser.parse_args()

    episode = _summarize(args.episode_log, args.tail)
    prioritized = _summarize(args.prioritized_log, args.tail)

    print("[DogFightEnv][lstm_replay_compare] episode_log=", episode["path"])
    print("[DogFightEnv][lstm_replay_compare] prioritized_log=", prioritized["path"])
    print(
        "[DogFightEnv][lstm_replay_compare] rows="
        f" episode:{episode['rows']} prioritized:{prioritized['rows']} tail:{args.tail}"
    )
    print("metric,episode,prioritized,delta_prioritized_minus_episode")
    for metric in METRICS:
        ep_value = episode[metric]
        pr_value = prioritized[metric]
        if isinstance(ep_value, float) and isinstance(pr_value, float):
            delta = pr_value - ep_value
        else:
            delta = "n/a"
        print(
            f"{metric},{_format(ep_value)},{_format(pr_value)},{_format(delta)}"
        )


if __name__ == "__main__":
    main()
