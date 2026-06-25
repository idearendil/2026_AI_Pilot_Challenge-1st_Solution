"""Verify patched prioritized sequence replay contracts for SAC LSTM.

This is a no-Ray-Algorithm integrity check. It exercises the patched
`PrioritizedEpisodeReplayBuffer` with deterministic monotonic observations and
checks the failure modes that matter before running longer SAC LSTM training:

* sampled sequences are longer than 1 step when enough data exists;
* observations inside each sampled slice stay in chronological order;
* sampled slices do not cross source episode boundaries;
* one priority tree index is tracked per sampled sequence;
* scalar, vector, matrix, and flat TD-error priority updates are accepted.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Iterable, Type

import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.single_agent_episode import SingleAgentEpisode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_patched_buffer_class() -> Type:
    path = (
        _repo_root()
        / "RLLibLstm"
        / "ray_2_54_0_patched"
        / "ray"
        / "rllib"
        / "utils"
        / "replay_buffers"
        / "prioritized_episode_buffer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dogfight_integrity_prioritized_episode_buffer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load patched buffer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrioritizedEpisodeReplayBuffer


def _load_installed_buffer_class() -> Type:
    from ray.rllib.utils.replay_buffers.prioritized_episode_buffer import (
        PrioritizedEpisodeReplayBuffer,
    )

    return PrioritizedEpisodeReplayBuffer


def _make_episode(offset: float, length: int) -> SingleAgentEpisode:
    observation_space = Box(-100000.0, 100000.0, shape=(1,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    episode = SingleAgentEpisode(
        observation_space=observation_space,
        action_space=action_space,
    )
    episode.add_env_reset(np.asarray([offset], dtype=np.float32), infos={})
    for step in range(length):
        episode.add_env_step(
            observation=np.asarray([offset + step + 1], dtype=np.float32),
            action=np.asarray([step / 100.0], dtype=np.float32),
            reward=float(step),
            infos={},
            terminated=step == length - 1,
        )
    return episode


def _make_buffer(buffer_cls: Type, batch_length_t: int, batch_size_b: int):
    buffer = buffer_cls(
        capacity=4096,
        batch_size_B=batch_size_b,
        batch_length_T=batch_length_t,
        alpha=0.6,
    )
    episodes = [
        _make_episode(0.0, 6),
        _make_episode(1000.0, 9),
        _make_episode(2000.0, 17),
        _make_episode(3000.0, 33),
    ]
    buffer.add(episodes)
    return buffer


def _obs_values(episode: SingleAgentEpisode) -> list[float]:
    return [
        float(np.asarray(episode.get_observations(t))[0])
        for t in range(len(episode))
    ]


def _check_sequence_values(values: Iterable[float], batch_length_t: int) -> None:
    values = list(values)
    if not values:
        raise RuntimeError("Sampled an empty sequence")
    if len(values) > batch_length_t:
        raise RuntimeError(
            f"Sequence exceeds requested length {batch_length_t}: {values}"
        )
    buckets = {int(value // 1000) for value in values if value >= 1000}
    if len(buckets) > 1:
        raise RuntimeError(f"Sequence crossed episode offset buckets: {values}")
    diffs = np.diff(values)
    if len(diffs) and not np.allclose(diffs, 1.0, atol=1e-5):
        raise RuntimeError(f"Sequence is not chronological +1 order: {values}")


def _sample_and_validate(
    buffer,
    *,
    batch_size_b: int,
    batch_length_t: int,
    min_batch_length_t: int = 0,
):
    episodes = buffer.sample(
        batch_size_B=batch_size_b,
        batch_length_T=batch_length_t,
        min_batch_length_T=min_batch_length_t,
        n_step=1,
        beta=0.4,
        include_extra_model_outputs=True,
    )
    lengths = [len(episode) for episode in episodes]
    if not lengths:
        raise RuntimeError("No sequences sampled")
    if max(lengths) < batch_length_t:
        raise RuntimeError(
            f"No sampled sequence reached length {batch_length_t}: {lengths}"
        )
    if len(buffer._last_sampled_indices) != len(episodes):
        raise RuntimeError(
            "Expected one priority index per sampled sequence: "
            f"indices={len(buffer._last_sampled_indices)} episodes={len(episodes)}"
        )
    for episode in episodes:
        values = _obs_values(episode)
        _check_sequence_values(values, batch_length_t)
        weights = episode.get_extra_model_outputs("weights", slice(0, len(episode)))
        n_steps = episode.get_extra_model_outputs("n_step", slice(0, len(episode)))
        if len(weights) != len(episode) or len(n_steps) != len(episode):
            raise RuntimeError(
                "weights/n_step length mismatch: "
                f"len={len(episode)} weights={len(weights)} n_steps={len(n_steps)}"
            )
        if not np.all(np.asarray(weights, dtype=np.float64) > 0.0):
            raise RuntimeError(f"Non-positive importance weights: {weights}")
        if not np.all(np.asarray(n_steps, dtype=np.int64) == 1):
            raise RuntimeError(f"Unexpected n_step values: {n_steps}")
    return episodes, lengths


def _update_and_validate(buffer, priorities, label: str) -> None:
    expected = len(buffer._last_sampled_indices)
    if expected <= 0:
        raise RuntimeError(f"No sampled indices before {label} priority update")
    buffer.update_priorities(priorities)
    if buffer._last_sampled_indices:
        raise RuntimeError(f"{label} priority update did not clear sampled indices")


def _run_case(
    name: str,
    buffer_cls: Type,
    *,
    batch_length_t: int,
    batch_size_b: int,
    rounds: int,
) -> None:
    buffer = _make_buffer(buffer_cls, batch_length_t, batch_size_b)
    all_lengths: list[int] = []
    update_modes = ("matrix", "flat", "vector", "scalar")
    for round_index in range(rounds):
        episodes, lengths = _sample_and_validate(
            buffer,
            batch_size_b=batch_size_b,
            batch_length_t=batch_length_t,
            min_batch_length_t=4,
        )
        all_lengths.extend(lengths)
        count = len(episodes)
        mode = update_modes[round_index % len(update_modes)]
        if mode == "matrix":
            priorities = np.arange(
                count * batch_length_t,
                dtype=np.float32,
            ).reshape(count, batch_length_t)
        elif mode == "flat":
            priorities = np.arange(count * batch_length_t, dtype=np.float32)
        elif mode == "vector":
            priorities = np.linspace(0.1, 2.0, count, dtype=np.float32)
        else:
            priorities = np.asarray(0.75, dtype=np.float32)
        _update_and_validate(buffer, priorities, mode)

    counts = {
        length: all_lengths.count(length)
        for length in sorted(set(all_lengths))
    }
    print(
        f"[DogFightEnv][prioritized_seq_integrity] case={name} "
        f"rounds={rounds} sampled_sequences={len(all_lengths)} "
        f"length_counts={counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["patched", "installed", "both"],
        default="installed",
        help="Which buffer implementation to verify.",
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--batch-size-b", type=int, default=64)
    parser.add_argument("--batch-length-t", type=int, default=8)
    parser.add_argument("--debug-samples", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("DOGFIGHT_PRIORITIZED_SEQ_DEBUG", "1")
    os.environ["DOGFIGHT_PRIORITIZED_SEQ_DEBUG_LIMIT"] = str(args.debug_samples)

    targets: list[tuple[str, Type]] = []
    if args.target in {"patched", "both"}:
        targets.append(("patched", _load_patched_buffer_class()))
    if args.target in {"installed", "both"}:
        targets.append(("installed", _load_installed_buffer_class()))

    for name, buffer_cls in targets:
        _run_case(
            name,
            buffer_cls,
            batch_length_t=args.batch_length_t,
            batch_size_b=args.batch_size_b,
            rounds=args.rounds,
        )

    print("[DogFightEnv][prioritized_seq_integrity] result=PASS")


if __name__ == "__main__":
    main()
