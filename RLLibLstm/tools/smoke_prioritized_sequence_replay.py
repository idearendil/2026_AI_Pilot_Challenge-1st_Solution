"""Smoke-check patched prioritized sequence replay without Ray Algorithm build.

The script imports the patched `PrioritizedEpisodeReplayBuffer` copy stored under
RLLibLstm, creates simple episodes with monotonic observations, samples
`batch_length_T=8` sequences, and updates priorities with a sequence-shaped
TD-error array. It catches the two regressions observed in SAC LSTM work:

* prioritized replay returning only 1-step/padded samples;
* priority update failing on `(B, T)` TD errors.
"""

from __future__ import annotations

import importlib.util
import os
import argparse
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.single_agent_episode import SingleAgentEpisode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_patched_buffer_class():
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
        "dogfight_patched_prioritized_episode_buffer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load patched buffer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrioritizedEpisodeReplayBuffer


def _make_episode(offset: float, length: int = 24) -> SingleAgentEpisode:
    observation_space = Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--installed",
        action="store_true",
        help="Use the Ray class installed in site-packages instead of RLLibLstm patched copy.",
    )
    args = parser.parse_args()

    os.environ.setdefault("DOGFIGHT_PRIORITIZED_SEQ_DEBUG", "1")
    os.environ.setdefault("DOGFIGHT_PRIORITIZED_SEQ_DEBUG_LIMIT", "5")

    if args.installed:
        from ray.rllib.utils.replay_buffers.prioritized_episode_buffer import (
            PrioritizedEpisodeReplayBuffer,
        )

        buffer_cls = PrioritizedEpisodeReplayBuffer
    else:
        buffer_cls = _load_patched_buffer_class()
    buffer = buffer_cls(
        capacity=512,
        batch_size_B=16,
        batch_length_T=8,
        alpha=0.6,
    )
    buffer.add([_make_episode(0.0), _make_episode(100.0), _make_episode(200.0)])

    episodes = buffer.sample(
        batch_size_B=16,
        batch_length_T=8,
        n_step=1,
        beta=0.4,
    )
    lengths = [len(episode) for episode in episodes]
    probes = [
        [
            float(np.asarray(episode.get_observations(t))[0])
            for t in range(min(len(episode), 8))
        ]
        for episode in episodes[:3]
    ]
    print("[DogFightEnv][prioritized_seq_smoke] lengths=", lengths)
    print("[DogFightEnv][prioritized_seq_smoke] probes=", probes)
    print(
        "[DogFightEnv][prioritized_seq_smoke] last_sampled_indices=",
        len(buffer._last_sampled_indices),
    )

    if not lengths or max(lengths) < 8:
        raise RuntimeError(f"No sampled prioritized sequence reached length 8: {lengths}")
    if len(buffer._last_sampled_indices) != len(episodes):
        raise RuntimeError(
            "Expected one priority index per sampled sequence: "
            f"indices={len(buffer._last_sampled_indices)} episodes={len(episodes)}"
        )
    for probe in probes:
        if len(probe) >= 3 and abs(probe[0]) > 1e-8 and all(
            abs(value) <= 1e-8 for value in probe[1:]
        ):
            raise RuntimeError(f"Probe looks padded instead of sequential: {probe}")

    td_errors = np.arange(len(episodes) * 8, dtype=np.float32).reshape(len(episodes), 8)
    buffer.update_priorities(td_errors)
    if buffer._last_sampled_indices:
        raise RuntimeError("Priority update did not clear last sampled indices")

    print("[DogFightEnv][prioritized_seq_smoke] result=PASS")


if __name__ == "__main__":
    main()
