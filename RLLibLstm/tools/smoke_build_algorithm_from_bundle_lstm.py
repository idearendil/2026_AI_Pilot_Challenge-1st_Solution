"""Smoke-check real build_algorithm_from_bundle() with SAC LSTM bundle.

This script intentionally exercises the heavier inference path:

    lightweight bundle -> build_algorithm_from_bundle() -> RLActionProvider

It starts a local Ray instance through build_algorithm_from_bundle(). Always run
`ray stop --force` before this smoke if a previous Ray process may be alive.
The script calls provider.close() and ray.shutdown() in a finally block.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_imports() -> Path:
    root = _repo_root()
    mytrain = root / "DogFightEnv" / "MyTrainEnv"
    sys.path.insert(0, str(mytrain))
    sys.path.insert(0, str(mytrain / "src"))
    return mytrain


def _state_norm(state) -> float:
    if isinstance(state, dict):
        return float(sum(_state_norm(value) for value in state.values()))
    if hasattr(state, "detach"):
        return float(state.detach().float().abs().sum().cpu().item())
    return 0.0


def _make_observation(step: int, obs_size: int) -> np.ndarray:
    obs = np.zeros(obs_size, dtype=np.float32)
    obs[0] = np.float32(step / 100.0)
    if obs_size > 1:
        obs[1] = np.float32(-step / 200.0)
    return obs


def main() -> None:
    mytrain = _prepare_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default=str(mytrain / "artifacts" / "models" / "f16_single_agent" / "latest"),
    )
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    os.environ["DOGFIGHT_RNNSAC_DEBUG"] = "1"
    os.environ.setdefault("DOGFIGHT_RNNSAC_DEBUG_LIMIT", "20")

    import ray

    from dogfight.ai.action_provider import ActionContext
    from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
    from dogfight.ai.rl_action_provider import RLActionProvider
    from dogfight.ai.rllib_utils import build_algorithm_from_bundle

    provider = None
    try:
        metadata, _ = load_lightweight_policy_bundle(args.bundle_dir)
        obs_size = metadata.get("observation_summary", {}).get("size") or 16
        provider = RLActionProvider(
            bundle_dir=args.bundle_dir,
            algorithm_factory=build_algorithm_from_bundle,
        )

        print("[DogFightEnv][ray_bundle_smoke] bundle=", args.bundle_dir)
        print(
            "[DogFightEnv][ray_bundle_smoke] use_lstm_sac=",
            metadata["metadata"].get("use_lstm_sac"),
        )
        for step in range(args.steps):
            result = provider.compute_action(
                ActionContext(
                    sim=None,
                    opponent_sim=None,
                    observation=_make_observation(step, int(obs_size)),
                )
            )
            print(
                "[DogFightEnv][ray_bundle_smoke] "
                f"step={step} action_shape={tuple(result.action.shape)} "
                f"action={result.action.tolist()} "
                f"state_norm={_state_norm(provider._module_state):.6f}"
            )

        provider.reset()
        print(
            "[DogFightEnv][ray_bundle_smoke] "
            f"after_reset_state={provider._module_state}"
        )
    finally:
        if provider is not None:
            provider.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
