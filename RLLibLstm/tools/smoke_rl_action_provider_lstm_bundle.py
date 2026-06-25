"""Smoke-check RLActionProvider recurrent inference with a lightweight bundle.

This avoids ray.init() by constructing the patched SAC RLModule directly and
wrapping it in a tiny Algorithm-like object. The goal is to validate the same
RLActionProvider code path used by local/Unreal inference:

    bundle weights -> RLModule -> RLActionProvider.compute_action()
    STATE_OUT at step t -> STATE_IN at step t+1
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_imports() -> Path:
    root = _repo_root()
    mytrain = root / "DogFightEnv" / "MyTrainEnv"
    sys.path.insert(0, str(mytrain))
    sys.path.insert(0, str(mytrain / "src"))
    return mytrain


class _FakeEnvRunner:
    def __init__(self, module):
        self.module = module

    def set_state(self, state: dict) -> None:
        self.module.set_state(state["rl_module"])


class _FakeAlgorithm:
    def __init__(self, module):
        self._module = module
        self.env_runner = _FakeEnvRunner(module)

    def get_module(self, policy_id: str | None = None):
        return self._module

    def stop(self) -> None:
        return None


def _build_module_from_metadata(metadata: dict):
    from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
        DefaultSACTorchRLModule,
    )
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    payload_metadata = metadata.get("metadata", {})
    algorithm_config = metadata.get("algorithm_config", {})
    env_config = algorithm_config.get("env_config", {})

    obs_size = (
        metadata.get("observation_summary", {}).get("size")
        or payload_metadata.get("observation_size")
        or _infer_obs_size(env_config)
    )
    action_dim = payload_metadata.get("action_dim", 4)
    lstm_cell_size = payload_metadata.get("lstm_cell_size", 64)
    max_seq_len = payload_metadata.get("max_seq_len", 8)

    model_config = asdict(
        DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
            head_fcnet_hiddens=[],
            head_fcnet_activation="relu",
            use_lstm=True,
            max_seq_len=int(max_seq_len),
            lstm_cell_size=int(lstm_cell_size),
        )
    )
    model_config["twin_q"] = True

    module = RLModuleSpec(
        module_class=DefaultSACTorchRLModule,
        observation_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(obs_size),),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
        model_config=model_config,
    ).build()
    # Lightweight bundles are extracted from the full RLModule state, including
    # target Q networks. Build those attributes before applying state.
    module.make_target_networks()
    return module


def _infer_obs_size(env_config: dict) -> int:
    mode = env_config.get("observation_mode", "tactical16")
    return {
        "classic12": 12,
        "relative14": 14,
        "tactical16": 16,
        "legacy37": 37,
    }.get(mode, 16)


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

    from dogfight.ai.action_provider import ActionContext
    from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
    from dogfight.ai.rl_action_provider import RLActionProvider

    bundle_dir = Path(args.bundle_dir)
    metadata, _ = load_lightweight_policy_bundle(bundle_dir)
    obs_size = metadata.get("observation_summary", {}).get("size") or 16

    def algorithm_factory(bundle_metadata: dict):
        return _FakeAlgorithm(_build_module_from_metadata(bundle_metadata))

    provider = RLActionProvider(
        bundle_dir=str(bundle_dir),
        algorithm_factory=algorithm_factory,
    )

    print("[DogFightEnv][provider_smoke] bundle=", bundle_dir)
    print("[DogFightEnv][provider_smoke] use_lstm_sac=", metadata["metadata"].get("use_lstm_sac"))
    for step in range(args.steps):
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            observation=_make_observation(step, int(obs_size)),
        )
        result = provider.compute_action(context)
        print(
            "[DogFightEnv][provider_smoke] "
            f"step={step} action_shape={tuple(result.action.shape)} "
            f"action={result.action.tolist()} "
            f"state_norm={_state_norm(provider._module_state):.6f}"
        )

    provider.reset()
    print(
        "[DogFightEnv][provider_smoke] "
        f"after_reset_state={provider._module_state}"
    )


if __name__ == "__main__":
    main()
