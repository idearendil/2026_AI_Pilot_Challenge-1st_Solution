"""Smoke-check DogFightEnv's patched SAC actor-LSTM RLModule path.

This script intentionally does not call ray.init() or build a full Algorithm.
It validates the local RLModule input/output contract that matters before
running a heavier RLlib training smoke.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch

from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def _shape_summary(value: object) -> object:
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return type(value).__name__


def main() -> None:
    os.environ["DOGFIGHT_RNNSAC_DEBUG"] = "1"
    os.environ["DOGFIGHT_RNNSAC_DEBUG_LIMIT"] = "5"

    model_config = asdict(
        DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="relu",
            head_fcnet_hiddens=[],
            head_fcnet_activation="relu",
            use_lstm=True,
            max_seq_len=4,
            lstm_cell_size=8,
        )
    )
    # SAC catalog still expects twin_q in the New API model config dict.
    model_config["twin_q"] = True

    module = RLModuleSpec(
        module_class=DefaultSACTorchRLModule,
        observation_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        ),
        model_config=model_config,
    ).build()
    module.make_target_networks()

    initial_state = module.get_initial_state()
    batched_state = {
        key: value.unsqueeze(0) for key, value in initial_state.items()
    }
    batch = {
        Columns.OBS: torch.zeros(1, 1, 3),
        Columns.STATE_IN: batched_state,
        Columns.SEQ_LENS: torch.tensor([1], dtype=torch.int32),
    }
    output = module.forward_inference(batch)
    train_batch = {
        Columns.OBS: torch.zeros(1, 4, 3),
        Columns.NEXT_OBS: torch.ones(1, 4, 3),
        Columns.ACTIONS: torch.zeros(1, 4, 2),
        Columns.STATE_IN: batched_state,
        Columns.NEXT_STATE_IN: batched_state,
        Columns.SEQ_LENS: torch.tensor([4], dtype=torch.int32),
    }
    train_output = module.forward_train(train_batch)

    print("[DogFightEnv][smoke] is_stateful=", module.is_stateful())
    print("[DogFightEnv][smoke] state_in=", _shape_summary(batched_state))
    print("[DogFightEnv][smoke] output_keys=", sorted(output.keys()))
    print(
        "[DogFightEnv][smoke] state_out=",
        _shape_summary(output.get(Columns.STATE_OUT)),
    )
    print(
        "[DogFightEnv][smoke] action_dist_inputs=",
        _shape_summary(output.get(Columns.ACTION_DIST_INPUTS)),
    )
    print("[DogFightEnv][smoke] train_output_keys=", sorted(train_output.keys()))
    print(
        "[DogFightEnv][smoke] train_state_out=",
        _shape_summary(train_output.get(Columns.STATE_OUT)),
    )
    print(
        "[DogFightEnv][smoke] train_next_state_out=",
        _shape_summary(train_output.get(Columns.NEXT_STATE_OUT)),
    )


if __name__ == "__main__":
    main()
