"""Smoke-check DogFightEnv's full recurrent actor-critic SAC RLModule path.

This script avoids ray.init() and full Algorithm construction. It validates the
local New API RLModule contract for Ray 1.9.2 RNNSAC-style Q networks:
actor LSTM state is supplied by the caller, while Q/twin/target Q LSTM state is
zero-initialized inside the module forward helper.
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
from ray.rllib.algorithms.sac.sac_learner import (
    QF_PREDS,
    QF_TWIN_PREDS,
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


def _assert_shape(name: str, value: object, expected: tuple[int, ...]) -> None:
    actual = _shape_summary(value)
    if actual != expected:
        raise AssertionError(f"{name} shape expected {expected}, got {actual}")


def main() -> None:
    os.environ["DOGFIGHT_RNNSAC_DEBUG"] = "1"
    os.environ["DOGFIGHT_RNNSAC_DEBUG_LIMIT"] = "5"

    batch_size = 2
    seq_len = 4
    obs_dim = 3
    action_dim = 2
    hidden = 8

    model_config = asdict(
        DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="relu",
            head_fcnet_hiddens=[],
            head_fcnet_activation="relu",
            use_lstm=True,
            max_seq_len=seq_len,
            lstm_cell_size=hidden,
        )
    )
    model_config["twin_q"] = True
    model_config["dogfight_lstm_scope"] = "actor_critic"
    model_config["dogfight_q_lstm_zero_init"] = True

    module = RLModuleSpec(
        module_class=DefaultSACTorchRLModule,
        observation_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        ),
        model_config=model_config,
    ).build()
    module.make_target_networks()

    for name in (
        "qf_encoder",
        "qf_twin_encoder",
        "target_qf_encoder",
        "target_qf_twin_encoder",
    ):
        encoder = getattr(module, name)
        if not hasattr(encoder, "get_initial_state"):
            raise AssertionError(f"{name} is not recurrent")

    actor_initial_state = module.get_initial_state()
    actor_state = {
        key: value.unsqueeze(0).expand(batch_size, *value.shape).contiguous()
        for key, value in actor_initial_state.items()
    }

    obs = (
        torch.arange(batch_size * seq_len * obs_dim, dtype=torch.float32)
        .reshape(batch_size, seq_len, obs_dim)
        / 100.0
    )
    actions = (
        torch.arange(batch_size * seq_len * action_dim, dtype=torch.float32)
        .reshape(batch_size, seq_len, action_dim)
        / 10.0
    )
    train_batch = {
        Columns.OBS: obs,
        Columns.NEXT_OBS: obs + 1.0,
        Columns.ACTIONS: actions,
        Columns.STATE_IN: actor_state,
        Columns.NEXT_STATE_IN: actor_state,
        Columns.SEQ_LENS: torch.tensor([seq_len, seq_len], dtype=torch.int32),
    }
    train_output = module.forward_train(train_batch)

    _assert_shape("qf_preds", train_output[QF_PREDS], (batch_size, seq_len))
    _assert_shape("qf_twin_preds", train_output[QF_TWIN_PREDS], (batch_size, seq_len))
    _assert_shape("q_curr", train_output["q_curr"], (batch_size, seq_len))
    _assert_shape(
        "q_target_next",
        train_output["q_target_next"],
        (batch_size, seq_len),
    )
    _assert_shape(
        "state_out_h",
        train_output[Columns.STATE_OUT]["h"],
        (batch_size, 1, hidden),
    )
    _assert_shape(
        "next_state_out_h",
        train_output[Columns.NEXT_STATE_OUT]["h"],
        (batch_size, 1, hidden),
    )

    q_records = getattr(module, "_dogfight_q_debug_records", [])
    if not q_records:
        raise AssertionError("Q debug records were not produced")
    for record in q_records:
        if record["q_concat_shape"] != (batch_size, seq_len, obs_dim + action_dim):
            raise AssertionError(f"bad q concat shape: {record}")
        if record["q_state_in"]["h"] != (batch_size, 1, hidden):
            raise AssertionError(f"bad q state_in shape: {record}")

    print("[DogFightEnv][actor_critic_smoke] is_stateful=", module.is_stateful())
    print("[DogFightEnv][actor_critic_smoke] actor_state=", _shape_summary(actor_state))
    print(
        "[DogFightEnv][actor_critic_smoke] train_output_shapes=",
        {
            QF_PREDS: _shape_summary(train_output[QF_PREDS]),
            QF_TWIN_PREDS: _shape_summary(train_output[QF_TWIN_PREDS]),
            "q_curr": _shape_summary(train_output["q_curr"]),
            "q_target_next": _shape_summary(train_output["q_target_next"]),
        },
    )
    print("[DogFightEnv][actor_critic_smoke] q_debug_first=", q_records[0])
    print("[DogFightEnv][actor_critic_smoke] PASS")


if __name__ == "__main__":
    main()
