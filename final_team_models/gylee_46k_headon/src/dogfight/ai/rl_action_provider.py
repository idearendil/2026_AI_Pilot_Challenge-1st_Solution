from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle


class RLActionProvider(ActionProvider):
    def __init__(
        self,
        algorithm=None,
        bundle_dir: str | None = None,
        algorithm_factory: Callable[[dict], object] | None = None,
        policy_id: str = "default_policy",
        explore: bool = False,
        obs_builder: Callable[[ActionContext], np.ndarray] | None = None,
        confidence: float = 0.9,
    ):
        self.policy_id = policy_id
        self.explore = explore
        self.obs_builder = obs_builder
        self.confidence = confidence
        self._owns_algorithm = False
        self._module_state = None
        self._debug_lstm_io = False
        self._debug_lstm_prints = 0

        if algorithm is not None:
            self.algorithm = algorithm
            self.metadata = None
        elif bundle_dir is not None:
            if algorithm_factory is None:
                raise ValueError("algorithm_factory is required when loading a lightweight bundle")
            self.metadata, weights = load_lightweight_policy_bundle(Path(bundle_dir))
            self.algorithm = algorithm_factory(self.metadata)
            try:
                # Old API stack: Policy-based weight loading
                self.algorithm.get_policy(self.policy_id).set_weights(weights)
            except AttributeError:
                # New API stack: load directly into the local RLModule state.
                env_runner = getattr(self.algorithm, "env_runner", None)
                if env_runner is not None and hasattr(env_runner, "set_state"):
                    env_runner.set_state({"rl_module": weights})
                elif env_runner is not None and hasattr(env_runner, "module"):
                    env_runner.module.set_state(weights)
                else:
                    raise RuntimeError(
                        "Unable to apply RLModule weights to the RLlib algorithm"
                    )
            self._owns_algorithm = True
        else:
            raise ValueError("Either algorithm or bundle_dir must be provided")

    def reset(self, context: ActionContext | None = None) -> None:
        """Reset recurrent module state at episode boundaries."""
        had_state = self._module_state is not None
        self._module_state = None
        self._debug_lstm_prints = 0
        if self._debug_lstm_io:
            frame_index = None
            if context is not None and isinstance(context.info, dict):
                frame_index = context.info.get("frame_index")
            print(
                "[DogFightEnv][RLActionProvider][LSTM_RESET] "
                f"had_state={had_state} frame_index={frame_index}"
            )

    def compute_action(self, context: ActionContext) -> ActionResult:
        observation = context.observation
        if observation is None:
            if self.obs_builder is None:
                raise ValueError("observation is required when obs_builder is not configured")
            observation = self.obs_builder(context)

        action = self._compute_module_action(observation)
        action = clip_action(action)

        return ActionResult(
            action=action,
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": self.explore},
        )

    def close(self) -> None:
        if self._owns_algorithm and hasattr(self.algorithm, "stop"):
            self.algorithm.stop()

    def _compute_module_action(self, observation: np.ndarray) -> np.ndarray:
        """Compute an action through Ray 2.54 RLModule inference APIs."""
        import os
        import torch
        from ray.rllib.core.columns import Columns

        module = self.algorithm.get_module(self.policy_id)
        if module is None:
            module = self.algorithm.get_module()
        if module is None:
            raise RuntimeError("Unable to find an RLModule for action inference")

        self._debug_lstm_io = os.environ.get("DOGFIGHT_RNNSAC_DEBUG") == "1"
        self._ensure_module_state(module, torch)

        obs = np.asarray(observation, dtype=np.float32)
        if self._module_state:
            # Recurrent RLModules expect obs as (B, T, obs_dim). This explicit
            # time axis is the inference-side mirror of replay sequence batches.
            obs_batch = obs[None, None, :]
        else:
            obs_batch = obs[None, :]
        batch = {Columns.OBS: torch.as_tensor(obs_batch, dtype=torch.float32)}
        if self._module_state:
            batch[Columns.STATE_IN] = self._module_state
            batch[Columns.SEQ_LENS] = torch.as_tensor([1], dtype=torch.int32)

        with torch.no_grad():
            output = module.forward_inference(batch)
            if Columns.STATE_OUT in output:
                self._module_state = _detach_state(output[Columns.STATE_OUT])
                self._print_lstm_io_debug(batch, output)

            if Columns.ACTIONS in output:
                action = output[Columns.ACTIONS]
            else:
                logits = output[Columns.ACTION_DIST_INPUTS]
                if self.explore:
                    dist_class = module.get_exploration_action_dist_cls()
                else:
                    dist_class = module.get_inference_action_dist_cls()
                action_dist = dist_class.from_logits(logits)
                if not self.explore:
                    action_dist = action_dist.to_deterministic()
                action = action_dist.sample()

        return _to_numpy_action(action)

    def _ensure_module_state(self, module, torch_module) -> None:
        """Create a batched recurrent state if the loaded RLModule is stateful."""
        if self._module_state is not None:
            return
        if not hasattr(module, "get_initial_state"):
            return
        initial_state = module.get_initial_state()
        if not initial_state:
            return
        self._module_state = _batch_state(initial_state, torch_module)

    def _print_lstm_io_debug(self, batch: dict, output: dict) -> None:
        """Print the recurrent inference contract for visual inspection."""
        if not self._debug_lstm_io or self._debug_lstm_prints >= 20:
            return
        from ray.rllib.core.columns import Columns

        obs = batch[Columns.OBS]
        state_in = batch.get(Columns.STATE_IN)
        state_out = output.get(Columns.STATE_OUT)
        print(
            "[DogFightEnv][RLActionProvider][LSTM_IO] "
            f"obs_shape={tuple(obs.shape)} "
            f"seq_lens={batch.get(Columns.SEQ_LENS)} "
            f"state_in={_state_shape_summary(state_in)} "
            f"state_out={_state_shape_summary(state_out)}"
        )
        self._debug_lstm_prints += 1


def _to_numpy_action(action) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action_array = np.asarray(action, dtype=np.float32)
    while action_array.ndim > 1 and action_array.shape[0] == 1:
        action_array = action_array[0]
    return action_array


def _batch_state(state, torch_module):
    if isinstance(state, dict):
        return {key: _batch_state(value, torch_module) for key, value in state.items()}
    tensor = state if hasattr(state, "detach") else torch_module.as_tensor(state)
    return tensor.detach().clone().float().unsqueeze(0)


def _detach_state(state):
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    return state.detach().clone() if hasattr(state, "detach") else state


def _state_shape_summary(state) -> str:
    if state is None:
        return "None"
    if isinstance(state, dict):
        return "{" + ", ".join(
            f"{key}: {_state_shape_summary(value)}"
            for key, value in state.items()
        ) + "}"
    shape = getattr(state, "shape", None)
    return str(tuple(shape)) if shape is not None else type(state).__name__
