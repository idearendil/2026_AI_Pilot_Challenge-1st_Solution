from typing import Any, Dict

import os

import gymnasium as gym

from ray.rllib.algorithms.sac.default_sac_rl_module import DefaultSACRLModule
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_DIST_INPUTS_NEXT,
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    QF_PREDS,
    QF_TARGET_NEXT,
    QF_TWIN_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder, Model
from ray.rllib.core.rl_module.apis import QNetAPI, TargetNetworkAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.util.annotations import DeveloperAPI

torch, nn = try_import_torch()


@DeveloperAPI
class DefaultSACTorchRLModule(TorchRLModule, DefaultSACRLModule):
    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = SACCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def _forward_inference(self, batch: Dict) -> Dict[str, Any]:
        output = {}

        # DogFightEnv SAC actor-LSTM patch:
        # When model_config.use_lstm=True, RLlib's TorchLSTMEncoder requires
        # batch[Columns.STATE_IN] and returns Columns.STATE_OUT. We pass the batch
        # through unchanged so ConnectorV2 and RLActionProvider can manage state.
        pi_encoder_outs = self.pi_encoder(batch)
        if Columns.STATE_OUT in pi_encoder_outs:
            output[Columns.STATE_OUT] = pi_encoder_outs[Columns.STATE_OUT]
            self._dogfight_debug_lstm_io("inference", batch, output)

        # Pi head.
        # assume action space is either discrete or continuous.
        output[Columns.ACTION_DIST_INPUTS] = self.pi(pi_encoder_outs[ENCODER_OUT])

        return output

    @override(RLModule)
    def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        return self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch: Dict) -> Dict[str, Any]:
        # Call the `super`'s `forward_train`
        super()._forward_train(batch)
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self._forward_train_discrete(batch)
        elif isinstance(self.action_space, gym.spaces.Box):
            return self._forward_train_continuous(batch)
        else:
            raise ValueError(
                f"Unsupported action space type: {type(self.action_space)}. "
                "Only discrete and continuous action spaces are supported."
            )

    def _forward_train_discrete(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = {}

        # SAC needs also Q function values and action logits for next observations.
        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        ## calculate values for the Q target ##
        # Also encode the next observations (and next actions for the Q net).
        pi_encoder_next_outs = self.pi_encoder(batch_next)
        action_logits_next = self.pi(pi_encoder_next_outs[ENCODER_OUT])
        # TODO(inyoung): get the action dist class and use that. But currently TorchCategorical
        # does not get the prob value of the actual torch distribution. So we use softmax directly
        # for now.
        action_probs_next = torch.nn.functional.softmax(action_logits_next, dim=-1)

        output[ACTION_PROBS_NEXT] = action_probs_next
        output[ACTION_LOG_PROBS_NEXT] = action_probs_next.log()

        # (B, action_dim)
        qf_target_next = self.forward_target(batch_next, squeeze=False)
        output[QF_TARGET_NEXT] = qf_target_next

        qf_preds = self._qf_forward_train_helper(
            batch_curr, self.qf_encoder, self.qf, squeeze=False
        )
        # we don't need straight-through gradient here
        output[QF_PREDS] = qf_preds
        if self.twin_q:
            qf_twin_preds = self._qf_forward_train_helper(
                batch_curr, self.qf_twin_encoder, self.qf_twin, squeeze=False
            )
            output[QF_TWIN_PREDS] = qf_twin_preds

        ## calculate values for gradient ##
        pi_encoder_outs = self.pi_encoder(batch_curr)
        action_logits = self.pi(pi_encoder_outs[ENCODER_OUT])
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        output[ACTION_PROBS] = action_probs
        output[ACTION_LOG_PROBS] = action_probs.log()

        return output

    def _forward_train_continuous(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        self._dogfight_q_debug_records = []

        # SAC needs also Q function values and action logits for next observations.
        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # DogFightEnv SAC actor-LSTM patch:
        # AddStatesFromEpisodesToBatch supplies STATE_IN/NEXT_STATE_IN when the
        # module is stateful. Passing them here prevents the classic failure mode
        # where replay samples are transitions instead of sequences.
        if Columns.STATE_IN in batch:
            batch_curr[Columns.STATE_IN] = batch[Columns.STATE_IN]
        if Columns.NEXT_STATE_IN in batch:
            batch_next[Columns.STATE_IN] = batch[Columns.NEXT_STATE_IN]

        # Encoder forward passes.
        pi_encoder_outs = self.pi_encoder(batch_curr)
        if Columns.STATE_OUT in pi_encoder_outs:
            output[Columns.STATE_OUT] = pi_encoder_outs[Columns.STATE_OUT]

        # Also encode the next observations (and next actions for the Q net).
        pi_encoder_next_outs = self.pi_encoder(batch_next)
        if Columns.STATE_OUT in pi_encoder_next_outs:
            output[Columns.NEXT_STATE_OUT] = pi_encoder_next_outs[Columns.STATE_OUT]

        # Q-network(s) forward passes. In dogfight_lstm_scope=actor_critic,
        # the helper below builds [obs, action] and zero-initializes critic LSTM
        # state internally so actor STATE_IN can never be reused by Q networks.
        batch_curr.update({Columns.ACTIONS: batch[Columns.ACTIONS]})
        output[QF_PREDS] = self._qf_forward_train_helper(
            batch_curr,
            self.qf_encoder,
            self.qf,
            debug_label="qf_rollout",
        )  # self._qf_forward_train(batch_curr)[QF_PREDS]
        # If necessary make a forward pass through the twin Q network.
        if self.twin_q:
            output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                batch_curr,
                self.qf_twin_encoder,
                self.qf_twin,
                debug_label="qf_twin_rollout",
            )

        # Policy head.
        action_logits = self.pi(pi_encoder_outs[ENCODER_OUT])
        # Also get the action logits for the next observations.
        action_logits_next = self.pi(pi_encoder_next_outs[ENCODER_OUT])
        output[Columns.ACTION_DIST_INPUTS] = action_logits
        output[ACTION_DIST_INPUTS_NEXT] = action_logits_next

        # Get the train action distribution for the current policy and current state.
        # This is needed for the policy (actor) loss in SAC.
        action_dist_class = self.get_train_action_dist_cls()
        action_dist_curr = action_dist_class.from_logits(action_logits)
        # Get the train action distribution for the current policy and next state.
        # For the Q (critic) loss in SAC, we need to sample from the current policy at
        # the next state.
        action_dist_next = action_dist_class.from_logits(action_logits_next)

        # Sample actions for the current state. Note that we need to apply the
        # reparameterization trick (`rsample()` instead of `sample()`) to avoid the
        # expectation over actions.
        actions_resampled = action_dist_curr.rsample()
        # Compute the log probabilities for the current state (for the critic loss).
        output["logp_resampled"] = action_dist_curr.logp(actions_resampled)

        # Sample actions for the next state.
        actions_next_resampled = action_dist_next.sample().detach()
        # Compute the log probabilities for the next state.
        output["logp_next_resampled"] = (
            action_dist_next.logp(actions_next_resampled)
        ).detach()

        # Compute Q-values for the current policy in the current state with
        # the sampled actions.
        q_batch_curr = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: actions_resampled,
        }
        # Make sure we perform a "straight-through gradient" pass here,
        # ignoring the gradients of the q-net, however, still recording
        # the gradients of the policy net (which was used to rsample the actions used
        # here). This is different from doing `.detach()` or `with torch.no_grads()`,
        # as these two methds would fully block all gradient recordings, including
        # the needed policy ones.
        all_params = list(self.qf.parameters()) + list(self.qf_encoder.parameters())
        if self.twin_q:
            all_params += list(self.qf_twin.parameters()) + list(
                self.qf_twin_encoder.parameters()
            )

        for param in all_params:
            param.requires_grad = False
        output["q_curr"] = self.compute_q_values(q_batch_curr)
        for param in all_params:
            param.requires_grad = True

        # Compute Q-values from the target Q network for the next state with the
        # sampled actions for the next state.
        q_batch_next = {
            Columns.OBS: batch[Columns.NEXT_OBS],
            Columns.ACTIONS: actions_next_resampled,
        }
        output["q_target_next"] = self.forward_target(q_batch_next).detach()

        self._dogfight_debug_lstm_io("train_continuous", batch, output)

        # Return the network outputs.
        return output

    def _dogfight_debug_lstm_io(self, label: str, batch: Dict, output: Dict) -> None:
        """Print recurrent SAC batch contracts for visual inspection.

        Enable with DOGFIGHT_RNNSAC_DEBUG=1. The obs_probe line is intentionally
        included to catch reversed sequence order or transition-only replay batches.
        """
        if os.environ.get("DOGFIGHT_RNNSAC_DEBUG") != "1":
            return
        limit = int(os.environ.get("DOGFIGHT_RNNSAC_DEBUG_LIMIT", "20"))
        count = getattr(self, "_dogfight_lstm_debug_count", 0)
        if count >= limit:
            return
        setattr(self, "_dogfight_lstm_debug_count", count + 1)

        def shape(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return {k: shape(v) for k, v in value.items()}
            return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__

        obs = batch.get(Columns.OBS)
        obs_probe = None
        if hasattr(obs, "detach") and obs.ndim >= 3 and obs.shape[-1] > 0:
            obs_probe = obs[0, :, 0].detach().cpu().tolist()
        q_debug = getattr(self, "_dogfight_q_debug_records", None)
        if q_debug:
            q_debug = q_debug[:6]
        print(
            "[DogFightEnv][RLlibSAC][LSTM_IO] "
            f"label={label} "
            f"obs_shape={shape(batch.get(Columns.OBS))} "
            f"next_obs_shape={shape(batch.get(Columns.NEXT_OBS))} "
            f"actions_shape={shape(batch.get(Columns.ACTIONS))} "
            f"seq_lens={batch.get(Columns.SEQ_LENS)} "
            f"state_in={shape(batch.get(Columns.STATE_IN))} "
            f"next_state_in={shape(batch.get(Columns.NEXT_STATE_IN))} "
            f"state_out={shape(output.get(Columns.STATE_OUT))} "
            f"next_state_out={shape(output.get(Columns.NEXT_STATE_OUT))} "
            f"obs_probe_first_feature={obs_probe} "
            f"q_debug={q_debug}"
        )

    def _dogfight_is_actor_critic_lstm(self) -> bool:
        """Return whether DogFightEnv recurrent Q networks are enabled."""
        model_config = getattr(self, "model_config", {}) or {}
        return model_config.get("dogfight_lstm_scope") == "actor_critic"

    def _dogfight_zero_state_for_encoder(
        self,
        encoder: Encoder,
        reference_tensor,
    ) -> Dict[str, Any] | None:
        """Create batch-major zero state for a Q LSTM encoder.

        Q state is deliberately local to the learner forward pass in v1. This
        mirrors the old zero_init_states=True RNNSAC contract while avoiding any
        accidental reuse of the actor recurrent state from replay.
        """
        if not hasattr(encoder, "get_initial_state"):
            return None
        initial_state = encoder.get_initial_state()
        batch_size = reference_tensor.shape[0]

        def batch_state(value):
            if not hasattr(value, "to"):
                value = torch.as_tensor(value)
            value = value.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )
            return value.unsqueeze(0).expand(batch_size, *value.shape).contiguous()

        return {key: batch_state(value) for key, value in initial_state.items()}

    def _dogfight_record_q_debug(
        self,
        label: str | None,
        q_obs_action,
        q_state_in,
        q_state_out,
        qf_out,
    ) -> None:
        """Record recurrent Q I/O probes for DOGFIGHT_RNNSAC_DEBUG prints."""
        if not self._dogfight_is_actor_critic_lstm():
            return
        if os.environ.get("DOGFIGHT_RNNSAC_DEBUG") != "1":
            return

        def shape(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return {key: shape(item) for key, item in value.items()}
            return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__

        obs_probe = None
        action_probe = None
        if hasattr(q_obs_action, "detach") and q_obs_action.ndim >= 3:
            if q_obs_action.shape[-1] > 0:
                obs_probe = q_obs_action[0, :, 0].detach().cpu().tolist()
            obs_dim = getattr(self.observation_space, "shape", (0,))[0]
            if q_obs_action.shape[-1] > obs_dim:
                action_probe = q_obs_action[0, :, obs_dim].detach().cpu().tolist()

        record = {
            "label": label,
            "q_concat_shape": shape(q_obs_action),
            "q_state_in": shape(q_state_in),
            "q_state_out": shape(q_state_out),
            "q_out_shape": shape(qf_out),
            "q_obs_probe_first_feature": obs_probe,
            "q_action_probe_first_feature": action_probe,
        }
        records = getattr(self, "_dogfight_q_debug_records", None)
        if records is None:
            records = []
            self._dogfight_q_debug_records = records
        records.append(record)

    @override(TargetNetworkAPI)
    def forward_target(
        self, batch: Dict[str, Any], squeeze: bool = True
    ) -> Dict[str, Any]:
        target_qvs = self._qf_forward_train_helper(
            batch,
            self.target_qf_encoder,
            self.target_qf,
            squeeze=squeeze,
            debug_label="target_qf",
        )

        # If a twin Q network should be used, calculate twin Q-values and use the
        # minimum.
        if self.twin_q:
            target_qvs = torch.min(
                target_qvs,
                self._qf_forward_train_helper(
                    batch,
                    self.target_qf_twin_encoder,
                    self.target_qf_twin,
                    squeeze=squeeze,
                    debug_label="target_qf_twin",
                ),
            )

        return target_qvs

    @override(QNetAPI)
    def compute_q_values(
        self, batch: Dict[str, Any], squeeze: bool = True
    ) -> Dict[str, Any]:
        qvs = self._qf_forward_train_helper(
            batch,
            self.qf_encoder,
            self.qf,
            squeeze=squeeze,
            debug_label="q_curr_qf",
        )
        # If a twin Q network should be used, calculate twin Q-values and use the
        # minimum.
        if self.twin_q:
            qvs = torch.min(
                qvs,
                self._qf_forward_train_helper(
                    batch,
                    self.qf_twin_encoder,
                    self.qf_twin,
                    squeeze=squeeze,
                    debug_label="q_curr_qf_twin",
                ),
            )
        return qvs

    @override(DefaultSACRLModule)
    def _qf_forward_train_helper(
        self,
        batch: Dict[str, Any],
        encoder: Encoder,
        head: Model,
        squeeze: bool = True,
        debug_label: str | None = None,
    ) -> Dict[str, Any]:
        """Executes the forward pass for Q networks.

        Args:
            batch: Dict containing a concatenated tensor with observations
                and actions under the key `Columns.OBS`.
            encoder: An `Encoder` model for the Q state-action encoder.
            head: A `Model` for the Q head.
            squeeze: If True, squeezes the last dimension of the output if it is 1. Used for continuous action spaces.

        Returns:
            The estimated Q-value for the input action for continuous action spaces.
            Or the Q-values for all actions for discrete action spaces.
        """
        # Construct batch. Note, we need to feed observations and actions.
        if isinstance(self.action_space, gym.spaces.Box):
            actions = batch[Columns.ACTIONS]
            # Contract: [obs, action] is the only supported continuous-Q input
            # order for Ray 1.9.2 RNNSAC compatibility. Do not move action behind
            # the LSTM or swap this concat order.
            q_obs_action = torch.concat((batch[Columns.OBS], actions), dim=-1)
            qf_batch = {
                Columns.OBS: q_obs_action
            }
        else:
            # For discrete action spaces, we don't need to include the actions
            # in the batch, as the Q function outputs the Q-values for each action
            q_obs_action = batch[Columns.OBS]
            qf_batch = {Columns.OBS: q_obs_action}

        q_state_in = None
        if self._dogfight_is_actor_critic_lstm():
            q_state_in = self._dogfight_zero_state_for_encoder(
                encoder,
                qf_batch[Columns.OBS],
            )
            if q_state_in is not None:
                qf_batch[Columns.STATE_IN] = q_state_in

        # Encoder forward pass.
        qf_encoder_outs = encoder(qf_batch)

        # Q head forward pass.
        # (B,latent_size) -> (B, 1|action_dim)
        qf_out = head(qf_encoder_outs[ENCODER_OUT])
        if squeeze:
            # Squeeze the last dimension if it is 1.
            qf_out = qf_out.squeeze(-1)
        self._dogfight_record_q_debug(
            debug_label,
            q_obs_action,
            q_state_in,
            qf_encoder_outs.get(Columns.STATE_OUT),
            qf_out,
        )
        return qf_out
