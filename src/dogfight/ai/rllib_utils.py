from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED_ALGORITHMS = {
    "ppo": "PPO",
    "sac": "SAC",
}

SERIALIZED_OBJECT_KEYS = {
    "class",
    "policy_mapping_fn",
    "sample_collector",
    "stats_cls_lookup",
}


def normalize_algorithm_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {name!r}. Supported: {', '.join(sorted(SUPPORTED_ALGORITHMS))}")
    return normalized


def build_algorithm_config(algorithm_name: str, env_name: str, env_config: dict, args: dict[str, Any]):
    algorithm_name = normalize_algorithm_name(algorithm_name)

    if algorithm_name == "ppo":
        from ray.rllib.algorithms.ppo import PPOConfig

        config = PPOConfig()
        config = config.training(
            lr=args["lr"],
            gamma=args["gamma"],
            train_batch_size=args["train_batch_size"],
            minibatch_size=args["minibatch_size"],
            lambda_=args["gae_lambda"],
            clip_param=args["clip_param"],
        )
        if args.get("use_lstm"):
            config = _apply_lstm_model_config(config, args)
    elif algorithm_name == "sac":
        from ray.rllib.algorithms.sac import SACConfig

        replay_buffer_config = _build_replay_buffer_config(args)
        if args.get("use_lstm_sac"):
            # DogFightEnv SAC actor-LSTM contract:
            # Use EpisodeReplayBuffer by default because it is the stable path.
            # PrioritizedEpisodeReplayBuffer is opt-in and requires the
            # DogFightEnv Ray 2.54 prioritized sequence replay patch stored under
            # RLLibLstm; otherwise it samples 1-step fragments padded to T.
            if args.get("use_lstm_prioritized_replay"):
                replay_buffer_config["type"] = "PrioritizedEpisodeReplayBuffer"
                if args.get("debug_io") or args.get("debug_lstm_io"):
                    os.environ.setdefault("DOGFIGHT_PRIORITIZED_SEQ_DEBUG", "1")
                    os.environ.setdefault("DOGFIGHT_PRIORITIZED_SEQ_DEBUG_LIMIT", "20")
            else:
                replay_buffer_config["type"] = "EpisodeReplayBuffer"
            replay_buffer_config.setdefault(
                "batch_length_T",
                int(args.get("max_seq_len", 8)),
            )
        config = SACConfig()
        training_args = {
            "actor_lr": args.get("actor_lr", args["lr"]),
            "critic_lr": args.get("critic_lr", args["lr"]),
            "alpha_lr": args.get("alpha_lr", args["lr"]),
            "gamma": args["gamma"],
            "train_batch_size": args["train_batch_size"],
            "tau": args["tau"],
            "target_entropy": args["target_entropy"],
        }
        if replay_buffer_config:
            training_args["replay_buffer_config"] = replay_buffer_config
        config = config.training(**training_args)
        if args.get("use_lstm_sac"):
            config = _apply_lstm_sac_model_config(config, args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    if (
        not args.get("use_lstm")
        and not args.get("use_lstm_sac")
        and _has_mlp_model_config(args)
    ):
        config = _apply_mlp_model_config(config, args)

    config = config.environment(env_name, env_config=env_config).framework(args["framework"])

    if hasattr(config, "env_runners"):
        runner_args = {
            "num_env_runners": args["num_env_runners"],
            "num_envs_per_env_runner": args.get("num_envs_per_env_runner", 1),
        }
        if "rollout_fragment_length" in args:
            runner_args["rollout_fragment_length"] = _normalize_rollout_fragment_length(
                args["rollout_fragment_length"]
            )
        if "batch_mode" in args:
            runner_args["batch_mode"] = args["batch_mode"]
        config = config.env_runners(**runner_args)
    elif hasattr(config, "rollouts"):  # pragma: no cover
        rollout_args = {
            "num_rollout_workers": args["num_env_runners"],
            "num_envs_per_worker": args.get("num_envs_per_env_runner", 1),
        }
        if "rollout_fragment_length" in args:
            rollout_args["rollout_fragment_length"] = _normalize_rollout_fragment_length(
                args["rollout_fragment_length"]
            )
        if "batch_mode" in args:
            rollout_args["batch_mode"] = args["batch_mode"]
        config = config.rollouts(**rollout_args)

    from dogfight.ai.callbacks import DogFightCallbacks
    config = config.callbacks(DogFightCallbacks)

    return config


def _normalize_rollout_fragment_length(value):
    """Accept YAML/CLI string values while preserving RLlib's 'auto' option."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return "auto"
        return int(stripped)
    return value


def _build_replay_buffer_config(args: dict[str, Any]) -> dict[str, Any]:
    """Build an SAC replay buffer config from supported aliases."""
    replay_buffer_config = dict(args.get("replay_buffer_config") or {})
    capacity = args.get("replay_buffer_capacity", args.get("replay_buffer_size"))
    if capacity is not None:
        replay_buffer_config["capacity"] = int(capacity)
    return replay_buffer_config


def _has_mlp_model_config(args: dict[str, Any]) -> bool:
    """Return whether caller explicitly requested a custom MLP config."""
    model_config = args.get("model_config") or {}
    return bool(model_config.get("enabled", False))


def _parse_int_list(value: Any, default: list[int]) -> list[int]:
    """Parse YAML/CLI list syntax into a list of hidden-layer sizes."""
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    text = str(value).strip()
    if not text or text == "[]":
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_optional_bool(value: Any, default: bool) -> bool:
    """Parse common YAML/CLI boolean spellings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_network_spec(value: Any) -> dict[str, Any] | None:
    """Parse and normalize the optional DogFight SAC layer sequence spec."""
    if value in (None, "", {}):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("network_spec must be a mapping or JSON object")

    spec_type = str(value.get("type", "")).strip()
    if spec_type != "sequence_v1":
        raise ValueError(
            "Only network.type='sequence_v1' is implemented on the Ray 2.54 "
            f"New API SAC path; got {spec_type!r}."
        )

    normalized = {"type": "sequence_v1"}
    for role in ("actor", "critic"):
        if role in value:
            normalized[role] = _parse_sequence_v1_stack(value[role], role)
    if "actor" not in normalized:
        raise ValueError("sequence_v1 network spec requires an actor section")
    return normalized


def _parse_sequence_v1_stack(section: Any, role: str) -> dict[str, Any]:
    """Convert a linear/activation/lstm sequence into RLlib mappable fields."""
    if not isinstance(section, dict):
        raise ValueError(f"network.{role} must be a mapping")
    if "pre_lstm_hiddens" in section:
        return {
            "pre_lstm_hiddens": _parse_int_list(
                section.get("pre_lstm_hiddens"),
                [],
            ),
            "pre_lstm_activation": str(
                section.get("pre_lstm_activation") or "relu"
            ),
            "lstm_cell_size": int(section.get("lstm_cell_size", 64)),
            "lstm_layers": int(section.get("lstm_layers", 1)),
            "post_lstm_hiddens": _parse_int_list(
                section.get("post_lstm_hiddens"),
                [],
            ),
            "post_lstm_activation": str(
                section.get("post_lstm_activation") or "relu"
            ),
            "input": section.get("input", "obs" if role == "actor" else "obs_action"),
            "zero_init_state": bool(section.get("zero_init_state", role == "critic")),
        }

    encoder_layers = section.get("encoder", [])
    head_layers = section.get("head", [])
    if not isinstance(encoder_layers, list):
        raise ValueError(f"network.{role}.encoder must be a list")
    if not isinstance(head_layers, list):
        raise ValueError(f"network.{role}.head must be a list when provided")

    pre_lstm_hiddens: list[int] = []
    post_lstm_hiddens: list[int] = []
    pre_activation = None
    post_activation = None
    lstm_hidden = None
    lstm_layers = 1
    after_lstm = False

    for layer in [*encoder_layers, *head_layers]:
        if not isinstance(layer, dict) or len(layer) != 1:
            raise ValueError(
                f"network.{role} layers must be single-key mappings, got {layer!r}"
            )
        kind, config = next(iter(layer.items()))
        kind = str(kind).strip().lower()
        if kind == "linear":
            out = config.get("out") if isinstance(config, dict) else config
            if isinstance(out, str):
                # Output aliases are consumed by SAC heads. They are documented in
                # YAML to make the contract visible, but not part of hidden layers.
                allowed = {"action_dist_inputs", "q_value"}
                if out not in allowed:
                    raise ValueError(
                        f"network.{role} linear output alias {out!r} is not supported"
                    )
                continue
            hidden = int(out)
            if hidden <= 0:
                raise ValueError(f"network.{role} linear out must be positive")
            if after_lstm:
                post_lstm_hiddens.append(hidden)
            else:
                pre_lstm_hiddens.append(hidden)
        elif kind == "activation":
            activation = (
                config.get("type") if isinstance(config, dict) else str(config)
            )
            activation = str(activation).strip().lower()
            if activation not in {"relu", "tanh", "silu", "swish", "linear"}:
                raise ValueError(
                    f"network.{role} activation {activation!r} is not supported"
                )
            if after_lstm:
                post_activation = post_activation or activation
            else:
                pre_activation = pre_activation or activation
        elif kind == "lstm":
            if lstm_hidden is not None:
                raise ValueError(f"network.{role} supports exactly one lstm layer block")
            if not isinstance(config, dict):
                raise ValueError(f"network.{role}.lstm must be a mapping")
            lstm_hidden = int(config.get("hidden", config.get("cell_size", 64)))
            lstm_layers = int(config.get("layers", config.get("num_layers", 1)))
            if lstm_hidden <= 0 or lstm_layers <= 0:
                raise ValueError(
                    f"network.{role}.lstm hidden/layers must be positive"
                )
            after_lstm = True
        else:
            raise ValueError(f"network.{role} layer kind {kind!r} is not supported")

    if lstm_hidden is None:
        raise ValueError(f"network.{role} sequence_v1 requires one lstm layer")
    if not pre_lstm_hiddens:
        raise ValueError(
            f"network.{role} requires at least one pre-LSTM linear layer for "
            "the current RLlib tokenizer mapping"
        )
    return {
        "pre_lstm_hiddens": pre_lstm_hiddens,
        "pre_lstm_activation": pre_activation or "relu",
        "lstm_cell_size": lstm_hidden,
        "lstm_layers": lstm_layers,
        "post_lstm_hiddens": post_lstm_hiddens,
        "post_lstm_activation": post_activation or pre_activation or "relu",
        "input": section.get("input", "obs" if role == "actor" else "obs_action"),
        "zero_init_state": bool(section.get("zero_init_state", role == "critic")),
    }


def _build_default_model_config(args: dict[str, Any], *, use_lstm: bool = False):
    """Build RLlib's DefaultModelConfig from optional DogFight MLP settings."""
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    model_config = args.get("model_config") or {}
    network_spec = _parse_network_spec(
        args.get("network_spec") or model_config.get("network_spec")
    )
    actor_stack = (network_spec or {}).get("actor", {})
    kwargs = {
        "fcnet_hiddens": _parse_int_list(
            actor_stack.get("pre_lstm_hiddens")
            if actor_stack
            else model_config.get("fcnet_hiddens"),
            [256, 256],
        ),
        "fcnet_activation": str(
            actor_stack.get("pre_lstm_activation")
            if actor_stack
            else model_config.get("fcnet_activation") or "relu"
        ),
        "head_fcnet_hiddens": _parse_int_list(
            actor_stack.get("post_lstm_hiddens")
            if actor_stack
            else model_config.get("head_fcnet_hiddens"),
            [],
        ),
        "head_fcnet_activation": str(
            actor_stack.get("post_lstm_activation")
            if actor_stack
            else model_config.get("head_fcnet_activation") or "relu"
        ),
        "vf_share_layers": _parse_optional_bool(
            model_config.get("vf_share_layers"),
            True,
        ),
    }
    if use_lstm:
        kwargs.update(
            {
                "use_lstm": True,
                "max_seq_len": int(args.get("max_seq_len", 8)),
                "lstm_cell_size": int(
                    actor_stack.get("lstm_cell_size")
                    if actor_stack
                    else args.get("lstm_cell_size", 64)
                ),
            }
        )
    return DefaultModelConfig(**kwargs)


def _apply_mlp_model_config(config, args: dict[str, Any]):
    """Apply a non-recurrent RLlib DefaultModelConfig MLP."""
    return config.rl_module(model_config=_build_default_model_config(args))


def _apply_lstm_model_config(config, args: dict[str, Any]):
    """Apply a generic RLlib DefaultModelConfig LSTM for non-SAC algorithms."""
    return config.rl_module(model_config=_build_default_model_config(args, use_lstm=True))


def _apply_lstm_sac_model_config(config, args: dict[str, Any]):
    """Enable the patched Ray 2.54 SAC actor/recurrent-Q path.

    The local RLlib patch makes SAC stateful by routing STATE_IN/STATE_OUT through
    the actor encoder. With lstm_scope=actor_critic, Q/twin/target Q additionally
    encode [obs, action] through an internal zero-initialized LSTM. Keeping this
    config in one place makes the train/replay contract visible: max_seq_len
    controls replay sequence length, and lstm_cell_size controls recurrent state.
    """
    if args.get("debug_io") or args.get("debug_lstm_io"):
        os.environ["DOGFIGHT_RNNSAC_DEBUG"] = "1"
        os.environ.setdefault("DOGFIGHT_RNNSAC_DEBUG_LIMIT", "20")

    lstm_scope = str(args.get("lstm_scope") or "actor_only")
    if lstm_scope not in {"actor_only", "actor_critic"}:
        raise ValueError(
            "lstm_scope must be 'actor_only' or 'actor_critic', "
            f"got {lstm_scope!r}"
        )
    model_config = dataclasses.asdict(
        _build_default_model_config(args, use_lstm=True)
    )
    network_spec = _parse_network_spec(
        args.get("network_spec") or (args.get("model_config") or {}).get("network_spec")
    )
    model_config["dogfight_lstm_scope"] = lstm_scope
    model_config["dogfight_q_lstm_zero_init"] = lstm_scope == "actor_critic"
    if network_spec:
        critic_stack = network_spec.get("critic")
        if critic_stack and critic_stack.get("input") != "obs_action":
            raise ValueError("sequence_v1 critic input must be 'obs_action'")
        if critic_stack and lstm_scope != "actor_critic":
            raise ValueError(
                "sequence_v1 critic recurrent stack requires "
                "lstm_scope='actor_critic'"
            )
        model_config["dogfight_network_spec"] = network_spec
        if network_spec["actor"].get("lstm_layers", 1) != 1:
            model_config["dogfight_actor_lstm_num_layers"] = int(
                network_spec["actor"]["lstm_layers"]
            )
        if critic_stack:
            model_config["dogfight_q_fcnet_hiddens"] = list(
                critic_stack["pre_lstm_hiddens"]
            )
            model_config["dogfight_q_fcnet_activation"] = critic_stack[
                "pre_lstm_activation"
            ]
            model_config["dogfight_q_head_fcnet_hiddens"] = list(
                critic_stack["post_lstm_hiddens"]
            )
            model_config["dogfight_q_head_fcnet_activation"] = critic_stack[
                "post_lstm_activation"
            ]
            model_config["dogfight_q_lstm_cell_size"] = int(
                critic_stack["lstm_cell_size"]
            )
            model_config["dogfight_q_lstm_num_layers"] = int(
                critic_stack["lstm_layers"]
            )
    return config.rl_module(model_config=model_config)


def _sanitize_inference_config(value):
    """Drop RLlib runtime objects that were stringified in JSON metadata."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in SERIALIZED_OBJECT_KEYS:
                continue
            if key == "_model_config" and isinstance(item, (str, dict)):
                # Lightweight bundle metadata is JSON, so Ray dataclass configs
                # such as DefaultModelConfig are stored as strings/dicts. Ray
                # 2.54 expects _model_config to be a real dataclass instance
                # when build_algo() calls AlgorithmConfig.model_config. Drop the
                # serialized value here and reconstruct LSTM config from bundle
                # metadata below when needed.
                continue
            sanitized[key] = _sanitize_inference_config(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_inference_config(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("<class ") or stripped.startswith("<function "):
            return None
    return value


def _apply_bundle_model_config(config, bundle_meta: dict[str, Any]) -> tuple[Any, bool]:
    """Rebuild saved DefaultModelConfig for lightweight bundle restore."""
    saved_model_config = bundle_meta.get("model_config")
    if not isinstance(saved_model_config, dict) or not saved_model_config:
        return config, False

    use_lstm = bool(saved_model_config.get("use_lstm", False))
    model_args = {
        "model_config": {
            "enabled": True,
            "fcnet_hiddens": saved_model_config.get("fcnet_hiddens"),
            "fcnet_activation": saved_model_config.get("fcnet_activation"),
            "head_fcnet_hiddens": saved_model_config.get("head_fcnet_hiddens"),
            "head_fcnet_activation": saved_model_config.get("head_fcnet_activation"),
            "vf_share_layers": saved_model_config.get("vf_share_layers"),
        },
        "max_seq_len": saved_model_config.get("max_seq_len", 8),
        "lstm_cell_size": saved_model_config.get("lstm_cell_size", 64),
        "network_spec": saved_model_config.get("dogfight_network_spec")
        or bundle_meta.get("network_spec"),
    }
    if use_lstm:
        model_args["lstm_scope"] = (
            saved_model_config.get("dogfight_lstm_scope")
            or bundle_meta.get("lstm_scope")
            or "actor_only"
        )
        model_config = dataclasses.asdict(
            _build_default_model_config(model_args, use_lstm=True)
        )
        model_config["dogfight_lstm_scope"] = model_args["lstm_scope"]
        model_config["dogfight_q_lstm_zero_init"] = (
            model_args["lstm_scope"] == "actor_critic"
        )
    else:
        model_config = _build_default_model_config(model_args, use_lstm=False)
    return (config.rl_module(model_config=model_config), True)


def build_algorithm_from_bundle(metadata: dict):
    try:
        import ray
        from ray.tune.registry import register_env
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ray[rllib] is required for loading lightweight RL bundles") from exc

    from dogfight.ai.inference_env import RLLibInferenceEnv

    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    pythonpath_entries = [str(root), str(src)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        local_mode=True,
        runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
    )

    algorithm_class = metadata.get("algorithm_class", "").strip()
    config_dict = _sanitize_inference_config(metadata.get("algorithm_config", {}))
    env_name = config_dict.get("env")
    if env_name:
        register_env(env_name, lambda env_config: RLLibInferenceEnv(env_config))

    config_dict["num_env_runners"] = 0   # new API key
    config_dict["num_workers"] = 0       # old API key (ignored or warned by new API)
    config_dict["callbacks"] = None      # inference does not use training callbacks
    config_dict["disable_env_checking"] = True

    if algorithm_class == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig

        config_class = PPOConfig
    elif algorithm_class == "SAC":
        from ray.rllib.algorithms.sac import SACConfig

        config_class = SACConfig
    else:
        raise ValueError(f"Unsupported lightweight bundle algorithm: {algorithm_class!r}")

    if hasattr(config_class, "from_dict"):
        config = config_class.from_dict(config_dict)
    else:  # pragma: no cover
        config = config_class()
        if hasattr(config, "update_from_dict"):
            config = config.update_from_dict(config_dict)
        else:
            for key, value in config_dict.items():
                setattr(config, key, value)

    bundle_meta = metadata.get("metadata", {})
    config, restored_model_config = _apply_bundle_model_config(config, bundle_meta)
    if (
        not restored_model_config
        and algorithm_class == "SAC"
        and bundle_meta.get("use_lstm_sac")
    ):
        lstm_cell_size = int(bundle_meta.get("lstm_cell_size", 64))
        max_seq_len = int(bundle_meta.get("max_seq_len", 8))
        config = _apply_lstm_sac_model_config(
            config,
            {
                "use_lstm_sac": True,
                "lstm_cell_size": lstm_cell_size,
                "max_seq_len": max_seq_len,
                "lstm_scope": bundle_meta.get("lstm_scope", "actor_only"),
                "network_spec": bundle_meta.get("network_spec"),
                "debug_io": os.environ.get("DOGFIGHT_RNNSAC_DEBUG") == "1",
            },
        )
        if os.environ.get("DOGFIGHT_RNNSAC_DEBUG") == "1":
            print(
                "[DogFightEnv][RLlibConfig][LSTM_RESTORE] "
                f"use_lstm_sac=True max_seq_len={max_seq_len} "
                f"lstm_cell_size={lstm_cell_size}"
            )
    elif restored_model_config and os.environ.get("DOGFIGHT_RNNSAC_DEBUG") == "1":
        model_config = bundle_meta.get("model_config", {})
        print(
            "[DogFightEnv][RLlibConfig][MODEL_RESTORE] "
            f"fcnet_hiddens={model_config.get('fcnet_hiddens')} "
            f"use_lstm={model_config.get('use_lstm')}"
        )

    if hasattr(config, "env_runners"):
        config = config.env_runners(num_env_runners=0)
    elif hasattr(config, "rollouts"):  # pragma: no cover
        config = config.rollouts(num_rollout_workers=0)

    return config.build_algo()
