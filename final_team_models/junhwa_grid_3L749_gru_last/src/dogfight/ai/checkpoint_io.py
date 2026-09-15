from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_rl_module(algorithm, policy_id: str = "default_policy"):
    if not hasattr(algorithm, "get_module"):
        return None
    try:
        module = algorithm.get_module(policy_id)
        if module is not None:
            return module
    except Exception:
        pass
    try:
        return algorithm.get_module()
    except Exception:
        return None


def _extract_policy_weights(algorithm, policy_id: str = "default_policy"):
    module = _get_rl_module(algorithm, policy_id)
    if module is not None and hasattr(module, "get_state"):
        return module.get_state()

    try:
        policy = algorithm.get_policy(policy_id)
        if policy is not None:
            return policy.get_weights()
    except Exception:
        pass

    all_weights = algorithm.get_weights()
    if isinstance(all_weights, dict):
        return all_weights.get(policy_id) or next(iter(all_weights.values()), {})
    return all_weights


def _apply_policy_weights(
    algorithm,
    weights,
    policy_id: str = "default_policy",
) -> bool:
    env_runner = getattr(algorithm, "env_runner", None)
    if env_runner is not None and hasattr(env_runner, "set_state"):
        env_runner.set_state({"rl_module": weights})
        return True
    if env_runner is not None and hasattr(env_runner, "module"):
        env_runner.module.set_state(weights)
        return True

    module = _get_rl_module(algorithm, policy_id)
    if module is not None and hasattr(module, "set_state"):
        module.set_state(weights)
        return True

    try:
        policy = algorithm.get_policy(policy_id)
        if policy is not None:
            policy.set_weights(weights)
            return True
    except Exception:
        pass

    for candidate in ({policy_id: weights}, weights):
        try:
            algorithm.set_weights(candidate)
            return True
        except Exception:
            pass
    return False


def save_lightweight_policy_bundle(
    algorithm,
    output_dir,
    policy_id: str = "default_policy",
    metadata: dict | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights = _extract_policy_weights(algorithm, policy_id)
    config = algorithm.config.to_dict() if hasattr(algorithm.config, "to_dict") else dict(algorithm.config)
    metadata_payload = dict(metadata or {})
    model_config = getattr(algorithm.config, "model_config", None)
    if isinstance(model_config, dict) and model_config:
        metadata_payload.setdefault("model_config", _json_safe(model_config))

    payload = {
        "algorithm_class": algorithm.__class__.__name__,
        "policy_id": policy_id,
        "algorithm_config": _json_safe(config),
        "metadata": _json_safe(metadata_payload),
    }

    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    weights_path = output_path / "policy_weights.pkl.gz"
    with gzip.open(weights_path, "wb") as file:
        pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def load_lightweight_policy_bundle(bundle_dir):
    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / "metadata.json"
    weights_path = bundle_path / "policy_weights.pkl.gz"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with gzip.open(weights_path, "rb") as file:
        weights = pickle.load(file)

    return metadata, weights


def apply_lightweight_policy_bundle(
    algorithm,
    bundle_dir,
    policy_id: str = "default_policy",
) -> dict:
    """Load a lightweight bundle into an RLlib algorithm and verify it."""
    metadata, weights = load_lightweight_policy_bundle(bundle_dir)

    if not _apply_policy_weights(algorithm, weights, policy_id):
        raise RuntimeError(
            "Neither RLModule state loading nor old Policy weight loading worked."
        )

    try:
        module = _get_rl_module(algorithm, policy_id)
        back = module.get_state() if module is not None else algorithm.get_weights()
        if back is None:
            raise RuntimeError("Weight verification returned None after set.")
    except Exception as exc:
        raise RuntimeError(f"Weight verification failed: {exc}") from exc

    return metadata
