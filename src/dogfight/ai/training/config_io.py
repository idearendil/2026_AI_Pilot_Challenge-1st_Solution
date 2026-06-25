from __future__ import annotations

from pathlib import Path

import yaml


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge nested experiment config into base in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_experiment_env_config(path: str, root: Path) -> dict:
    """Load env_config from an experiment YAML path."""
    if not path:
        return {}

    exp_path = Path(path)
    if not exp_path.is_absolute():
        exp_path = (root / exp_path).resolve()
    data = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    env_config = data.get("env_config", {})
    if env_config is None:
        return {}
    if not isinstance(env_config, dict):
        raise ValueError("experiment YAML env_config must be a mapping")
    return env_config
