"""Helpers for loading student-provided training hooks."""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from dogfight.ai.curriculum import CurriculumStage


def load_reward_hook(module_name: str) -> tuple[Callable[..., tuple[float, dict]], dict]:
    """Load a student reward module.

    The module must define:
      - MY_REWARD_CONFIG: dict
      - compute_reward(...): callable returning (float, dict)
    """
    name = module_name.strip()
    if not name:
        raise ValueError("reward module name is empty")

    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import reward module {name!r}: {exc}") from exc

    reward_config = getattr(module, "MY_REWARD_CONFIG", None)
    if not isinstance(reward_config, dict):
        raise RuntimeError(
            f"Reward module {name!r} must define MY_REWARD_CONFIG as a dict."
        )

    compute_reward = getattr(module, "compute_reward", None)
    if not callable(compute_reward):
        raise RuntimeError(
            f"Reward module {name!r} must define callable compute_reward(...)."
        )

    return compute_reward, dict(reward_config)


def load_curriculum_stages(module_name: str) -> list[CurriculumStage]:
    """Load custom CurriculumStage definitions from a student module."""
    name = module_name.strip()
    if not name:
        raise ValueError("curriculum stages module name is empty")

    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import stages module {name!r}: {exc}") from exc

    get_stages = getattr(module, "get_stages", None)
    if not callable(get_stages):
        raise RuntimeError(f"Stages module {name!r} must define get_stages().")

    stages = get_stages()
    if not isinstance(stages, list) or not stages:
        raise RuntimeError(f"Stages module {name!r} get_stages() must return a non-empty list.")
    if not all(isinstance(stage, CurriculumStage) for stage in stages):
        raise RuntimeError(
            f"Stages module {name!r} get_stages() must return CurriculumStage objects."
        )

    indexes = [stage.index for stage in stages]
    if len(indexes) != len(set(indexes)):
        raise RuntimeError(f"Stages module {name!r} has duplicate stage indexes.")

    return sorted(stages, key=lambda stage: stage.index)


def load_observation_hook(module_name: str) -> dict[str, Any]:
    """Load a student observation module.

    The module must define:
      - OBSERVATION_SIZE: int, or observation_size(): int
      - build_observation(...): callable returning a 1-D array-like
    Optional:
      - OBSERVATION_MODE: str
      - OBSERVATION_LOW / OBSERVATION_HIGH: scalar or array-like bounds
      - describe_observation(): dict
    """
    name = module_name.strip()
    if not name:
        raise ValueError("observation module name is empty")

    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import observation module {name!r}: {exc}") from exc

    build_observation = getattr(module, "build_observation", None)
    if not callable(build_observation):
        raise RuntimeError(
            f"Observation module {name!r} must define callable build_observation(...)."
        )

    if hasattr(module, "OBSERVATION_SIZE"):
        size = int(getattr(module, "OBSERVATION_SIZE"))
    else:
        observation_size = getattr(module, "observation_size", None)
        if not callable(observation_size):
            raise RuntimeError(
                f"Observation module {name!r} must define OBSERVATION_SIZE "
                "or callable observation_size()."
            )
        size = int(observation_size())
    if size <= 0:
        raise RuntimeError(f"Observation module {name!r} size must be positive.")

    describe = getattr(module, "describe_observation", None)
    if callable(describe):
        description = describe()
        if not isinstance(description, dict):
            raise RuntimeError(
                f"Observation module {name!r} describe_observation() must return a dict."
            )
    else:
        description = {
            "mode": getattr(module, "OBSERVATION_MODE", "custom"),
            "size": size,
            "features": [],
            "description": f"Custom observation from {name}.",
        }

    return {
        "module": name,
        "mode": str(getattr(module, "OBSERVATION_MODE", description.get("mode", "custom"))),
        "size": size,
        "low": getattr(module, "OBSERVATION_LOW", -1.0),
        "high": getattr(module, "OBSERVATION_HIGH", 1.0),
        "build_observation": build_observation,
        "description": description,
    }
