"""Dogfight package with optional training dependencies loaded lazily."""

from __future__ import annotations

from typing import Any


__all__ = ["DogFightEnv"]


def __getattr__(name: str) -> Any:
    if name == "DogFightEnv":
        from .envs.single_agent_env import DogFightEnv

        return DogFightEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
