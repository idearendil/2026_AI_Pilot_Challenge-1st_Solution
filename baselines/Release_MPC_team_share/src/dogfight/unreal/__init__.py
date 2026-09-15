"""Competition protocol adapters loaded without optional policy dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AIType",
    "CMD",
    "ConstantCommandPolicy",
    "GameControl",
    "Init",
    "MessageType",
    "PlaneInfo",
    "ProviderCommandPolicy",
    "RLLightweightCommandPolicy",
    "SetPlaneID",
    "SimulationState",
    "UnrealAIPilotUDPClient",
]


_EXPORTS = {
    "UnrealAIPilotUDPClient": ("client", "UnrealAIPilotUDPClient"),
    "ConstantCommandPolicy": ("policies", "ConstantCommandPolicy"),
    "ProviderCommandPolicy": ("policies", "ProviderCommandPolicy"),
    "RLLightweightCommandPolicy": ("policies", "RLLightweightCommandPolicy"),
    "AIType": ("protocol", "AIType"),
    "CMD": ("protocol", "CMD"),
    "GameControl": ("protocol", "GameControl"),
    "Init": ("protocol", "Init"),
    "MessageType": ("protocol", "MessageType"),
    "PlaneInfo": ("protocol", "PlaneInfo"),
    "SetPlaneID": ("protocol", "SetPlaneID"),
    "SimulationState": ("protocol", "SimulationState"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
