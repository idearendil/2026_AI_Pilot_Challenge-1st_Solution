"""Native MPC with a post-search predictive ground-safety shield."""

from typing import TYPE_CHECKING, Any

from .config import SafeMPCConfig, load_safe_mpc_config
from .planner import SafeMPCPlanner, SafePlanResult

if TYPE_CHECKING:
    from .provider import SafeMPCActionProvider

__all__ = [
    "SafeMPCActionProvider",
    "SafeMPCConfig",
    "SafeMPCPlanner",
    "SafePlanResult",
    "load_safe_mpc_config",
]


def __getattr__(name: str) -> Any:
    if name == "SafeMPCActionProvider":
        from .provider import SafeMPCActionProvider

        globals()[name] = SafeMPCActionProvider
        return SafeMPCActionProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
