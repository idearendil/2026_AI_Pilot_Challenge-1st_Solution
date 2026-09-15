"""AI adapters with optional RL/BT dependencies loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AIPilot",
    "ActionContext",
    "ActionProvider",
    "ActionResult",
    "BTActionProvider",
    "DashboardJsonlLogger",
    "activate_rule_xml",
    "HybridActionProvider",
    "RLActionProvider",
    "build_algorithm_config",
    "build_algorithm_from_bundle",
    "load_lightweight_policy_bundle",
    "normalize_algorithm_name",
    "save_lightweight_policy_bundle",
    "save_training_record",
    "copy_experiment_yaml",
    "load_experiment_metadata",
    "training_row_to_dashboard_metrics",
]


_EXPORTS = {
    "AIPilot": ("native_bt", "AIPilot"),
    "ActionContext": ("action_provider", "ActionContext"),
    "ActionProvider": ("action_provider", "ActionProvider"),
    "ActionResult": ("action_provider", "ActionResult"),
    "BTActionProvider": ("bt_action_provider", "BTActionProvider"),
    "DashboardJsonlLogger": ("dashboard_logger", "DashboardJsonlLogger"),
    "activate_rule_xml": ("bt_rule_manager", "activate_rule_xml"),
    "HybridActionProvider": ("hybrid_action_provider", "HybridActionProvider"),
    "RLActionProvider": ("rl_action_provider", "RLActionProvider"),
    "build_algorithm_config": ("rllib_utils", "build_algorithm_config"),
    "build_algorithm_from_bundle": ("rllib_utils", "build_algorithm_from_bundle"),
    "load_lightweight_policy_bundle": ("checkpoint_io", "load_lightweight_policy_bundle"),
    "normalize_algorithm_name": ("rllib_utils", "normalize_algorithm_name"),
    "save_lightweight_policy_bundle": ("checkpoint_io", "save_lightweight_policy_bundle"),
    "save_training_record": ("training_record", "save_training_record"),
    "copy_experiment_yaml": ("dashboard_logger", "copy_experiment_yaml"),
    "load_experiment_metadata": ("dashboard_logger", "load_experiment_metadata"),
    "training_row_to_dashboard_metrics": ("dashboard_logger", "training_row_to_dashboard_metrics"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
