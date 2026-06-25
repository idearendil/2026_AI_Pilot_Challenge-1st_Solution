# -*- coding: utf-8 -*-
"""Minimal student reward template.

Edit this file when you want to change what the agent learns.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) must return (total_reward: float, components: dict).
  - Each item in components is recorded as ep_reward_<name> by the callbacks.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    "step_penalty": -0.01,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -10.0,
}


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Return a small runnable reward example.

    The arguments expose aircraft state, damage, geometry, WEZ settings, and
    termination status. Add your own tactical components here.
    """
    components: dict[str, float] = {
        "step": float(reward_config.get("step_penalty", -0.01)),
    }

    # TODO: Add your own shaping terms.
    # Useful inputs:
    #   geo_info._get_distance(ownship_state, target_state)
    #   geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    #   geo_info._get_aspect_angle(ownship_state, target_state)
    #   ownship_damage, target_damage, wez_config

    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if target_health <= 0.0 < ownship_health:
            terminal_reward = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(reward_config.get("loss_reward", -100.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", -10.0))
    components["terminal"] = terminal_reward

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
