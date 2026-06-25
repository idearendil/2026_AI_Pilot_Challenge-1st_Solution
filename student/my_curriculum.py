# -*- coding: utf-8 -*-
"""Minimal student curriculum template.

Use this file only when you want to replace the built-in curriculum.
Run with:
  python train_curriculum.py --stages-module student.my_curriculum

For the full built-in curriculum and two-circle head-on reference, see:
  src/dogfight/ai/curriculum.py
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


def get_stages() -> list[CurriculumStage]:
    """Return a small runnable curriculum example."""
    return [
        CurriculumStage(
            index=0,
            name="student_fixed_target",
            description="Minimal fixed-target warmup stage.",
            target_mode="fixed",
            episode_step_limit=3600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={},
            randomization={
                "enabled": True,
                "radius": 500.0,
                "r_roll": 5.0,
                "r_pitch": 5.0,
                "r_heading": 30.0,
            },
            advance_conditions={},
            advance_window=5,
        ),
        CurriculumStage(
            index=1,
            name="student_bt_dogfight",
            description="Minimal behavior-tree opponent stage.",
            target_mode="behavior_tree",
            episode_step_limit=18000,
            max_iterations=20,
            checkpoint_interval=5,
            reward_overrides={},
            randomization={
                "enabled": True,
                "radius": 1500.0,
                "r_roll": 10.0,
                "r_pitch": 8.0,
                "r_heading": 120.0,
            },
            advance_conditions={},
            advance_window=5,
        ),
    ]


__all__ = ["get_stages"]
