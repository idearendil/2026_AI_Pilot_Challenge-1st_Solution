# -*- coding: utf-8 -*-
"""Minimal custom observation example.

This module is loaded with:

  python train_rllib.py --observation-mode custom --observation-module student.my_observation

Keep OBSERVATION_SIZE synchronized with the vector returned by build_observation().
"""
from __future__ import annotations

import numpy as np

from dogfight.envs.observation import normalize
from dogfight.sim.state_schema import StateIndex


OBSERVATION_MODE = "student8"
OBSERVATION_SIZE = 8
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    """Return a custom 8-D observation vector as float32."""
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)

    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]), -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]), -90.0, 90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]), 0.0, 360.0)
    obs[3] = normalize(float(ownship_state[StateIndex.KCAS]), 0.0, 600.0)
    obs[4] = normalize(float(ownship_state[StateIndex.ALT]), 0.0, 15000.0)
    obs[5] = normalize(float(distance), 0.0, 20000.0)
    obs[6] = normalize(float(ata), -180.0, 180.0)
    obs[7] = normalize(float(aa), -180.0, 180.0)
    return obs


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "ownship_roll_norm",
            "ownship_pitch_norm",
            "ownship_yaw_norm",
            "ownship_kcas_norm",
            "ownship_alt_norm",
            "distance_norm",
            "ata_norm",
            "aa_norm",
        ],
        "description": "Student custom 8-D observation example.",
    }
