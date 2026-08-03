from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc.config import load_config  # noqa: E402
from mpc.native import (  # noqa: E402
    NativeCostWeights,
    NativePredictor,
    PublicState,
    TargetSample,
)
from mpc.planner import MPCPlanner  # noqa: E402
from mpc.transforms import AngularRateEstimator, body_to_ned_matrix, body_velocity_to_ned  # noqa: E402


def test_body_frame_axes_and_velocity() -> None:
    np.testing.assert_allclose(body_to_ned_matrix(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        body_velocity_to_ned([100.0, 0.0, 0.0], [0.0, 0.0, 90.0]),
        [0.0, 100.0, 0.0], atol=1e-10,
    )


def test_so3_rate_wrap_is_small() -> None:
    estimator = AngularRateEstimator()
    estimator.update([0.0, 0.0, 179.0], 0.0)
    rate = estimator.update([0.0, 0.0, -179.0], 0.1)
    assert 0.0 < rate[2] < 1.0


def test_config_timing_contract() -> None:
    config = load_config(ROOT / "configs" / "mpc.yaml")
    assert config.action_repeat == 6
    assert config.steps_per_knot == 30
    assert config.knot_count == 4
    assert config.cem.candidates == 48
    assert config.cem.adaptive_candidate_counts == (48,)
    assert config.weights.damage_dealt == 140.0
    assert config.weights.damage_taken == 180.0
    assert config.weights.control_zone == 9.0
    assert config.weights.nose_advantage == 9.0
    assert config.weights.threat_geometry == 12.0
    assert config.weights.terminal_geometry == 28.0
    assert config.weights.nose_advantage > 0.0
    assert config.weights.overshoot > 0.0


def test_default_config_matches_deployed_yaml() -> None:
    assert load_config() == load_config(ROOT / "configs" / "mpc.yaml")


def test_attack_score_strictly_prefers_zero_ata() -> None:
    config = load_config(ROOT / "configs" / "mpc.yaml")
    predictor = NativePredictor(
        ROOT / "runtime" / "predictor" / "Release" / "MPCJSBSim.dll", ROOT, 1
    )
    state = PublicState(
        0.0, 0.0, -5000.0, 0.0, 0.0, 0.0,
        250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    weights = NativeCostWeights(*vars(config.weights).values())
    controls = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float64)

    def score(angle_deg: float) -> float:
        angle = np.radians(angle_deg)
        position = np.array([np.cos(angle), np.sin(angle), 0.0]) * 1200.0
        sample = TargetSample(
            position[0], position[1], -5000.0,
            250.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return predictor.evaluate_batch(
            state, [sample, sample], controls, 1, weights
        )[0].score

    try:
        score_0 = score(0.0)
        score_10 = score(10.0)
        score_30 = score(30.0)
        score_180 = score(180.0)
    finally:
        predictor.close()
    assert score_0 > score_10 > score_30
    assert score_30 > score_180


def test_nose_advantage_is_relative_and_not_opponent_specific() -> None:
    """The cost prefers a generic nose advantage at equal non-WEZ range."""
    config = load_config(ROOT / "configs" / "mpc.yaml")
    predictor = NativePredictor(
        ROOT / "runtime" / "predictor" / "Release" / "MPCJSBSim.dll", ROOT, 1
    )
    state = PublicState(
        0.0, 0.0, -5000.0, 0.0, 0.0, 0.0,
        250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    weights = NativeCostWeights(*vars(config.weights).values())
    controls = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float64)

    def score(target_yaw: float) -> float:
        sample = TargetSample(
            5000.0, 0.0, -5000.0,
            0.0, 0.0, 0.0, 0.0, 0.0, target_yaw,
        )
        return predictor.evaluate_batch(
            state, [sample, sample], controls, 1, weights
        )[0].score

    try:
        own_nose = score(0.0)       # target points away from ownship
        enemy_nose = score(180.0)   # target points toward ownship
    finally:
        predictor.close()
    assert own_nose > enemy_nose


def test_cem_sampling_is_antithetic_and_reset_reproducible() -> None:
    config = load_config(ROOT / "configs" / "mpc.yaml")
    planner = object.__new__(MPCPlanner)
    planner.config = config
    planner._rng_seed = int(config.cem.seed)
    planner.rng = np.random.default_rng(planner._rng_seed)
    planner._last_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    planner._mean = planner._std = None
    planner._profile_index = 0
    mean = np.zeros((config.knot_count, 4), dtype=np.float64)
    mean[:, 3] = 0.5
    std = np.full_like(mean, 0.1)
    first = planner._sample_candidates(mean, std, config.cem.candidates)
    structured_count = len(planner._structured_candidates(mean))
    random_count = config.cem.candidates - structured_count
    half = random_count // 2
    np.testing.assert_allclose(
        first[:half] - mean,
        -(first[half:random_count] - mean),
        atol=1.0e-12,
    )
    planner.reset()
    second = planner._sample_candidates(mean, std, config.cem.candidates)
    np.testing.assert_array_equal(first, second)


def test_bfm_candidates_bank_then_pull_and_are_mirrored() -> None:
    config = load_config(ROOT / "configs" / "mpc.yaml")
    planner = object.__new__(MPCPlanner)
    planner.config = config
    planner._last_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    mean = np.zeros((config.knot_count, 4), dtype=np.float64)
    candidates = planner._structured_candidates(mean)
    assert len(candidates) == 10
    right = candidates[4]
    left = candidates[7]
    assert right[0, 0] > 0.0 and np.allclose(right[1:, 0], 0.0)
    assert right[0, 1] > right[1, 1]
    np.testing.assert_allclose(left[:, 0], -right[:, 0])
    np.testing.assert_allclose(left[:, 1], right[:, 1])
    np.testing.assert_allclose(left[:, 2], -right[:, 2])
    np.testing.assert_allclose(left[:, 3], right[:, 3])



def test_native_deterministic_replay() -> None:
    predictor = NativePredictor(
        ROOT / "runtime" / "predictor" / "Release" / "MPCJSBSim.dll", ROOT, 1
    )
    state = PublicState(
        3000.0, 0.0, -5000.0, 0.0, 0.0, 90.0,
        250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    controls = np.array([[0.2, -0.3, 0.05, 1.0], [-0.1, -0.5, 0.0, 0.8]])
    try:
        first = predictor.rollout_one(state, controls, 12)
        second = predictor.rollout_one(state, controls, 12)
    finally:
        predictor.close()
    first_values = np.array([getattr(first, name) for name, _ in first._fields_])
    second_values = np.array([getattr(second, name) for name, _ in second._fields_])
    # The independent reduced 6DoF predictor is repeatable to roundoff.
    np.testing.assert_allclose(first_values, second_values, rtol=1.0e-12, atol=2.0e-9)
    assert np.all(np.isfinite(first_values))
    assert first.sim_time_s > state.sim_time_s


def test_xml_fcs_rates_and_one_frame_dynamics_latency() -> None:
    predictor = NativePredictor(
        ROOT / "runtime" / "predictor" / "Release" / "MPCJSBSim.dll", ROOT, 1
    )
    state = PublicState(
        3000.0, 0.0, -5000.0, 0.0, 0.0, 90.0,
        250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    try:
        roll = predictor.rollout_debug(state, [[1.0, 0.0, 0.0, 1.0]], 1)
        rudder = predictor.rollout_debug(state, [[0.0, 0.0, 1.0, 1.0]], 1)
        idle = predictor.rollout_debug(state, [[0.0, 0.0, 0.0, 0.0]], 60)
    finally:
        predictor.close()
    # At Mach ~0.75, the XML's 0.3 s aileron traverse and Mach compensation
    # produce about 0.8 degree on the first frame.
    assert 0.7 < roll.aileron_deg < 0.9
    # The public rate still reflects the previous derivative on that frame.
    assert abs(np.degrees(roll.public_state.p_radps)) < 0.1
    # The shared yaw PID/kinematic output property creates a roughly 5.75 deg
    # first-frame rudder response in the supplied XML FCS.
    assert 5.0 < rudder.rudder_deg < 6.5
    # Idle N2 is 53%; after one second from max it is close to the measured 55%.
    assert 53.0 <= idle.engine_n2_percent <= 58.0
