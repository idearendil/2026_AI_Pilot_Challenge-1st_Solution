"""Unit tests for web log viewer parser and tactical math."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from log_data import (
    angle_between_deg,
    build_viewer_data,
    discover_log_pairs,
    forward_vector,
    in_wez,
    nearest_index,
    tactical_snapshot,
)


class LogDataTest(unittest.TestCase):
    def test_pair_discovery_and_replay_math(self) -> None:
        env_root = Path(__file__).resolve().parents[3] / "MyTrainEnv"
        logdir = env_root / "logs"
        pairs = discover_log_pairs(logdir)
        self.assertGreaterEqual(len(pairs), 1)

        ownship, target = pairs[0]
        data = build_viewer_data(ownship, target, None)
        self.assertEqual(nearest_index(data.ownship, 0.5), 29)
        self.assertGreater(nearest_index(data.ownship, 1.2), 0)
        snapshot = tactical_snapshot(data.ownship, data.target, 0.0)
        self.assertGreater(snapshot["distance_m"], 50.0)
        self.assertLess(abs(snapshot["relative_alt_m"]), 200.0)

    def test_vectors_and_wez_contract(self) -> None:
        forward = forward_vector(yaw_deg=90.0, pitch_deg=0.0)
        self.assertAlmostEqual(forward[0], 1.0, places=6)
        self.assertAlmostEqual(forward[1], 0.0, places=6)
        self.assertTrue(math.isclose(angle_between_deg((1, 0, 0), (0, 1, 0)), 90.0))
        self.assertTrue(in_wez(500.0, 0.5, 100.0, 1000.0, 2.0))
        self.assertFalse(in_wez(50.0, 0.5, 100.0, 1000.0, 2.0))
        self.assertFalse(in_wez(500.0, 2.0, 100.0, 1000.0, 2.0))


if __name__ == "__main__":
    unittest.main()
