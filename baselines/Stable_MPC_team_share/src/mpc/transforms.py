from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def body_to_ned_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return the standard aerospace body(x-forward,y-right,z-down)->NED DCM."""
    phi, theta, psi = np.radians([roll_deg, pitch_deg, yaw_deg])
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    return np.array(
        [
            [cth * cps, sphi * sth * cps - cphi * sps, cphi * sth * cps + sphi * sps],
            [cth * sps, sphi * sth * sps + cphi * cps, cphi * sth * sps - sphi * cps],
            [-sth, sphi * cth, cphi * cth],
        ],
        dtype=np.float64,
    )


def body_velocity_to_ned(body_velocity_mps, euler_deg) -> np.ndarray:
    return body_to_ned_matrix(*np.asarray(euler_deg, dtype=np.float64)) @ np.asarray(
        body_velocity_mps, dtype=np.float64
    )


def wrap_radians(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm with stable small-angle and near-pi handling."""
    r = np.asarray(rotation, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1.0e-7:
        return 0.5 * skew
    if math.pi - angle < 1.0e-5:
        diagonal = np.maximum((np.diag(r) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[1] = math.copysign(axis[1], r[0, 1] + r[1, 0])
        axis[2] = math.copysign(axis[2], r[0, 2] + r[2, 0])
        norm = float(np.linalg.norm(axis))
        return (axis / norm if norm > 1.0e-9 else np.array([1.0, 0.0, 0.0])) * angle
    return skew * (0.5 * angle / math.sin(angle))


@dataclass
class AngularRateEstimator:
    max_abs_radps: float = 6.0
    _previous_rotation: np.ndarray | None = None
    _previous_time_s: float | None = None

    def reset(self) -> None:
        self._previous_rotation = None
        self._previous_time_s = None

    def update(self, euler_deg, time_s: float) -> np.ndarray:
        current = body_to_ned_matrix(*np.asarray(euler_deg, dtype=np.float64))
        result = np.zeros(3, dtype=np.float64)
        if self._previous_rotation is not None and self._previous_time_s is not None:
            dt = float(time_s) - self._previous_time_s
            if 1.0e-5 < dt <= 1.0:
                delta_body = self._previous_rotation.T @ current
                result = rotation_log_vector(delta_body) / dt
                result = np.clip(result, -self.max_abs_radps, self.max_abs_radps)
        self._previous_rotation = current
        self._previous_time_s = float(time_s)
        return result
