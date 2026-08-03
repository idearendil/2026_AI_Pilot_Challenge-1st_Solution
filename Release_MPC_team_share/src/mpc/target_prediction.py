from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import PredictionConfig
from .native import TargetSample
from .transforms import wrap_radians


@dataclass
class TargetObservation:
    time_s: float
    position_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray
    euler_deg: np.ndarray


class TargetMotionPredictor:
    """Causal decaying constant-turn/acceleration predictor.

    It uses only the current and previous public pose/velocity samples.  The
    decay prevents a short, noisy angular-rate estimate from being extrapolated
    unchanged over the whole MPC horizon.
    """

    def __init__(self, config: PredictionConfig, simulation_hz: int = 60):
        self.config = config
        self.simulation_hz = int(simulation_hz)
        self.previous: TargetObservation | None = None
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.turn_rate = 0.0

    def reset(self) -> None:
        self.previous = None
        self.acceleration.fill(0.0)
        self.turn_rate = 0.0

    def update(self, observation: TargetObservation) -> None:
        if self.previous is not None:
            dt = float(observation.time_s - self.previous.time_s)
            if 1.0e-5 < dt <= 1.0:
                raw_acceleration = (observation.velocity_ned_mps - self.previous.velocity_ned_mps) / dt
                acc_norm = float(np.linalg.norm(raw_acceleration))
                if acc_norm > self.config.max_acceleration_mps2:
                    raw_acceleration *= self.config.max_acceleration_mps2 / acc_norm
                horizontal = observation.velocity_ned_mps[:2]
                previous_horizontal = self.previous.velocity_ned_mps[:2]
                raw_turn_rate = 0.0
                if np.linalg.norm(horizontal) > 10.0 and np.linalg.norm(previous_horizontal) > 10.0:
                    heading = math.atan2(horizontal[1], horizontal[0])
                    previous_heading = math.atan2(previous_horizontal[1], previous_horizontal[0])
                    raw_turn_rate = np.clip(
                        wrap_radians(heading - previous_heading) / dt,
                        -self.config.max_turn_rate_radps,
                        self.config.max_turn_rate_radps,
                    )
                a = self.config.acceleration_smoothing
                t = self.config.turn_rate_smoothing
                self.acceleration = (1.0 - a) * self.acceleration + a * raw_acceleration
                self.turn_rate = float((1.0 - t) * self.turn_rate + t * raw_turn_rate)
        self.previous = observation

    def predict(self, step_count: int) -> list[TargetSample]:
        if self.previous is None:
            raise RuntimeError("target predictor needs one observation before predict()")
        dt = 1.0 / self.simulation_hz
        position = self.previous.position_ned_m.astype(np.float64, copy=True)
        velocity = self.previous.velocity_ned_mps.astype(np.float64, copy=True)
        euler = self.previous.euler_deg.astype(np.float64, copy=True)
        output = [TargetSample.from_values(position, velocity, euler)]
        for index in range(1, int(step_count) + 1):
            future_time = index * dt
            turn = self.turn_rate * math.exp(
                -future_time / max(self.config.turn_rate_time_constant_s, 1.0e-6)
            )
            angle = turn * dt
            c, s = math.cos(angle), math.sin(angle)
            vn, ve = velocity[0], velocity[1]
            velocity[0], velocity[1] = c * vn - s * ve, s * vn + c * ve
            acceleration = self.acceleration * math.exp(
                -future_time / max(self.config.acceleration_time_constant_s, 1.0e-6)
            )
            velocity += acceleration * dt
            speed = float(np.linalg.norm(velocity))
            if speed > self.config.max_speed_mps:
                velocity *= self.config.max_speed_mps / speed
            elif 1.0e-6 < speed < self.config.min_speed_mps:
                velocity *= self.config.min_speed_mps / speed
            position += velocity * dt
            horizontal_speed = float(np.linalg.norm(velocity[:2]))
            if horizontal_speed > 1.0e-6:
                euler[2] = math.degrees(math.atan2(velocity[1], velocity[0])) % 360.0
                euler[1] = math.degrees(math.atan2(-velocity[2], horizontal_speed))
            euler[0] *= math.exp(-dt / 1.5)
            output.append(TargetSample.from_values(position, velocity, euler))
        return output
