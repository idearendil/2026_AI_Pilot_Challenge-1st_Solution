from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from .config import MPCConfig
from .native import (
    NativeCostWeights,
    NativePredictor,
    PublicState,
    TargetSample,
)


ACTION_LOW = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float64)
ACTION_HIGH = np.ones(4, dtype=np.float64)


@dataclass
class PlanResult:
    action: np.ndarray
    score: float
    elapsed_ms: float
    candidate_count: int
    valid_fraction: float
    predicted_damage_dealt: float
    predicted_damage_taken: float
    min_altitude_ft: float
    min_range_ft: float
    final_ata_deg: float
    fallback: bool = False
    error: str = ""


class MPCPlanner:
    def __init__(self, root: str | Path, config: MPCConfig):
        self.root = Path(root).resolve()
        self.config = config
        self._rng_seed = int(config.cem.seed)
        self.rng = np.random.default_rng(self._rng_seed)
        self.native = NativePredictor(
            self.root / config.native_dll,
            self.root / config.asset_root,
            config.max_candidates,
        )
        self.weights = NativeCostWeights(*vars(config.weights).values())
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        profiles = [
            int(x) for x in config.cem.adaptive_candidate_counts
            if 1 <= int(x) <= config.cem.candidates
        ]
        if config.cem.candidates not in profiles:
            profiles.insert(0, config.cem.candidates)
        self._profiles = tuple(sorted(set(profiles), reverse=True))
        self._profile_index = 0
        self._last_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def reset(self) -> None:
        # A match must not depend on how many plans were evaluated in an
        # earlier match in the same process.
        self.rng = np.random.default_rng(self._rng_seed)
        self._mean = None
        self._std = None
        self._profile_index = 0
        self._last_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def close(self) -> None:
        self.native.close()

    def _shift_warm_start(self) -> tuple[np.ndarray, np.ndarray]:
        knot_count = self.config.knot_count
        if self._mean is None or self._std is None:
            mean = np.tile(self._last_action, (knot_count, 1))
            std = np.tile(np.asarray(self.config.cem.initial_std), (knot_count, 1))
            return mean, std
        shift = (1.0 / self.config.policy_hz) / self.config.knot_seconds
        source = np.arange(knot_count, dtype=np.float64)
        query = np.minimum(source + shift, knot_count - 1.0)
        mean = np.column_stack([
            np.interp(query, source, self._mean[:, axis]) for axis in range(4)
        ])
        std = np.column_stack([
            np.interp(query, source, self._std[:, axis]) for axis in range(4)
        ])
        return mean, std

    @staticmethod
    def _clip(actions: np.ndarray) -> np.ndarray:
        return np.clip(actions, ACTION_LOW, ACTION_HIGH)

    def _structured_candidates(self, mean: np.ndarray) -> list[np.ndarray]:
        knots = self.config.knot_count

        def constant(action) -> np.ndarray:
            return np.tile(np.asarray(action, dtype=np.float64), (knots, 1))

        def bank_then_pull(
            sign: float, entry_pitch: float, sustained_pitch: float
        ) -> np.ndarray:
            sequence = constant([0.0, sustained_pitch, 0.0, 1.0])
            # One 0.5 s roll knot is enough to establish a steep bank. Two
            # full-roll knots can rotate through the desired turn plane.
            sequence[0] = [0.90 * sign, entry_pitch, 0.10 * sign, 1.0]
            return sequence

        result = [
            mean.copy(),
            constant(self._last_action),
            constant([0.0, -0.92, 0.0, 1.0]),
            constant([0.0, 0.30, 0.0, 0.70]),
        ]
        for sign in (1.0, -1.0):
            result.append(bank_then_pull(sign, 0.05, -0.95))
            result.append(bank_then_pull(sign, -0.62, -0.92))
            result.append(bank_then_pull(sign, -0.25, -0.58))
        return result

    def _sample_candidates(
        self, mean: np.ndarray, std: np.ndarray, candidate_count: int
    ) -> np.ndarray:
        """Sample balanced CEM candidates plus a compact maneuver library."""
        structured = self._structured_candidates(mean)
        random_count = candidate_count - len(structured)
        if random_count < 2:
            raise ValueError("candidate population is too small for balanced sampling")
        pair_count = random_count // 2
        base = self.rng.standard_normal((pair_count, self.config.knot_count, 4))
        noise = np.concatenate((base, -base), axis=0)
        if noise.shape[0] < random_count:
            noise = np.concatenate((noise, np.zeros((1, self.config.knot_count, 4))), axis=0)
        random_candidates = self._clip(mean[None, :, :] + noise[:random_count] * std[None, :, :])
        return np.concatenate((random_candidates, np.asarray(structured)), axis=0)

    def _adapt_profile(self, elapsed_ms: float) -> None:
        budget = self.config.compute_budget_ms
        if elapsed_ms > budget and self._profile_index < len(self._profiles) - 1:
            self._profile_index += 1
        elif elapsed_ms < 0.45 * budget and self._profile_index > 0:
            self._profile_index -= 1

    def plan(self, state: PublicState, target_trajectory: list[TargetSample]) -> PlanResult:
        start = time.perf_counter()
        mean, std = self._shift_warm_start()
        minimum_std = np.asarray(self.config.cem.minimum_std, dtype=np.float64)
        maximum_std = np.asarray(self.config.cem.maximum_std, dtype=np.float64)
        candidate_count = self._profiles[self._profile_index]
        elite_count = max(2, int(math.ceil(candidate_count * self.config.cem.elite_fraction)))
        best_controls = mean.copy()
        best_diag = None
        valid_fraction = 0.0
        try:
            for _ in range(self.config.cem.iterations):
                candidates = self._sample_candidates(mean, std, candidate_count)
                diagnostics = self.native.evaluate_batch(
                    state,
                    target_trajectory,
                    candidates,
                    self.config.steps_per_knot,
                    self.weights,
                )
                scores = np.array([entry.score for entry in diagnostics], dtype=np.float64)
                valid = np.array([bool(entry.valid) for entry in diagnostics])
                valid_fraction = float(np.mean(valid))
                elite_indices = np.argpartition(scores, -elite_count)[-elite_count:]
                elites = candidates[elite_indices]
                new_mean = np.mean(elites, axis=0)
                new_std = np.std(elites, axis=0)
                mean = self.config.cem.mean_smoothing * mean + (1.0 - self.config.cem.mean_smoothing) * new_mean
                std = self.config.cem.std_smoothing * std + (1.0 - self.config.cem.std_smoothing) * new_std
                std = np.clip(std, minimum_std, maximum_std)
                best_index = int(np.argmax(scores))
                if best_diag is None or scores[best_index] > best_diag.score:
                    best_diag = diagnostics[best_index]
                    best_controls = candidates[best_index].copy()
            self._mean = mean
            self._std = std
            action = self._clip(best_controls[0]).astype(np.float32)
            if not np.all(np.isfinite(action)) or best_diag is None:
                raise FloatingPointError("CEM produced a non-finite action")
            self._last_action = action.astype(np.float64)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._adapt_profile(elapsed_ms)
            return PlanResult(
                action=action,
                score=float(best_diag.score),
                elapsed_ms=elapsed_ms,
                candidate_count=candidate_count,
                valid_fraction=valid_fraction,
                predicted_damage_dealt=float(best_diag.predicted_damage_dealt),
                predicted_damage_taken=float(best_diag.predicted_damage_taken),
                min_altitude_ft=float(best_diag.min_altitude_ft),
                min_range_ft=float(best_diag.min_range_ft),
                final_ata_deg=float(best_diag.final_ata_deg),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._adapt_profile(elapsed_ms)
            return PlanResult(
                action=self._last_action.astype(np.float32, copy=True),
                score=-1.0e12,
                elapsed_ms=elapsed_ms,
                candidate_count=candidate_count,
                valid_fraction=valid_fraction,
                predicted_damage_dealt=0.0,
                predicted_damage_taken=0.0,
                min_altitude_ft=float("nan"),
                min_range_ft=float("nan"),
                final_ata_deg=float("nan"),
                fallback=True,
                error=str(exc),
            )
