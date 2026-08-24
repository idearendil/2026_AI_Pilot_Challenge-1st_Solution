from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from mpc.config import MPCConfig
from mpc.native import PublicState, TargetSample
from mpc.planner import MPCPlanner
from mpc.transforms import body_velocity_to_ned

from .config import SafeMPCConfig


@dataclass
class SafePlanResult:
    action: np.ndarray
    score: float
    elapsed_ms: float
    candidate_count: int
    valid_fraction: float
    safe_fraction: float
    predicted_damage_dealt: float
    predicted_damage_taken: float
    min_altitude_ft: float
    min_range_ft: float
    final_ata_deg: float
    native_best_safe: bool
    safety_reranked: bool
    selected_native_rank: int
    safety_check_mode: str = "full"
    safety_candidates_checked: int = 0
    safety_rollout_count: int = 0
    safety_policy: str = "predictive"
    current_altitude_m: float = float("nan")
    current_sink_mps: float = float("nan")
    current_pullout_margin_m: float = float("nan")
    native_min_altitude_m: float = float("nan")
    native_pullout_margin_m: float = float("nan")
    selected_pullout_margin_m: float = float("nan")
    native_score_loss: float = 0.0
    first_action_delta_l2: float = 0.0
    robust_scenario_count: int = 1
    robust_candidate_count: int = 0
    robust_selected: bool = False
    robust_native_loss: float = 0.0
    robust_regret_gain: float = 0.0
    candidate_selection: str = "all_samples"
    selection_changed: bool = False
    selection_native_score_loss: float = 0.0
    safety_override: bool = False
    fallback: bool = False
    error: str = ""


class SafeMPCPlanner(MPCPlanner):
    """Unchanged native CEM with a post-search predictive safety shield.

    Candidate generation, native scoring, elite selection, and warm-start
    updates are inherited from :class:`MPCPlanner`. Safety never changes the
    CEM distribution. It can only replace the final native winner with the
    highest-scoring recoverable candidate, or issue a deterministic recovery
    command when every otherwise-valid candidate is unsafe.
    """

    def __init__(
        self,
        root: str | Path,
        mpc_config: MPCConfig,
        safety_config: SafeMPCConfig,
    ):
        super().__init__(root, mpc_config)
        self.safety = safety_config

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _velocity_ned(state: PublicState) -> np.ndarray:
        return body_velocity_to_ned(
            [state.u_mps, state.v_mps, state.w_mps],
            [state.roll_deg, state.pitch_deg, state.yaw_deg],
        )

    def _pullout_margin_m(self, state: PublicState) -> float:
        altitude = -float(state.down_m)
        sink_speed = max(0.0, float(self._velocity_ned(state)[2]))
        roll_excess_rad = math.radians(
            max(
                0.0,
                abs(self._wrap_degrees(state.roll_deg))
                - self.safety.safety_level_roll_deg,
            )
        )
        roll_time = roll_excess_rad / self.safety.safety_roll_rate_radps
        roll_loss = sink_speed * roll_time + 0.5 * 9.80665 * roll_time**2
        pull_loss = sink_speed**2 / (
            2.0 * self.safety.safety_pullout_accel_mps2
        )
        return altitude - (
            self.safety.safety_clearance_m + roll_loss + pull_loss
        )

    def _candidate_is_safe(
        self,
        initial_state: PublicState,
        candidate: np.ndarray,
        diagnostic,
    ) -> bool:
        if not bool(diagnostic.valid) or not math.isfinite(float(diagnostic.score)):
            return False
        minimum_altitude_m = float(diagnostic.min_altitude_ft) * 0.3048
        if minimum_altitude_m < self.safety.safety_clearance_m:
            return False
        terminal = self.native.rollout_one(
            initial_state,
            candidate,
            self.config.steps_per_knot,
        )
        return self._pullout_margin_m(terminal) >= 0.0

    def _candidate_pullout_margin_m(
        self,
        initial_state: PublicState,
        candidate: np.ndarray,
        diagnostic,
    ) -> float:
        if not bool(diagnostic.valid) or not math.isfinite(float(diagnostic.score)):
            return float("-inf")
        minimum_altitude_m = float(diagnostic.min_altitude_ft) * 0.3048
        if minimum_altitude_m < self.safety.safety_clearance_m:
            return float("-inf")
        terminal = self.native.rollout_one(
            initial_state,
            candidate,
            self.config.steps_per_knot,
        )
        return self._pullout_margin_m(terminal)

    def _recovery_action(self, state: PublicState) -> tuple[np.ndarray, str]:
        roll = self._wrap_degrees(state.roll_deg)
        if abs(roll) > self.safety.safety_level_roll_deg:
            return (
                np.asarray([-np.sign(roll), 0.28, 0.0, 1.0], dtype=np.float32),
                "predictive_recovery_level",
            )
        return (
            np.asarray([0.0, -0.98, 0.0, 1.0], dtype=np.float32),
            "predictive_recovery_pull",
        )

    def _robust_select(
        self,
        state: PublicState,
        response_trajectories: list[list[TargetSample]],
        pool_controls: list[np.ndarray],
        pool_diagnostics: list,
        pool_scores: list[float],
    ) -> tuple[np.ndarray, object, float, int, int, int, bool, float, float]:
        """Rerank native-close safe candidates by normalized scenario regret."""

        # Fixed maneuver-library entries can appear unchanged in both CEM
        # iterations.  They must not consume multiple top-K slots in this
        # post-search comparison.  Deduplication happens only here, after all
        # native elite and warm-start updates are complete.
        unique_indices: dict[bytes, int] = {}
        for index, controls in enumerate(pool_controls):
            key = np.ascontiguousarray(controls, dtype=np.float64).tobytes()
            previous = unique_indices.get(key)
            if previous is None or pool_scores[index] > pool_scores[previous]:
                unique_indices[key] = index
        unique = list(unique_indices.values())
        pool_controls = [pool_controls[index] for index in unique]
        pool_diagnostics = [pool_diagnostics[index] for index in unique]
        pool_scores = [pool_scores[index] for index in unique]

        scores = np.asarray(pool_scores, dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        native_scale = max(1.0, float(np.std(scores[np.isfinite(scores)])))
        loss_limit = self.safety.robust_native_loss_std * native_scale
        native_best_score = float(scores[order[0]])
        eligible = [
            int(index)
            for index in order
            if scores[index] >= native_best_score - loss_limit
        ][: self.safety.robust_top_k]
        base_pool_index = eligible[0]
        if len(eligible) < 2 or not response_trajectories:
            return (
                pool_controls[base_pool_index],
                pool_diagnostics[base_pool_index],
                float(scores[base_pool_index]),
                1,
                len(eligible),
                1,
                False,
                0.0,
                0.0,
            )

        controls = np.asarray(
            [pool_controls[index] for index in eligible],
            dtype=np.float64,
        )
        scenario_scores = [scores[eligible].copy()]
        scenario_valid = [np.ones(len(eligible), dtype=bool)]
        for trajectory in response_trajectories:
            diagnostics = self.native.evaluate_batch(
                state,
                trajectory,
                controls,
                self.config.steps_per_knot,
                self.weights,
            )
            values = np.asarray(
                [float(item.score) for item in diagnostics],
                dtype=np.float64,
            )
            valid = np.asarray(
                [bool(item.valid) for item in diagnostics],
                dtype=bool,
            ) & np.isfinite(values)
            # If an entire response path fails to simulate, it contains no
            # comparative information and the conservative action is to keep
            # the native winner rather than manufacture a ranking.
            if not np.any(valid):
                return (
                    pool_controls[base_pool_index],
                    pool_diagnostics[base_pool_index],
                    float(scores[base_pool_index]),
                    1,
                    len(eligible),
                    1,
                    False,
                    0.0,
                    0.0,
                )
            scenario_scores.append(values)
            scenario_valid.append(valid)

        matrix = np.vstack(scenario_scores)
        valid_matrix = np.vstack(scenario_valid)
        objective = np.full(len(eligible), np.inf, dtype=np.float64)
        common_valid = np.all(valid_matrix, axis=0)
        if np.any(common_valid):
            regrets = np.zeros_like(matrix)
            for scenario in range(matrix.shape[0]):
                values = matrix[scenario, valid_matrix[scenario]]
                scale = max(1.0, float(np.std(values)))
                best = float(np.max(values))
                regrets[scenario] = (best - matrix[scenario]) / scale
            mean_regret = np.mean(regrets, axis=0)
            worst_regret = np.max(regrets, axis=0)
            risk = self.safety.robust_worst_case_weight
            objective[common_valid] = (
                (1.0 - risk) * mean_regret[common_valid]
                + risk * worst_regret[common_valid]
            )
        if not np.any(np.isfinite(objective)):
            selected = 0
        else:
            selected = int(np.argmin(objective))
        selected_pool_index = eligible[selected]
        base_objective = float(objective[0])
        selected_objective = float(objective[selected])
        if math.isfinite(base_objective) and math.isfinite(selected_objective):
            regret_gain = base_objective - selected_objective
        elif selected != 0 and math.isfinite(selected_objective):
            # The native winner failed at least one response rollout while the
            # alternative remained valid. Use a finite sentinel for telemetry.
            regret_gain = 1.0 + selected_objective
        else:
            regret_gain = 0.0
        robust_selected = selected != 0 and regret_gain > 1.0e-12
        if not robust_selected:
            selected = 0
            selected_pool_index = base_pool_index
            regret_gain = 0.0
        selected_score = float(scores[selected_pool_index])
        rank = 1 + int(np.sum(scores > selected_score))
        return (
            pool_controls[selected_pool_index],
            pool_diagnostics[selected_pool_index],
            selected_score,
            rank,
            len(eligible),
            matrix.shape[0],
            robust_selected,
            native_best_score - selected_score,
            regret_gain,
        )

    def plan(
        self,
        state: PublicState,
        target_trajectory: list[TargetSample],
        response_trajectories: list[list[TargetSample]] | None = None,
    ) -> SafePlanResult:
        started = time.perf_counter()
        mean, std = self._shift_warm_start()
        minimum_std = np.asarray(self.config.cem.minimum_std, dtype=np.float64)
        maximum_std = np.asarray(self.config.cem.maximum_std, dtype=np.float64)
        candidate_count = self._profiles[self._profile_index]
        elite_count = max(
            2,
            int(math.ceil(candidate_count * self.config.cem.elite_fraction)),
        )
        pool_controls: list[np.ndarray] = []
        pool_diagnostics: list = []
        pool_scores: list[float] = []
        pool_valid: list[bool] = []
        valid_fraction = 0.0
        safe_fraction = 0.0
        safety_candidates_checked = 0
        safety_rollout_count = 0
        last_iteration_start = 0
        last_elite_mean: np.ndarray | None = None
        try:
            for _ in range(self.config.cem.iterations):
                # This is exactly MPCPlanner's population generator. In
                # particular, no safety or auxiliary proposal consumes an RNG
                # draw or displaces one of the 48 native candidates.
                last_iteration_start = len(pool_controls)
                candidates = self._sample_candidates(mean, std, candidate_count)
                diagnostics = self.native.evaluate_batch(
                    state,
                    target_trajectory,
                    candidates,
                    self.config.steps_per_knot,
                    self.weights,
                )
                scores = np.asarray(
                    [float(entry.score) for entry in diagnostics],
                    dtype=np.float64,
                )
                valid = np.asarray(
                    [bool(entry.valid) for entry in diagnostics],
                    dtype=bool,
                ) & np.isfinite(scores)
                valid_fraction = float(np.mean(valid))

                # Keep the base MPC's CEM update bit-for-bit independent of
                # safety. The native implementation assigns unusable rollouts
                # a losing score, as assumed by MPCPlanner itself.
                elite_indices = np.argpartition(scores, -elite_count)[-elite_count:]
                elites = candidates[elite_indices]
                new_mean = np.mean(elites, axis=0)
                last_elite_mean = new_mean.copy()
                new_std = np.std(elites, axis=0)
                mean = (
                    self.config.cem.mean_smoothing * mean
                    + (1.0 - self.config.cem.mean_smoothing) * new_mean
                )
                std = (
                    self.config.cem.std_smoothing * std
                    + (1.0 - self.config.cem.std_smoothing) * new_std
                )
                std = np.clip(std, minimum_std, maximum_std)

                pool_controls.extend(candidate.copy() for candidate in candidates)
                pool_diagnostics.extend(diagnostics)
                pool_scores.extend(float(score) for score in scores)
                pool_valid.extend(bool(item) for item in valid)

            # Preserve the native-only distribution regardless of which
            # control is finally executed.
            self._mean = mean
            self._std = std

            if not pool_controls:
                raise RuntimeError("native CEM produced no candidates")
            native_scores = np.asarray(pool_scores, dtype=np.float64)
            native_valid = (
                np.asarray(pool_valid, dtype=bool) & np.isfinite(native_scores)
            )
            native_order = np.argsort(-native_scores, kind="stable")
            native_index = int(native_order[0])
            native_best_controls = pool_controls[native_index]
            native_best_diagnostic = pool_diagnostics[native_index]
            native_best_score = float(native_scores[native_index])

            # CEM convention and optimizer's-curse ablations.  These modes
            # change only the action chosen after the two native CEM updates;
            # the sampled candidates, elites, mean/std, and RNG stream above
            # remain identical to the production controller.
            derived_index: int | None = None
            selection_mode = self.safety.candidate_selection
            if selection_mode in {
                "smoothed_mean",
                "elite_mean",
                "winner_mean_blend",
            }:
                if selection_mode == "smoothed_mean":
                    derived_controls = mean.copy()
                elif selection_mode == "elite_mean":
                    if last_elite_mean is None:
                        raise RuntimeError("final CEM elite mean is unavailable")
                    derived_controls = last_elite_mean.copy()
                else:
                    blend = self.safety.selection_blend_fraction
                    derived_controls = (
                        (1.0 - blend) * native_best_controls + blend * mean
                    )
                derived_controls = self._clip(derived_controls)
                derived_diagnostic = self.native.evaluate_batch(
                    state,
                    target_trajectory,
                    derived_controls[None, :, :],
                    self.config.steps_per_knot,
                    self.weights,
                )[0]
                derived_index = len(pool_controls)
                pool_controls.append(derived_controls.copy())
                pool_diagnostics.append(derived_diagnostic)
                pool_scores.append(float(derived_diagnostic.score))
                pool_valid.append(
                    bool(derived_diagnostic.valid)
                    and math.isfinite(float(derived_diagnostic.score))
                )

            scores = np.asarray(pool_scores, dtype=np.float64)
            valid = np.asarray(pool_valid, dtype=bool) & np.isfinite(scores)
            ranking_scores = scores.copy()
            finite_scores = scores[valid & (scores > -1.0e11)]
            priority_bonus = (
                max(1.0, float(np.ptp(finite_scores)) + 1.0)
                if finite_scores.size
                else 1.0
            )
            if selection_mode == "last_iteration":
                preferred = np.arange(last_iteration_start, len(native_scores))
                ranking_scores[preferred[native_valid[preferred]]] += priority_bonus
            elif derived_index is not None and valid[derived_index]:
                ranking_scores[derived_index] += priority_bonus
            order = np.argsort(-ranking_scores, kind="stable")
            preferred_index = int(order[0])
            selection_changed = preferred_index != native_index
            selection_native_score_loss = (
                native_best_score - float(scores[preferred_index])
            )
            current_velocity_ned = self._velocity_ned(state)
            current_altitude_m = -float(state.down_m)
            current_sink_mps = max(0.0, float(current_velocity_ned[2]))
            current_pullout_margin_m = self._pullout_margin_m(state)

            # The diagnostic minimum altitude is a zero-cost necessary safety
            # condition.  Only candidates surviving it need a terminal native
            # rollout for the pull-out margin check.
            prefiltered = np.zeros(len(pool_controls), dtype=bool)
            for index in np.flatnonzero(valid):
                minimum_altitude_m = (
                    float(pool_diagnostics[index].min_altitude_ft) * 0.3048
                )
                prefiltered[index] = (
                    minimum_altitude_m >= self.safety.safety_clearance_m
                )

            safe = np.zeros(len(pool_controls), dtype=bool)
            terminal_margins = np.full(len(pool_controls), np.nan, dtype=np.float64)
            safety_selection_scores = ranking_scores.copy()
            full_check = (
                self.safety.safety_check_mode == "full"
                or self.safety.robust_enabled
            )
            emergency_deferred = (
                self.safety.safety_policy == "emergency"
                and not self.safety.robust_enabled
                and current_pullout_margin_m
                >= self.safety.emergency_trigger_margin_m
            )
            altitude_only = self.safety.safety_policy == "altitude"
            risk_priced = self.safety.safety_policy == "risk_priced"
            if emergency_deferred:
                safe[preferred_index] = True
            elif altitude_only:
                # Receding-horizon safety: require the native 2 s rollout to
                # stay above a hard floor, but do not require the terminal
                # state to be able to complete a full pull-out without future
                # replanning.  This avoids rejecting trajectories that remain
                # hundreds of metres clear and can be corrected next update.
                if full_check:
                    safe = prefiltered.copy()
                    safety_candidates_checked = len(order)
                else:
                    for index in order:
                        safety_candidates_checked += 1
                        if prefiltered[index]:
                            safe[index] = True
                            break
            elif risk_priced:
                # Exact branch-and-bound for a soft terminal-risk objective.
                # A candidate's adjusted score can never exceed its native
                # score, so once the next raw score cannot beat the incumbent
                # there is no need to roll out the rest of the population.
                safety_selection_scores.fill(float("-inf"))
                best_adjusted_score = float("-inf")
                for index in order:
                    if (
                        not full_check
                        and math.isfinite(best_adjusted_score)
                        and float(scores[index]) <= best_adjusted_score
                    ):
                        break
                    safety_candidates_checked += 1
                    if not prefiltered[index]:
                        continue
                    safety_rollout_count += 1
                    try:
                        terminal_margins[index] = self._candidate_pullout_margin_m(
                            state,
                            pool_controls[index],
                            pool_diagnostics[index],
                        )
                        deficit_m = max(0.0, -float(terminal_margins[index]))
                        adjusted_score = float(ranking_scores[index]) - (
                            self.safety.safety_risk_penalty_per_m * deficit_m
                        )
                        safety_selection_scores[index] = adjusted_score
                        safe[index] = math.isfinite(adjusted_score)
                        best_adjusted_score = max(
                            best_adjusted_score,
                            adjusted_score,
                        )
                    except Exception:
                        safe[index] = False
            else:
                for index in order:
                    safety_candidates_checked += 1
                    if not prefiltered[index]:
                        continue
                    safety_rollout_count += 1
                    try:
                        terminal_margins[index] = self._candidate_pullout_margin_m(
                            state,
                            pool_controls[index],
                            pool_diagnostics[index],
                        )
                        safe[index] = terminal_margins[index] >= 0.0
                    except Exception:
                        safe[index] = False
                    if safe[index] and not full_check:
                        break

            # Full mode reports the exact safe population. Lazy mode reports
            # the cheap prefilter fraction and marks its mode explicitly in
            # the result; unvisited terminal states are intentionally unknown.
            safe_fraction = 1.0 if emergency_deferred else float(
                np.mean(safe if full_check else prefiltered)
            )
            native_best_safe = bool(safe[native_index])
            native_min_altitude_m = (
                float(native_best_diagnostic.min_altitude_ft) * 0.3048
            )
            native_pullout_margin_m = float(terminal_margins[native_index])
            safe_indices = np.flatnonzero(safe)
            if not safe_indices.size:
                action, _ = self._recovery_action(state)
                self._last_action = action.astype(np.float64)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._adapt_profile(elapsed_ms)
                return SafePlanResult(
                    action=action,
                    score=native_best_score,
                    elapsed_ms=elapsed_ms,
                    candidate_count=candidate_count,
                    valid_fraction=valid_fraction,
                    safe_fraction=safe_fraction,
                    predicted_damage_dealt=float(
                        native_best_diagnostic.predicted_damage_dealt
                    ),
                    predicted_damage_taken=float(
                        native_best_diagnostic.predicted_damage_taken
                    ),
                    min_altitude_ft=float(native_best_diagnostic.min_altitude_ft),
                    min_range_ft=float(native_best_diagnostic.min_range_ft),
                    final_ata_deg=float(native_best_diagnostic.final_ata_deg),
                    native_best_safe=False,
                    safety_reranked=False,
                    selected_native_rank=0,
                    safety_check_mode=self.safety.safety_check_mode,
                    safety_candidates_checked=safety_candidates_checked,
                    safety_rollout_count=safety_rollout_count,
                    safety_policy=self.safety.safety_policy,
                    candidate_selection=selection_mode,
                    selection_changed=selection_changed,
                    selection_native_score_loss=selection_native_score_loss,
                    current_altitude_m=current_altitude_m,
                    current_sink_mps=current_sink_mps,
                    current_pullout_margin_m=current_pullout_margin_m,
                    native_min_altitude_m=native_min_altitude_m,
                    native_pullout_margin_m=native_pullout_margin_m,
                    safety_override=True,
                )

            safe_index = int(
                safe_indices[
                    np.argmax(safety_selection_scores[safe_indices])
                ]
            )
            safety_selection_changed = safe_index != preferred_index
            selected_controls = pool_controls[safe_index]
            selected_diagnostic = pool_diagnostics[safe_index]
            selected_score = float(scores[safe_index])
            selected_rank = 1 + int(np.sum(scores > selected_score))
            selected_pullout_margin_m = float(terminal_margins[safe_index])
            robust_candidate_count = 0
            robust_scenario_count = 1
            robust_selected = False
            robust_native_loss = 0.0
            robust_regret_gain = 0.0
            if self.safety.robust_enabled:
                safe_pool_controls = [pool_controls[index] for index in safe_indices]
                safe_pool_diagnostics = [
                    pool_diagnostics[index] for index in safe_indices
                ]
                safe_pool_scores = [float(scores[index]) for index in safe_indices]
                (
                    selected_controls,
                    selected_diagnostic,
                    selected_score,
                    selected_rank,
                    robust_candidate_count,
                    robust_scenario_count,
                    robust_selected,
                    robust_native_loss,
                    robust_regret_gain,
                ) = self._robust_select(
                    state,
                    response_trajectories or [],
                    safe_pool_controls,
                    safe_pool_diagnostics,
                    safe_pool_scores,
                )

            action = self._clip(selected_controls[0]).astype(np.float32)
            if not np.all(np.isfinite(action)):
                raise FloatingPointError("safe MPC produced a non-finite action")
            self._last_action = action.astype(np.float64)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._adapt_profile(elapsed_ms)
            # If the native winner is unsafe, final selection necessarily
            # changed even when another candidate happens to tie its score.
            safety_reranked = safety_selection_changed
            native_score_loss = native_best_score - selected_score
            first_action_delta_l2 = float(
                np.linalg.norm(
                    np.asarray(native_best_controls[0], dtype=np.float64)
                    - np.asarray(selected_controls[0], dtype=np.float64)
                )
            )
            return SafePlanResult(
                action=action,
                score=selected_score,
                elapsed_ms=elapsed_ms,
                candidate_count=candidate_count,
                valid_fraction=valid_fraction,
                safe_fraction=safe_fraction,
                predicted_damage_dealt=float(
                    selected_diagnostic.predicted_damage_dealt
                ),
                predicted_damage_taken=float(
                    selected_diagnostic.predicted_damage_taken
                ),
                min_altitude_ft=float(selected_diagnostic.min_altitude_ft),
                min_range_ft=float(selected_diagnostic.min_range_ft),
                final_ata_deg=float(selected_diagnostic.final_ata_deg),
                native_best_safe=native_best_safe,
                safety_reranked=safety_reranked,
                selected_native_rank=selected_rank,
                safety_check_mode=self.safety.safety_check_mode,
                safety_candidates_checked=safety_candidates_checked,
                safety_rollout_count=safety_rollout_count,
                safety_policy=self.safety.safety_policy,
                current_altitude_m=current_altitude_m,
                current_sink_mps=current_sink_mps,
                current_pullout_margin_m=current_pullout_margin_m,
                native_min_altitude_m=native_min_altitude_m,
                native_pullout_margin_m=native_pullout_margin_m,
                selected_pullout_margin_m=selected_pullout_margin_m,
                native_score_loss=native_score_loss,
                first_action_delta_l2=first_action_delta_l2,
                robust_scenario_count=robust_scenario_count,
                robust_candidate_count=robust_candidate_count,
                robust_selected=robust_selected,
                robust_native_loss=robust_native_loss,
                robust_regret_gain=robust_regret_gain,
                candidate_selection=selection_mode,
                selection_changed=selection_changed,
                selection_native_score_loss=selection_native_score_loss,
            )
        except Exception as exc:
            action = self._last_action.astype(np.float32, copy=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._adapt_profile(elapsed_ms)
            return SafePlanResult(
                action=action,
                score=-1.0e12,
                elapsed_ms=elapsed_ms,
                candidate_count=candidate_count,
                valid_fraction=valid_fraction,
                safe_fraction=safe_fraction,
                predicted_damage_dealt=0.0,
                predicted_damage_taken=0.0,
                min_altitude_ft=float("nan"),
                min_range_ft=float("nan"),
                final_ata_deg=float("nan"),
                native_best_safe=False,
                safety_reranked=False,
                selected_native_rank=0,
                safety_check_mode=self.safety.safety_check_mode,
                safety_candidates_checked=safety_candidates_checked,
                safety_rollout_count=safety_rollout_count,
                safety_policy=self.safety.safety_policy,
                candidate_selection=self.safety.candidate_selection,
                fallback=True,
                error=str(exc),
            )
