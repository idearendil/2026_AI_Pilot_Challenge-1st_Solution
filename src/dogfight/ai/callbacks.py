from __future__ import annotations

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class DogFightCallbacks(DefaultCallbacks):
    """RLlib callbacks that collect per-episode dogfight metrics.

    Metrics recorded in episode.custom_metrics (auto-aggregated by RLlib):
      Outcome    : win, loss, draw, timeout, crash
      Reward     : ep_reward_{step,pursuit,damage,safety,terminal}
      Tactical   : ep_wez_steps, ep_mean_distance, ep_min_distance,
                   ep_altitude_penalty_steps, initial/final ATA/AA,
                   headon_guard_fail
      Action     : action_{roll,pitch,rudder,throttle}_{mean,std},
                   action_saturation_rate
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_episode_start(self, *, episode, **kwargs):
        self._episode_data(episode)["actions"] = []

    def on_episode_step(self, *, episode, **kwargs):
        action = self._last_action(episode)
        if action is not None:
            try:
                self._episode_data(episode)["actions"].append(
                    np.asarray(action, dtype=np.float32)
                )
            except Exception:
                pass

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        info = self._last_info(episode)
        if not info:
            return

        # ── Outcome ───────────────────────────────────────────────────────
        outcome = info.get("outcome", "other")
        for key in ("win", "loss", "draw", "timeout", "crash"):
            self._record_metric(episode, metrics_logger, key, float(outcome == key))

        # ── Reward components (cumulative episode totals) ─────────────────
        for key, val in info.get("ep_reward_components", {}).items():
            self._record_metric(
                episode, metrics_logger, f"ep_reward_{key}", float(val)
            )

        # ── Tactical metrics ──────────────────────────────────────────────
        self._record_metric(
            episode, metrics_logger, "ep_wez_steps", float(info.get("ep_wez_steps", 0))
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_mean_distance",
            float(info.get("ep_mean_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_min_distance",
            float(info.get("ep_min_distance", 0.0)),
        )
        self._record_metric(
            episode,
            metrics_logger,
            "ep_altitude_penalty_steps",
            float(info.get("ep_altitude_penalty_steps", 0)),
        )
        for key in (
            "initial_alpha_deg",
            "initial_ata_deg",
            "initial_aa_deg",
            "initial_distance_m",
            "final_ata_deg",
            "final_aa_deg",
        ):
            if key in info:
                self._record_metric(
                    episode, metrics_logger, key, float(info.get(key, 0.0))
                )
        self._record_metric(
            episode,
            metrics_logger,
            "headon_guard_fail",
            float(bool(info.get("headon_guard_fail", False))),
        )

        # ── Action distribution ───────────────────────────────────────────
        actions = self._episode_data(episode).get("actions", [])
        if actions:
            arr = np.stack(actions)  # (steps, 4)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            for i, name in enumerate(("roll", "pitch", "rudder", "throttle")):
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_mean", float(means[i])
                )
                self._record_metric(
                    episode, metrics_logger, f"action_{name}_std", float(stds[i])
                )
            # Saturation: fraction of steps where any axis hits ±1
            self._record_metric(
                episode,
                metrics_logger,
                "action_saturation_rate",
                float(np.mean(np.abs(arr) >= 0.99)),
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _episode_data(episode) -> dict:
        """Return mutable per-episode storage for old and new RLlib APIs."""
        if hasattr(episode, "user_data"):
            return episode.user_data
        return episode.custom_data

    @staticmethod
    def _record_metric(episode, metrics_logger, key: str, value: float) -> None:
        """Record a metric through the callback API available in this RLlib version."""
        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics[key] = value
        elif metrics_logger is not None:
            metrics_logger.log_value(("custom_metrics", key), value, reduce="mean")

    @staticmethod
    def _last_action(episode):
        """Retrieve last action, handling both old and new RLlib API."""
        try:
            return episode.last_action_for()
        except Exception:
            pass

        try:
            return episode.get_actions(-1)
        except Exception:
            return None

    @staticmethod
    def _last_info(episode) -> dict:
        """Retrieve last info dict, handling both old and new RLlib API."""
        try:
            return episode.last_info_for() or {}
        except TypeError:
            # New API: requires agent_id kwarg
            try:
                return episode.last_info_for(agent_id=None) or {}
            except Exception:
                pass
        except Exception:
            pass
        try:
            return episode.get_infos(-1) or {}
        except Exception:
            pass
        return {}
