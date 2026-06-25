from __future__ import annotations

from dogfight.sim.state_schema import StateIndex


def evaluate_termination(
    ownship_state,
    target_state,
    ownship_sim,
    target_sim,
    max_engage_time: float,
    min_altitude: float,
    current_timestep: int,
    episode_step_limit: int | None,
    geo_info=None,
    geometry_guard: dict | None = None,
):
    terminated = False
    truncated = False
    end_condition = ""

    if not ownship_sim.fdm_update_success or not target_sim.fdm_update_success:
        terminated = True
        end_condition = "FDM Update Fail"
    elif ownship_state[StateIndex.ALT] < min_altitude:
        terminated = True
        end_condition = "ownship altitude below min"
    elif target_state[StateIndex.ALT] < min_altitude:
        terminated = True
        end_condition = "target altitude below min"
    elif ownship_state[StateIndex.HEALTH] <= 0:
        terminated = True
        end_condition = "ownship destroyed"
    elif target_state[StateIndex.HEALTH] <= 0:
        terminated = True
        end_condition = "target destroyed"
    elif ownship_state[StateIndex.FUEL] == 0 or target_state[StateIndex.FUEL] == 0:
        terminated = True
        end_condition = "fuel fail"
    elif _is_two_circle_guard_failed(
        ownship_state,
        target_state,
        geo_info,
        geometry_guard,
    ):
        terminated = True
        end_condition = "two circle headon guard fail"
    elif ownship_state[StateIndex.SIM_TIME] > max_engage_time:
        truncated = True
        end_condition = "max time out"
    elif episode_step_limit is not None and current_timestep >= episode_step_limit:
        truncated = True
        end_condition = "episode step limit"

    return terminated, truncated, end_condition


def _is_two_circle_guard_failed(
    ownship_state,
    target_state,
    geo_info,
    geometry_guard: dict | None,
) -> bool:
    """Return True when the optional two-circle head-on guard is violated."""
    if geo_info is None or not geometry_guard:
        return False
    if not geometry_guard.get("enabled", False):
        return False
    if geometry_guard.get("mode") != "two_circle_headon":
        return False

    alpha_deg = abs(float(geometry_guard.get("alpha_deg", 0.0)))
    ata_deg = abs(
        float(geo_info._get_antenna_train_angle(ownship_state, target_state, True))
    )
    early_max = float(geometry_guard.get("early_alpha_max_deg", 80.0))
    mid_max = float(geometry_guard.get("mid_alpha_max_deg", 140.0))

    if alpha_deg <= early_max:
        return ata_deg > float(geometry_guard.get("crossed_ata_limit_deg", 90.0))
    if alpha_deg <= mid_max:
        margin = float(geometry_guard.get("turn_margin_deg", 10.0))
        return ata_deg > alpha_deg + margin
    return False
