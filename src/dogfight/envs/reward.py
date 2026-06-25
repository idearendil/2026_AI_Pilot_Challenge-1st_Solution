from __future__ import annotations

from dogfight.sim.state_schema import StateIndex


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Compute step reward and return (total, components) tuple.

    components keys: step, pursuit, damage, safety, terminal
    """
    reward_mode = reward_config.get("mode", "default")
    if reward_mode not in (None, "default"):
        raise ValueError(
            f"Release reward mode {reward_mode!r} is not supported. "
            "Research-only reward modes such as ref_old_1vs1 live in MyTrainEnv."
        )

    components: dict[str, float] = {}

    # ── 0. Survival bonus (curriculum Stage 0 only, defaults to 0) ────────
    r_survival = float(reward_config.get("survival_bonus", 0.0))
    components["survival"] = r_survival

    # ── 1. Step penalty (time efficiency) ─────────────────────────────────
    r_step = float(reward_config["step_penalty"])
    components["step"] = r_step

    # ── 2. Pursuit shaping: smooth ATA × range gradient ───────────────────
    #   Replaces the old binary wez_bonus with a continuous gradient that
    #   provides learning signal even before entering the narrow WEZ cone.
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = abs(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    half_angle = float(reward_config["pursuit_half_angle_deg"])
    pursuit_range = float(reward_config["pursuit_range_m"])
    ata_factor = max(0.0, 1.0 - ata / half_angle)
    range_factor = max(0.0, 1.0 - distance / pursuit_range)
    r_pursuit = float(reward_config["pursuit_scale"]) * ata_factor * range_factor
    components["pursuit"] = r_pursuit

    # ── 3. Damage differential ─────────────────────────────────────────────
    #   Peaks inside the WEZ naturally — no separate wez_bonus needed.
    #   Scale reduced (200 → 20) so terminal rewards retain directional pull.
    r_damage = float(reward_config["damage_scale"]) * (target_damage - ownship_damage)
    components["damage"] = r_damage

    # ── 4. Safety: low altitude penalty ───────────────────────────────────
    r_safety = 0.0
    if float(ownship_state[StateIndex.ALT]) < 600.0:
        r_safety = -float(reward_config["low_altitude_penalty"])
    components["safety"] = r_safety

    # ── 5. Terminal reward ─────────────────────────────────────────────────
    #   timeout_health_scale removed — damage_scale already integrates health
    #   differential throughout the episode.
    r_terminal = 0.0
    if terminated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition == "two circle headon guard fail":
            r_terminal = float(reward_config.get("guard_fail_penalty", -50.0))
        elif target_health <= 0.0 < ownship_health:
            r_terminal = float(reward_config["win_reward"])
        elif ownship_health <= 0.0 < target_health:
            r_terminal = float(reward_config["loss_reward"])
        else:
            r_terminal = float(reward_config["draw_reward"])
    components["terminal"] = r_terminal

    total = r_survival + r_step + r_pursuit + r_damage + r_safety + r_terminal
    return float(total), components


def describe_reward(reward_config: dict, wez_config: dict) -> dict:
    return {
        "description": (
            "Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) "
            "+ damage differential + low altitude penalty + terminal rewards."
        ),
        "survival_bonus": reward_config.get("survival_bonus", 0.0),
        "step_penalty": reward_config["step_penalty"],
        "damage_scale": reward_config["damage_scale"],
        "pursuit_scale": reward_config.get("pursuit_scale", 0.0),
        "pursuit_half_angle_deg": reward_config.get("pursuit_half_angle_deg", 30.0),
        "pursuit_range_m": reward_config.get("pursuit_range_m", 3000.0),
        "low_altitude_penalty": reward_config["low_altitude_penalty"],
        "win_reward": reward_config["win_reward"],
        "loss_reward": reward_config["loss_reward"],
        "draw_reward": reward_config["draw_reward"],
        "guard_fail_penalty": reward_config.get("guard_fail_penalty", -50.0),
        "wez": {
            "angle_deg": wez_config["angle_deg"],
            "min_range_m": wez_config["min_range_m"],
            "max_range_m": wez_config["max_range_m"],
        },
    }
