# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

여기서 보상을 직접 설계한다. 학습에만 영향을 주고(추론/제출에는 영향 없음),
`train.py --reward-module claude_code.my_reward` 로 활성화한다.

계약(원본 프레임워크와 동일):
  - MY_REWARD_CONFIG: dict (계수 모음)
  - compute_reward(...) -> (total: float, components: dict)
    components 의 각 항목은 학습 로그/대시보드에 ep_reward_<name> 으로 기록된다.

기본값은 원본 기본 보상(`src/dogfight/envs/reward.py`)과 동일하게 맞춰 두었으므로,
활성화만 해서는 결과가 바뀌지 않는다. 아래 항목을 자유롭게 수정/추가하라.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    "step_penalty": -0.01,
    "pursuit_scale": 0.3,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "damage_scale": 20.0,
    "low_altitude_penalty": 0.1,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "guard_fail_penalty": -50.0,
    "survival_bonus": 0.0,
}


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
    """step 보상과 (total, components) 반환.

    유용한 입력:
      geo_info._get_distance(ownship_state, target_state)
      geo_info._get_antenna_train_angle(ownship_state, target_state, False)  # ATA
      geo_info._get_aspect_angle(ownship_state, target_state, False)         # AA
      ownship_damage, target_damage, wez_config, ownship_state[StateIndex.*]
    """
    c: dict[str, float] = {}

    # 1) 생존 보너스 + 시간 페널티
    c["survival"] = float(reward_config.get("survival_bonus", 0.0))
    c["step"] = float(reward_config.get("step_penalty", -0.01))

    # 2) 추격 shaping: ATA × range 의 부드러운 gradient (WEZ 진입 전에도 신호 제공)
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = abs(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    half_angle = float(reward_config.get("pursuit_half_angle_deg", 30.0))
    pursuit_range = float(reward_config.get("pursuit_range_m", 3000.0))
    ata_factor = max(0.0, 1.0 - ata / half_angle)
    range_factor = max(0.0, 1.0 - distance / pursuit_range)
    c["pursuit"] = float(reward_config.get("pursuit_scale", 0.3)) * ata_factor * range_factor

    # 3) 피해 차분 (WEZ 안에서 자연히 커진다)
    c["damage"] = float(reward_config.get("damage_scale", 20.0)) * (target_damage - ownship_damage)

    # 4) 저고도 안전 페널티
    c["safety"] = (
        -float(reward_config.get("low_altitude_penalty", 0.1))
        if float(ownship_state[StateIndex.ALT]) < 600.0
        else 0.0
    )

    # 5) 종료 보상
    r_terminal = 0.0
    if terminated:
        oh = float(ownship_state[StateIndex.HEALTH])
        th = float(target_state[StateIndex.HEALTH])
        if end_condition == "two circle headon guard fail":
            r_terminal = float(reward_config.get("guard_fail_penalty", -50.0))
        elif th <= 0.0 < oh:
            r_terminal = float(reward_config.get("win_reward", 100.0))
        elif oh <= 0.0 < th:
            r_terminal = float(reward_config.get("loss_reward", -100.0))
        else:
            r_terminal = float(reward_config.get("draw_reward", -30.0))
    c["terminal"] = r_terminal

    return float(sum(c.values())), c


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
