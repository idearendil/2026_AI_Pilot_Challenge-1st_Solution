# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

규칙:
  - [종료] 상대 HP<=0 또는 상대가 최소고도(=300m≈1000ft) 이하로 종료  →  +win_reward(10)
  - [종료] 내 HP<=0   또는 내가  최소고도 이하로 종료                  →  +loss_reward(-10)
  - [보조] 양측이 살아있는(HP>0) 매 step:
           reward += (상대 HP 감소량 - 본인 HP 감소량) * damage_scale(10)
           (HP 감소량 = 이번 step 에 입은 damage = 함수 인자 target_damage / ownship_damage)

claude_code/train.py 는 기본적으로 이 모듈을 사용한다(끄려면 --reward-module "").

계약: MY_REWARD_CONFIG(dict) + compute_reward(...) -> (total: float, components: dict).
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
    "win_reward": 10.0,
    "loss_reward": -10.0,
    "damage_scale": 10.0,   # (상대 HP감소 - 내 HP감소) * 이 값, 양측 생존 중 매 step
}

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# min_altitude(기본 300m≈984ft) 에서 발생하므로 "1000ft 이하 종료"와 사실상 동일.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"


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
    """종료 ±win/loss + 양측 생존 중 HP 차분 shaping."""
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])

    # [보조] 양측 생존(HP>0) 중에만: (상대 HP감소 - 내 HP감소) * damage_scale.
    # target_damage / ownship_damage = 이번 step 에 각 기체가 입은 damage(=HP 감소량).
    r_damage = 0.0
    if own_hp > 0.0 and tgt_hp > 0.0:
        r_damage = (float(target_damage) - float(ownship_damage) * 0.75) * float(
            reward_config.get("damage_scale", 10.0)
        )

    # [종료] 기존 ±10 그대로
    r_terminal = 0.0
    if terminated:
        target_lost = tgt_hp <= 0.0 or end_condition == _TARGET_ALT_END
        ownship_lost = own_hp <= 0.0 or end_condition == _OWNSHIP_ALT_END
        if target_lost:
            r_terminal += float(reward_config.get("win_reward", 10.0))
        if ownship_lost:
            r_terminal += float(reward_config.get("loss_reward", -10.0))

    total = r_damage + r_terminal
    return float(total), {"damage": r_damage, "terminal": r_terminal}


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
