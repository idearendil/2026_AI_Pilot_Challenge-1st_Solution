# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

규칙:
  - [종료] 상대 HP<=0 또는 상대가 최소고도(=300m≈1000ft) 이하로 종료  →  +win_reward(10)
  - [종료] 내 HP<=0   또는 내가  최소고도 이하로 종료                  →  +loss_reward(-10)
  - [보조] 양측이 살아있는(HP>0) 매 step:
           reward += (상대 HP 감소량 - 본인 HP 감소량) * damage_scale(10)
           (HP 감소량 = 이번 step 에 입은 damage = 함수 인자 target_damage / ownship_damage)
  - [보조] 거리 접근: 직전 RL-step(=6 sim sub-step) 대비 두 기체 거리가 d[m] 줄었으면
           reward += d * distance_reward_scale(0.001). 멀어지면 d<0 → 음의 보상.
  - [보조] 조준 dense shaping(potential-based, 기본 off=0): P=(1+cos ATA)/2 × clip(1-dist/range,0,1),
           reward += (P_now - P_prev) * aim_reward_scale. 조준/사거리 개선=+, 텔레스코핑이라 episode
           총합 bounded(damage 에 종속). damage 보다 작은 보조 보상으로만 쓴다.

claude_code/train.py 는 기본적으로 이 모듈을 사용한다(끄려면 --reward-module "").

계약: MY_REWARD_CONFIG(dict) + compute_reward(...) -> (total: float, components: dict).

거리 접근 보상은 직전 step 의 거리를 모듈 상태로 추적해야 한다(reward_fn 은 stateless 계약).
에피소드 경계는 SIM_TIME 으로 감지한다(NaN-crash 등 compute_reward 를 건너뛰는 종료 경로도
안전하게 처리). 보상은 학습/평가 env 에서만 계산되고 추론(제출)에서는 호출되지 않으므로
SIM_TIME 이 항상 채워져 있어 안전하다. (병렬 워커는 프로세스별 별도 모듈이라 상태 충돌 없음;
단일 프로세스 eval 은 끝나고 rollout env 를 reset 하므로 새 episode 로 자동 감지된다.)
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


import math

MY_REWARD_CONFIG = {
    "win_reward": 10.0,
    "loss_reward": -10.0,
    "damage_scale": 10.0,   # (상대 HP감소 - 내 HP감소) * 이 값, 양측 생존 중 매 step
    "distance_reward_scale": 0.001,   # 직전 step 대비 줄어든 거리[m] * 이 값
    # 조준 dense shaping(potential-based) 계수. 0 이면 끔. damage 보다 작게 유지(보조 보상).
    "aim_reward_scale": 0.0,
    "aim_range_m": 3000.0,   # range_factor=clip(1-dist/이값,0,1) 의 스케일
}

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# min_altitude(기본 300m≈984ft) 에서 발생하므로 "1000ft 이하 종료"와 사실상 동일.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"

# dense shaping 직전-step 상태 (모듈 싱글톤). 에피소드 경계는 SIM_TIME 으로 감지.
_prev_distance_m: float | None = None
_prev_sim_time: float | None = None
_prev_aim_pot: float = 0.0   # 직전 step 의 조준 potential P (potential-based shaping용)


def reset_distance_tracker() -> None:
    """에피소드 시작 시 호출(선택). 호출 안 해도 SIM_TIME 으로 자동 감지된다."""
    global _prev_distance_m, _prev_sim_time, _prev_aim_pot
    _prev_distance_m = None
    _prev_sim_time = None
    _prev_aim_pot = 0.0


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
    """종료 ±win/loss + HP 차분 damage + 거리 접근 shaping + 조준 dense shaping."""
    global _prev_distance_m, _prev_sim_time, _prev_aim_pot
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])

    # [보조] 양측 생존(HP>0) 중에만: (상대 HP감소 - 내 HP감소) * damage_scale.
    # target_damage / ownship_damage = 이번 step 에 각 기체가 입은 damage(=HP 감소량).
    r_damage = 0.0
    if own_hp > 0.0 and tgt_hp > 0.0:
        r_damage = (float(target_damage) * 1.0 - float(ownship_damage) * 0.0) * float(
            reward_config.get("damage_scale", 10.0)
        )

    # [보조] 거리 접근: 직전 RL-step 대비 거리가 줄어든 양(m) * distance_reward_scale.
    # SIM_TIME 으로 에피소드 경계 감지(거꾸로 가거나 처음이면 새 episode → 보상 0).
    cur_distance = float(geo_info._get_distance(ownship_state, target_state))
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    new_episode = _prev_distance_m is None or cur_sim_time <= _prev_sim_time
    r_distance = 0.0
    if not new_episode:
        closed = _prev_distance_m - cur_distance   # 가까워졌으면 양수, 멀어졌으면 음수
        r_distance = closed * float(reward_config.get("distance_reward_scale", 0.0001))
    _prev_distance_m = cur_distance
    _prev_sim_time = cur_sim_time

    # [보조] 조준 dense shaping (potential-based). P = aim_factor × range_factor ∈ [0,1].
    #   aim_factor = (1+cos(ATA))/2  (보어사이트=1, 어느 각도서든 매끄러운 그래디언트)
    #   range_factor = clip(1 - dist/aim_range_m, 0, 1)  (가까울수록 1)
    # r_aim = (P_now - P_prev) × aim_reward_scale → 조준/사거리 개선=+, 악화=-. 텔레스코핑이라
    # episode 총합이 scale 로 bounded(damage 에 종속, 최적 정책 거의 불변). scale=0 이면 끔.
    aim_scale = float(reward_config.get("aim_reward_scale", 0.0))
    r_aim = 0.0
    if aim_scale != 0.0:
        ata = float(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
        aim_factor = (1.0 + math.cos(math.radians(ata))) / 2.0
        range_m = float(reward_config.get("aim_range_m", 3000.0))
        range_factor = max(0.0, 1.0 - cur_distance / range_m) if range_m > 0 else 0.0
        cur_aim_pot = aim_factor * range_factor
        if not new_episode:
            r_aim = (cur_aim_pot - _prev_aim_pot) * aim_scale
        _prev_aim_pot = cur_aim_pot

    # [종료] 기존 ±10 그대로
    r_terminal = 0.0
    if terminated:
        target_lost = tgt_hp <= 0.0 or end_condition == _TARGET_ALT_END
        ownship_lost = own_hp <= 0.0 or end_condition == _OWNSHIP_ALT_END
        if target_lost:
            r_terminal += float(reward_config.get("win_reward", 10.0))
        if ownship_lost:
            r_terminal += float(reward_config.get("loss_reward", -10.0))

    total = r_damage + r_distance + r_aim + r_terminal
    return float(total), {"damage": r_damage, "distance": r_distance,
                          "aim": r_aim, "terminal": r_terminal}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "reset_distance_tracker"]
