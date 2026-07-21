# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

규칙(종료 보상 3분리):
  - [종료·HP] 상대 HP<=0 으로 종료(내가 이김)          →  +win_reward(10)
              내 HP<=0   으로 종료(상대가 이김)         →  +loss_reward(-10)
  - [종료·고도] 내 고도가 최소고도(=300m≈1000ft) 이하로 종료  →  ownship_alt_reward(-20)
                상대 고도가 최소고도 이하로 종료             →  target_alt_reward(+1)
  - [보조] 양측이 살아있는(HP>0) 매 step:
           reward += (상대 HP 감소량 - 본인 HP 감소량) * damage_scale(10)
           (HP 감소량 = 이번 step 에 입은 damage = 함수 인자 target_damage / ownship_damage)
  - [보조] 거리 접근: 직전 RL-step(=6 sim sub-step) 대비 두 기체 거리가 d[m] 줄었으면
           reward += d * distance_reward_scale(0.001) * taper. 멀어지면 d<0 → 음의 보상.
           taper 는 직전/현재 거리의 **평균** 기준: 3000ft 이상 +1.0, 3000→500ft 선형 감소,
           500ft 이하는 부호 반전 -1.0 (WEZ 가 500~3000ft 라 더 붙으면 오히려 해롭다).
  - [보조] 조준: 직전 step 대비 LOS(|ATA|) 각도가 a[deg] 줄었으면
           reward += a * aim_reward_scale(0.02). 늘어나면 a<0 → 음의 보상.
           거리 항과 동일한 차분 형태라 텔레스코핑(episode 총합 = (첫 ATA - 끝 ATA)*scale).

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


_FT_TO_M = 0.3048

MY_REWARD_CONFIG = {
    "win_reward": 10.0,       # 상대 HP<=0 으로 종료(내가 이김)
    "loss_reward": -10.0,     # 내 HP<=0 으로 종료(상대가 이김)
    "ownship_alt_reward": -20.0,   # 내 고도가 최소고도 이하로 떨어져 종료
    "target_alt_reward": 1.0,      # 상대 고도가 최소고도 이하로 떨어져 종료
    "damage_scale": 10.0,   # (상대 HP감소 - 내 HP감소) * 이 값, 양측 생존 중 매 step
    "distance_reward_scale": 0.001,   # 직전 step 대비 줄어든 거리[m] * 이 값
    # 거리 보상 taper: WEZ(500~3000ft) 밖에서만 접근을 장려한다. 직전/현재 거리의 평균이
    # far 이상이면 계수 +1, 그 사이는 선형 감소, near 이하면 -1(접근에 페널티).
    "distance_taper_far_ft": 3000.0,
    "distance_taper_near_ft": 500.0,
    # 조준 shaping: 직전 step 대비 줄어든 LOS(ATA) 각도[deg] * 이 값. 늘어나면 음수.
    # distance 항과 에피소드 총합이 비슷해지도록 맞춘 값:
    #   distance  2100m × 0.001 ≈ 2.1   /   aim  90deg × 0.02 ≈ 1.8
    "aim_reward_scale": 0.02,
}
# 주의: 아래 compute_reward 는 이 dict 의 키를 **직접 인덱싱**한다(.get 폴백 없음).
# 계수를 바꾸려면 반드시 이 dict(또는 train.py 의 --*-reward-scale 오버라이드)를 고칠 것.
# 예전에는 .get(key, 폴백) 형태라 폴백만 고치면 아무 효과가 없는 함정이 있었다.

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# min_altitude(기본 300m≈984ft) 에서 발생하므로 "1000ft 이하 종료"와 사실상 동일.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"

# dense shaping 직전-step 상태 (모듈 싱글톤). 에피소드 경계는 SIM_TIME 으로 감지.
_prev_distance_m: float | None = None
_prev_sim_time: float | None = None
_prev_ata_deg: float = 0.0   # 직전 step 의 |ATA|[deg] (조준 차분 shaping용)


def reset_distance_tracker() -> None:
    """에피소드 시작 시 호출(선택). 호출 안 해도 SIM_TIME 으로 자동 감지된다."""
    global _prev_distance_m, _prev_sim_time, _prev_ata_deg
    _prev_distance_m = None
    _prev_sim_time = None
    _prev_ata_deg = 0.0


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
    """종료 ±win/loss + HP 차분 damage + 거리 접근 shaping + ATA 차분 조준 shaping."""
    global _prev_distance_m, _prev_sim_time, _prev_ata_deg
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])

    # [보조] 양측 생존(HP>0) 중에만: (상대 HP감소 - 내 HP감소) * damage_scale.
    # target_damage / ownship_damage = 이번 step 에 각 기체가 입은 damage(=HP 감소량).
    r_damage = 0.0
    if own_hp > 0.0 and tgt_hp > 0.0:
        r_damage = (float(target_damage) * 1.0 - float(ownship_damage) * 0.5) * float(
            reward_config["damage_scale"]
        )

    # [보조] 거리 접근: 직전 RL-step 대비 거리가 줄어든 양(m) * distance_reward_scale.
    # SIM_TIME 으로 에피소드 경계 감지(거꾸로 가거나 처음이면 새 episode → 보상 0).
    cur_distance = float(geo_info._get_distance(ownship_state, target_state))
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    new_episode = _prev_distance_m is None or cur_sim_time <= _prev_sim_time
    r_distance = 0.0
    if not new_episode:
        closed = _prev_distance_m - cur_distance   # 가까워졌으면 양수, 멀어졌으면 음수
        # WEZ(500~3000ft) 안에서는 더 붙어도 damage 가 안 들어가므로 접근 보상을 죽인다.
        # 기준 거리 = 직전/현재 거리의 평균.
        far_m = float(reward_config["distance_taper_far_ft"]) * _FT_TO_M
        near_m = float(reward_config["distance_taper_near_ft"]) * _FT_TO_M
        # near 이하로 더 붙는 것은 오히려 해로우므로 계수 부호를 뒤집는다(-1).
        mid_distance = 0.5 * (_prev_distance_m + cur_distance)
        if mid_distance <= near_m:
            taper = -1.0
        elif far_m > near_m:
            taper = (mid_distance - near_m) / (far_m - near_m)
            taper = min(1.0, max(0.0, taper))
        else:
            taper = 1.0
        r_distance = closed * float(reward_config["distance_reward_scale"]) * taper
    _prev_distance_m = cur_distance
    _prev_sim_time = cur_sim_time

    # [보조] 조준: 거리 항과 동일한 차분 형태. 직전 step 대비 LOS(|ATA|) 각도가 a[deg]
    # 줄었으면 reward += a * aim_reward_scale, 늘어나면 a<0 → 음의 보상. 텔레스코핑이라
    # episode 총합 = (첫 ATA - 마지막 ATA) * scale 로 bounded. scale=0 이면 끔.
    aim_scale = float(reward_config["aim_reward_scale"])
    r_aim = 0.0
    if aim_scale != 0.0:
        cur_ata = abs(float(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
        if not new_episode:
            r_aim = (_prev_ata_deg - cur_ata) * aim_scale
        _prev_ata_deg = cur_ata

    # [종료] 3분리: (1) HP 승/패 ±10  (2) 내 고도 하락 -20  (3) 상대 고도 하락 +1
    r_terminal = 0.0
    if terminated:
        # (1) HP 로 승부가 난 경우: 내가 이김 +10 / 상대가 이김 -10
        if tgt_hp <= 0.0:
            r_terminal += float(reward_config["win_reward"])
        if own_hp <= 0.0:
            r_terminal += float(reward_config["loss_reward"])
        # (2) 내 고도가 최소고도 이하로 떨어져 종료 → -20
        if end_condition == _OWNSHIP_ALT_END:
            r_terminal += float(reward_config["ownship_alt_reward"])
        # (3) 상대 고도가 최소고도 이하로 떨어져 종료 → +1
        if end_condition == _TARGET_ALT_END:
            r_terminal += float(reward_config["target_alt_reward"])

    total = r_damage + r_distance + r_aim + r_terminal
    return float(total), {"damage": r_damage, "distance": r_distance,
                          "aim": r_aim, "terminal": r_terminal}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "reset_distance_tracker"]
