# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

규칙(종료 보상 3분리):
  - [종료·HP] 상대 HP<=0 으로 종료(내가 이김)          →  +win_reward(0)
              내 HP<=0   으로 종료(상대가 이김)         →  +loss_reward(0)
  - [종료·고도] 내 고도가 최소고도(=300m≈1000ft) 이하로 종료  →  ownship_alt_reward(-20)
                상대 고도가 최소고도 이하로 종료             →  target_alt_reward(+5)
  - [보조] 양측이 살아있는(HP>0) 매 step:
           reward += (상대 HP 감소량 - 본인 HP 감소량*own_damage_weight) * damage_scale(10)
           (HP 감소량 = 이번 step 에 입은 damage = 함수 인자 target_damage / ownship_damage)
           own_damage_weight 기본 0.5, 학습 단계 k>=2(2000 iter~)에서 1.0 으로 상향(스케줄).
  - [보조] 상황 포텐셜 shaping: 거리/조준을 하나의 포텐셜 함수 x 로 합친 뒤 그 step 차분에
           계수를 곱해 준다.  reward += (x_cur - x_prev) * shaping_reward_scale.
           x = _shaping_potential(distance[ft], A1[deg], A2[deg]) 이고
             A1 = 아군이 상대를 볼 때의 LOS(|ATA|),  A2 = 상대가 아군을 볼 때의 LOS(|ATA|).
           거리가 가깝고(단, WEZ 안쪽), 내가 잘 조준하고, 상대는 못 조준할수록 x 가 커진다.
           차분 형태라 텔레스코핑(episode 총합 = (x_end - x_start) * scale) → bounded.
           _shaping_potential 정의(연속함수, distance 는 ft, A 는 deg 0~180):
             distance ∈ [0, 500):     (d+14000)*(90-A1)/90*2.5 - (d+14000)*(90-A2)/90*2.5 + 999500
             distance ∈ [500, 15000): (15000-d) + (15000-d)*(90-A1)/90*2.5 - (15000-d)*(90-A2)/90*2.5 + 985000
             distance ∈ [15000, ∞):   1000000 - d
           (경계 500ft·15000ft 에서 연속. 값은 대략 ~1e6 근처.)
           추가로 아군 고도(ft)가 1000~10000ft 구간이면 -(alt-10000)^2/101.25 을 더한다
           (10000ft 0 / 1000ft -800000). telescoping 이라 10000→1000ft 강하 시 고도항
           총합 = -800000*shaping_scale(0.00005) = -40 (예전 상한 4000ft·총합 -1.8 강화).

claude_code/train.py 는 기본적으로 이 모듈을 사용한다(끄려면 --reward-module "").

계약: MY_REWARD_CONFIG(dict) + compute_reward(...) -> (total: float, components: dict).

포텐셜 shaping 은 직전 step 의 포텐셜값 x 를 모듈 상태로 추적해야 한다(reward_fn 은 stateless
계약). 에피소드 경계는 SIM_TIME 으로 감지한다(NaN-crash 등 compute_reward 를 건너뛰는 종료
경로도 안전하게 처리). 보상은 학습/평가 env 에서만 계산되고 추론(제출)에서는 호출되지 않으므로
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

# 고도 안전 shaping 파라미터(_shaping_potential 고도항). 아군 고도가 TOP 이하로
# 내려가면 포텐셜을 -(alt-TOP)^2/DIVISOR 만큼 떨어뜨려 하강을 억제한다.
# telescoping 이라 episode 총합 = (alt_pot(끝고도)-alt_pot(시작고도))*shaping_scale.
# 목표: TOP(4000ft, 항=0)에서 FLOOR(1000ft)까지 강하 시 고도항 총합 ≈ -40.
#   필요 potential 차 = -40 / shaping_scale(0.00005) = -800000
#   -(1000-4000)^2/DIVISOR = -9000000/DIVISOR = -800000  →  DIVISOR = 11.25
# 범위는 원래대로 좁혀(1000~4000ft) 세기만 강화 유지: 좁은 구간에 -40 을 몰아넣어
# 4000ft 아래에서 하강 억제 신호가 예전보다 훨씬 가파르다(1000ft potential -800000).
_ALT_SHAPING_FLOOR_FT = 1000.0    # 이 아래는 사실상 패배 임박(min_altitude 근처)
_ALT_SHAPING_TOP_FT = 4000.0      # 이 위는 안전 → 고도항 0 (zero point)
_ALT_SHAPING_DIVISOR = 11.25      # 4000→1000ft 강하 시 고도항 총합 = -40

MY_REWARD_CONFIG = {
    "win_reward": 0.0,       # 상대 HP<=0 으로 종료(내가 이김)
    "loss_reward": 0.0,     # 내 HP<=0 으로 종료(상대가 이김)
    "ownship_alt_reward": -20.0,   # 내 고도가 최소고도 이하로 떨어져 종료
    "target_alt_reward": 5.0,      # 상대 고도가 최소고도 이하로 떨어져 종료
    "damage_scale": 10.0,   # (상대 HP감소 - 내 HP감소*own_damage_weight) * 이 값, 양측 생존 중 매 step
    # 내 HP 감소량에 곱하는 가중치. 기본 0.5(상대 피해보다 절반만 반영). 학습 스케줄이
    # 2000 iter(단계 k>=2) 도달 시 1.0(상대와 동일 취급)으로 올린다(ppo._apply_iteration_schedule).
    "own_damage_weight": 0.5,
    # 상황 포텐셜 shaping 계수. reward += (x_cur - x_prev) * 이 값.
    # x 는 _shaping_potential(거리[ft], A1[deg], A2[deg]) 로 대략 ~1e6 스케일이다.
    #   대표 궤적(원거리 20000ft·조준無 → 근거리 1000ft·내 조준0°·상대 90°)의
    #   포텐셜 상승 x_end-x_start ≈ 54000 → 이 값 0.0001 이면 episode 총합 ≈ 5.4 로
    #   target_alt_reward(5)·damage 한 방(≈10)과 같은 자릿수. 상대도 나를 조준하면
    #   경쟁항이 상쇄돼 총합이 줄어든다(설계 의도).
    "shaping_reward_scale": 0.00005,
}
# 주의: 아래 compute_reward 는 이 dict 의 키를 **직접 인덱싱**한다(.get 폴백 없음).
# 계수를 바꾸려면 반드시 이 dict(또는 train.py 의 --*-reward-scale 오버라이드)를 고칠 것.
# 예전에는 .get(key, 폴백) 형태라 폴백만 고치면 아무 효과가 없는 함정이 있었다.

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# min_altitude(기본 300m≈984ft) 에서 발생하므로 "1000ft 이하 종료"와 사실상 동일.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"

# dense shaping 직전-step 상태 (모듈 싱글톤). 에피소드 경계는 SIM_TIME 으로 감지.
_prev_x: float | None = None       # 직전 step 의 포텐셜값 x (거리/조준 shaping용)
_prev_sim_time: float | None = None


def _shaping_potential(distance_ft: float, a1_deg: float, a2_deg: float,
                       own_alt_ft: float) -> float:
    """거리(ft)/조준(A1,A2 deg, 0~180) + 아군 고도(ft) 를 합친 상황 포텐셜.

    A1 = 아군→상대 LOS(|ATA|), A2 = 상대→아군 LOS(|ATA|). 값이 클수록 유리.
    경계 500ft·15000ft 에서 연속. (90-A) 항은 A>90(등 뒤) 이면 음수가 되어 자연스럽게
    페널티로 작동하므로 clamp 하지 않는다.

    고도 항: FLOOR(1000ft)~TOP(4000ft) 구간에서만 -(alt-TOP)^2/DIVISOR 를 더한다.
    (TOP=4000ft 에서 0, 1000ft 에서 -800000). 고도가 분계점(min_altitude≈1000ft)에
    가까워질수록 포텐셜이 낮아져(차분이 음수) 하강을 억제한다(고도 하락 패배 방지).
    telescoping 이라 4000→1000ft 강하 시 고도항 episode 총합 = -40(예전 -1.8 강화).
    """
    if distance_ft <= 500.0:
        base = distance_ft + 14000.0
        x = (base * (90.0 - a1_deg) / 90.0 * 2.5
             - base * (90.0 - a2_deg) / 90.0 * 2.5
             + 999500.0)
    elif distance_ft <= 15000.0:
        base = 15000.0 - distance_ft
        x = (base
             + base * (90.0 - a1_deg) / 90.0 * 2.5
             - base * (90.0 - a2_deg) / 90.0 * 2.5
             + 985000.0)
    else:
        x = 1000000.0 - distance_ft

    if _ALT_SHAPING_FLOOR_FT <= own_alt_ft <= _ALT_SHAPING_TOP_FT:
        d = own_alt_ft - _ALT_SHAPING_TOP_FT
        x += -(d * d) / _ALT_SHAPING_DIVISOR
    return x


def reset_distance_tracker() -> None:
    """에피소드 시작 시 호출(선택). 호출 안 해도 SIM_TIME 으로 자동 감지된다."""
    global _prev_x, _prev_sim_time
    _prev_x = None
    _prev_sim_time = None


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
    """종료 ±win/loss + HP 차분 damage + 거리·조준 포텐셜 shaping(차분)."""
    global _prev_x, _prev_sim_time
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])

    # [보조] 양측 생존(HP>0) 중에만: (상대 HP감소 - 내 HP감소) * damage_scale.
    # target_damage / ownship_damage = 이번 step 에 각 기체가 입은 damage(=HP 감소량).
    r_damage = 0.0
    if own_hp > 0.0 and tgt_hp > 0.0:
        own_w = float(reward_config.get("own_damage_weight", 0.5))   # 스케줄로 0.5→1.0
        r_damage = (float(target_damage) * 1.0 - float(ownship_damage) * own_w) * float(
            reward_config["damage_scale"]
        )

    # [보조] 상황 포텐셜 shaping: 거리(ft)/조준(A1,A2)을 합친 포텐셜 x 의 step 차분.
    # SIM_TIME 으로 에피소드 경계 감지(거꾸로 가거나 처음이면 새 episode → 보상 0).
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    new_episode = _prev_x is None or cur_sim_time <= _prev_sim_time
    shaping_scale = float(reward_config["shaping_reward_scale"])
    r_shaping = 0.0
    if shaping_scale != 0.0:
        # _get_distance 는 meter → ft 로 환산. A1/A2 는 3D ATA(proj=False, 0~180).
        dist_ft = float(geo_info._get_distance(ownship_state, target_state)) / _FT_TO_M
        a1 = abs(float(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
        a2 = abs(float(
            geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
        # StateIndex.ALT 는 meter → ft 로 환산(고도 안전 항 입력).
        own_alt_ft = float(ownship_state[StateIndex.ALT]) / _FT_TO_M
        cur_x = _shaping_potential(dist_ft, a1, a2, own_alt_ft)
        if not new_episode:
            r_shaping = (cur_x - _prev_x) * shaping_scale
        _prev_x = cur_x
    _prev_sim_time = cur_sim_time

    # [종료] 3분리: (1) HP 승/패  (2) 내 고도 하락 -20  (3) 상대 고도 하락 +5
    r_terminal = 0.0
    if terminated:
        # (1) HP 로 승부가 난 경우
        if tgt_hp <= 0.0:
            r_terminal += float(reward_config["win_reward"])
        if own_hp <= 0.0:
            r_terminal += float(reward_config["loss_reward"])
        # (2) 내 고도가 최소고도 이하로 떨어져 종료 → -20
        if end_condition == _OWNSHIP_ALT_END:
            r_terminal += float(reward_config["ownship_alt_reward"])
        # (3) 상대 고도가 최소고도 이하로 떨어져 종료 → +5
        if end_condition == _TARGET_ALT_END:
            r_terminal += float(reward_config["target_alt_reward"])

    total = r_damage + r_shaping + r_terminal
    return float(total), {"damage": r_damage, "shaping": r_shaping,
                          "terminal": r_terminal}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "reset_distance_tracker"]
