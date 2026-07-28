# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 보상 함수.

규칙(= 종료 보상 + potential-based shaping(PBRS)):
  - [종료·고도] 상대 고도가 최소고도(=300m≈984ft) 이하로 떨어져 종료(내가 이김) → +alt_win_reward(+10)
                내 고도가 최소고도 이하로 떨어져 종료(상대가 이김)             → +alt_loss_reward(-10)
  - [종료·HP/시간] 한쪽 HP<=0 격추 또는 max step 도달로 종료
                → (내 HP - 상대 HP) * hp_terminal_scale(10).  (풀피=1.0, 격추=0.0)
                  격추든 max step 이든 항상 HP 차에 비례한다. 격추 시 진 쪽 HP=0 이므로
                  이긴 쪽은 자기 잔여 HP×10(만피면 +10, 데미지 입었으면 그만큼 감소),
                  진 쪽은 -상대HP×10 을 받는다(고정 ±10 이 아님).
  - [보조] potential-based shaping:  reward += γ·Φ(s') - Φ(s).   (γ = 학습 gamma, train.py 주입)
           terminal state 에서는 Φ(s')≡0 으로 강제 → 마지막 transition 이 -Φ(s) 로 텔레스코핑되어
           potential 차분과 종료 보상이 자연스럽게 이어진다. 이 형태는 정책 불변(policy-invariant).
           총 포텐셜 Φ = 아래 3개의 합:
             (1) 거리·조준 포텐셜 :  _shaping_potential(distance[ft], A1, A2) * shaping_reward_scale
                   A1 = 아군→상대 LOS(|ATA|), A2 = 상대→아군 LOS(|ATA|). 가깝고(WEZ 안) 내가 잘
                   조준하고 상대는 못 조준할수록 커진다. 원값 ~1e6 라 계수 0.00005 로 축약.
             (2) HP(damage) 포텐셜 :  (내 HP - 상대 HP) * damage_potential_scale(10)   → [-10, +10]
             (3) 고도 포텐셜       :  (min(내고도, cap) - min(상대고도, cap)) / div
                   내 고도·상대 고도는 StateIndex.ALT(미터). cap=4000m, div=300m(=최소고도).
                   양쪽 다 cap 이상이면 0(고고도에선 고도차 무시), 저고도에서 고도 우위를 보상.
           _shaping_potential 정의(연속함수, distance 는 ft, A 는 deg 0~180):
             distance ∈ [0, 500):     (d+14000)*(90-A1)/90*2.5 - (d+14000)*(90-A2)/90*2.5 + 999500
             distance ∈ [500, 15000): (15000-d) + (15000-d)*(90-A1)/90*2.5 - (15000-d)*(90-A2)/90*2.5 + 985000
             distance ∈ [15000, ∞):   1000000 - d
             (경계 500ft·15000ft 에서 연속. 값은 대략 ~1e6 근처.)

claude_code/train.py 는 기본적으로 이 모듈을 사용한다(끄려면 --reward-module "").

계약: MY_REWARD_CONFIG(dict) + compute_reward(...) -> (total: float, components: dict).

PBRS 는 직전 step 의 총 포텐셜값 Φ(s) 를 모듈 상태로 추적해야 한다(reward_fn 은 stateless
계약). 에피소드 경계는 SIM_TIME 으로 감지한다(NaN-crash 등 compute_reward 를 건너뛰는 종료
경로도 안전하게 처리 — reset 후 SIM_TIME 이 되감기면 새 episode 로 보고 shaping=0). 보상은
학습/평가 env 에서만 계산되고 추론(제출)에서는 호출되지 않으므로 SIM_TIME 이 항상 채워져 있어
안전하다. (병렬 워커는 프로세스별 별도 모듈이라 상태 충돌 없음; 단일 프로세스 eval 은 끝나고
rollout env 를 reset 하므로 새 episode 로 자동 감지된다.)
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
    # ── 종료 보상 ─────────────────────────────────────────────────────────
    "alt_win_reward": 10.0,      # 상대 고도 하락으로 종료(내가 이김)
    "alt_loss_reward": -10.0,    # 내 고도 하락으로 종료(상대가 이김)
    "hp_terminal_scale": 10.0,   # 격추/max-step 종료: (내 HP - 상대 HP) * 이 값 (풀피=1.0, 격추=0.0)
    # ── potential-based shaping(PBRS) 계수: reward += γ·Φ(s') - Φ(s) ──────
    # (1) 거리·조준 포텐셜에 곱하는 계수. 원값 ~1e6 라 0.00005 로 축약(예전과 동일).
    "shaping_reward_scale": 0.00005,
    # (2) HP 포텐셜 계수: Φ_hp = (내 HP - 상대 HP) * 이 값 → [-10, +10].
    "damage_potential_scale": 10.0,
    # (3) 고도 포텐셜: Φ_alt = (min(내고도, cap) - min(상대고도, cap)) / div. 고도는 미터(ALT).
    "altitude_potential_cap": 4000.0,   # cap[m]: 이 이상 고도는 동일 취급(고고도 고도차 무시)
    "altitude_potential_div": 300.0,    # div[m]: 최소고도(=300m)로 정규화
    # PBRS 의 γ. 학습 gamma 와 일치해야 정책 불변이 성립한다. train.py 가 args.gamma 로 덮어쓴다.
    "gamma": 0.99,
}
# 주의: 아래 compute_reward 는 이 dict 의 키를 **직접 인덱싱**한다(.get 폴백 없음).
# 계수를 바꾸려면 반드시 이 dict(또는 train.py 의 --*-reward-scale / gamma 오버라이드)를 고칠 것.
# 예전에는 .get(key, 폴백) 형태라 폴백만 고치면 아무 효과가 없는 함정이 있었다.

# 종료 사유 문자열 (src/dogfight/envs/termination.py 기준). 고도 종료는 env 의
# min_altitude(기본 300m≈984ft) 에서 발생하므로 "1000ft 이하 종료"와 사실상 동일.
_TARGET_ALT_END = "target altitude below min"
_OWNSHIP_ALT_END = "ownship altitude below min"

# PBRS 직전-step 상태 (모듈 싱글톤). 에피소드 경계는 SIM_TIME 으로 감지.
_prev_phi: float | None = None     # 직전 step 의 총 포텐셜 Φ(s)
_prev_sim_time: float | None = None


def _shaping_potential(distance_ft: float, a1_deg: float, a2_deg: float) -> float:
    """거리(ft)/조준(A1,A2 deg, 0~180)을 하나로 합친 상황 포텐셜.

    A1 = 아군→상대 LOS(|ATA|), A2 = 상대→아군 LOS(|ATA|). 값이 클수록 유리.
    경계 500ft·15000ft 에서 연속. (90-A) 항은 A>90(등 뒤) 이면 음수가 되어 자연스럽게
    페널티로 작동하므로 clamp 하지 않는다.
    """
    if distance_ft <= 500.0:
        base = distance_ft + 14000.0
        return (base * (90.0 - a1_deg) / 90.0 * 2.5
                - base * (90.0 - a2_deg) / 90.0 * 2.5
                + 999500.0)
    if distance_ft <= 15000.0:
        base = 15000.0 - distance_ft
        return (base
                + base * (90.0 - a1_deg) / 90.0 * 2.5
                - base * (90.0 - a2_deg) / 90.0 * 2.5
                + 985000.0)
    return 1000000.0 - distance_ft


def reset_distance_tracker() -> None:
    """에피소드 시작 시 호출(선택). 호출 안 해도 SIM_TIME 으로 자동 감지된다."""
    global _prev_phi, _prev_sim_time
    _prev_phi = None
    _prev_sim_time = None


def _total_potential(reward_config, geo_info, ownship_state, target_state,
                     own_hp: float, tgt_hp: float) -> float:
    """총 포텐셜 Φ(s) = 거리·조준 포텐셜(×계수) + HP 포텐셜 + 고도 포텐셜."""
    # (1) 거리·조준 포텐셜. _get_distance 는 meter → ft 로 환산. A1/A2 는 3D ATA(proj=False, 0~180).
    dist_ft = float(geo_info._get_distance(ownship_state, target_state)) / _FT_TO_M
    a1 = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    a2 = abs(float(geo_info._get_antenna_train_angle(target_state, ownship_state, False)))
    phi_range = _shaping_potential(dist_ft, a1, a2) * float(reward_config["shaping_reward_scale"])
    # (2) HP(damage) 포텐셜: (내 HP - 상대 HP) * scale.
    phi_hp = (own_hp - tgt_hp) * float(reward_config["damage_potential_scale"])
    # (3) 고도 포텐셜: (min(내고도, cap) - min(상대고도, cap)) / div. 고도는 ALT(미터).
    cap = float(reward_config["altitude_potential_cap"])
    div = float(reward_config["altitude_potential_div"])
    alt_own = float(ownship_state[StateIndex.ALT])
    alt_tgt = float(target_state[StateIndex.ALT])
    phi_alt = (min(alt_own, cap) - min(alt_tgt, cap)) / div
    return phi_range + phi_hp + phi_alt


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
    """종료 보상(고도 ±10 / HP·시간 (내HP-상대HP)×10) + PBRS shaping(γ·Φ(s')-Φ(s))."""
    global _prev_phi, _prev_sim_time
    own_hp = float(ownship_state[StateIndex.HEALTH])
    tgt_hp = float(target_state[StateIndex.HEALTH])
    done = bool(terminated or truncated)

    # [PBRS] reward += γ·Φ(s') - Φ(s).  terminal state 에서는 Φ(s')≡0 으로 강제한다.
    # SIM_TIME 으로 에피소드 경계 감지(되감기거나 처음이면 새 episode → shaping 0).
    cur_sim_time = float(ownship_state[StateIndex.SIM_TIME])
    new_episode = _prev_phi is None or cur_sim_time <= _prev_sim_time
    gamma = float(reward_config["gamma"])
    phi_cur = 0.0 if done else _total_potential(
        reward_config, geo_info, ownship_state, target_state, own_hp, tgt_hp)
    r_shaping = 0.0
    if not new_episode:
        r_shaping = gamma * phi_cur - _prev_phi
    _prev_phi = phi_cur
    _prev_sim_time = cur_sim_time

    # [종료] (a) 고도 하락: 내가 이김 +10 / 상대가 이김 -10.
    #        (b) 그 외 종료(격추=HP<=0, max step 등): (내 HP - 상대 HP) * scale.
    r_terminal = 0.0
    if done:
        if end_condition == _TARGET_ALT_END:          # 상대 고도 하락 → 내가 이김
            r_terminal = float(reward_config["alt_win_reward"])
        elif end_condition == _OWNSHIP_ALT_END:        # 내 고도 하락 → 상대가 이김
            r_terminal = float(reward_config["alt_loss_reward"])
        else:                                          # 격추 or max step
            r_terminal = (own_hp - tgt_hp) * float(reward_config["hp_terminal_scale"])

    total = r_shaping + r_terminal
    return float(total), {"damage": 0.0, "shaping": r_shaping,
                          "terminal": r_terminal}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "reset_distance_tracker"]
