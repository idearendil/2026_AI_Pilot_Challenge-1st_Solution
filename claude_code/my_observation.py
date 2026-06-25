# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 관측 벡터 + 제출환경 state 재구성.

핵심 아이디어
-------------
대결 서버 추론에서는 PlaneInfo(위치·자세·속도, 즉 state 인덱스 0~8)만 들어오고
KCAS(12)·ALT(44)·HEALTH(45)·WEZ 등은 0이 된다. 하지만:

  - 고도 = -D (= -state[2])                            ← 위치로 재구성
  - 속도(TAS) = ||(u,v,w)|| (= ||state[6:9]||)          ← 속도로 재구성
  - 거리/ATA/AA/LOS = geo_info 로 계산                  ← 위치·자세로 재구성
  - HP = 매 step 예상 damage 를 누적해서 추정           ← damage 공식으로 재구성

따라서 위치·방향·속도만 있으면 관측에 필요한 값을 거의 다 복원할 수 있다.

damage 공식 (대결 서버 규칙, r=거리[ft], theta=ATA[deg], |theta|=기수 이탈각)
  tier1 (항상)     : 500<=r<=3000 & |theta|<1 : rate = 1.0*(3000-r)/2500
  tier2 (100s 부터): 500<=r<=3500 & |theta|<2 : rate = 0.3*(3500-r)/3000
  tier3 (150s 부터): 500<=r<=4000 & |theta|<3 : rate = 0.1*(4000-r)/3500
  else 0
**시간 게이팅**: episode 경과 100s 부터 tier2, 150s 부터 tier3 가 추가로 활성화된다.
공식 값은 '초당 damage rate' 이며, 매 step HP -= rate * DT_PER_STEP 로 누적한다.
A의 공격범위 안에 B가 들어오면 B의 HP가 깎인다(=B가 데미지를 받음). 즉 내 HP 감소는
"상대 기수 기준 내가 그 cone 안에 있는가"(theta=ATA(상대→나))로 계산한다.

DT_PER_STEP(=0.1s) 추출 근거: 학습 env 의 update_damage 는 계수 * env._delta_t(=1/60)
를 매 sim sub-step 적용하고, RL-step 은 step_ratio(=6) sub-step 이므로 RL-step 당
시간 = 6/60 = 0.1s. (실측으로 env._delta_t=1/60, step_ratio=6 확인.) 학습 env 자체는
tier1 만(시간 게이팅 없음) 적용함도 코드/실측으로 확인했고, 시간 게이팅은 대결 서버 규칙.

구조
----
  - StateReconstructor : HP 누적(상태 보존) + 즉시 재구성 값 제공
  - 모듈 싱글톤 _RECON  : reset_reconstructor()/advance_reconstructor() 로 RL-step 당
    1회만 advance (학습=ppo.py, 추론=MLPActionProvider). build_observation 은 읽기만.
  - build_observation  : 위 재구성 값으로 16-D 관측 생성 (학습·검증·제출 동일 함수)

검증: claude_code/verify_reconstruction.py 가 학습 환경에서 재구성값 vs 실제 env
state 를 비교한다 (고도/속도/기하는 일치, HP 는 공식 차이 주석 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dogfight.envs.observation import normalize
from dogfight.sim.state_schema import StateIndex
from GeoMathUtil import GeometryInfo

METER_TO_FEET = 3.28084

# 학습 env 에서 추출한 시간 상수. env._delta_t = 1/SIM_HZ, RL action 1회 = STEP_RATIO frame.
SIM_HZ = 60
STEP_RATIO = 6
DT_PER_STEP = STEP_RATIO / SIM_HZ      # 0.1 s — RL-step 당 경과 시간(= damage 누적 dt)

# 대결 서버 damage tier 시간 게이팅 (episode 경과 시간 기준). 학습 env 엔 없음.
TIER2_START_SEC = 100.0
TIER3_START_SEC = 150.0

OBSERVATION_MODE = "claude16r"   # r = reconstruction-aware
OBSERVATION_SIZE = 16
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0

# 추론 환경의 고도 부호 규약. 학습은 NED(D=아래 양수)라 고도=-state[2].
# 실제 서버가 z를 "고도(위 양수)"로 주면 +1 로 바꾸세요(라이브 서버에서 확인 필요).
ALT_SIGN = -1.0


# ── damage 공식 (대결 서버 규칙) ──────────────────────────────────────────────

def damage_rate(r_ft: float, theta_deg: float, t_sec: float) -> float:
    """공격자 기준 damage '율'(per second). r=거리[ft], theta=ATA(공격자→피격자)[deg],
    t_sec=episode 경과 시간[s]. 시간 게이팅: tier2 는 100s, tier3 는 150s 부터 활성화.
    실제 HP 감소는 advance() 에서 rate * DT_PER_STEP 으로 적분한다."""
    a = abs(theta_deg)
    if 500.0 <= r_ft <= 3000.0 and a < 1.0:
        return 1.0 * (3000.0 - r_ft) / 2500.0
    if t_sec >= TIER2_START_SEC and 500.0 <= r_ft <= 3500.0 and a < 2.0:
        return 0.3 * (3500.0 - r_ft) / 3000.0
    if t_sec >= TIER3_START_SEC and 500.0 <= r_ft <= 4000.0 and a < 3.0:
        return 0.1 * (4000.0 - r_ft) / 3500.0
    return 0.0


# ── 무상태 재구성 헬퍼 ────────────────────────────────────────────────────────

def reconstruct_altitude(state) -> float:
    """고도[m] = ALT_SIGN * D. 학습 env 에서 실제 ALT(44)와 일치."""
    return ALT_SIGN * float(state[StateIndex.D])


def reconstruct_speed(state) -> float:
    """속도(TAS)[m/s] = ||(u,v,w)|| = ||state[6:9]||. 학습 env 의 KTAS(27)와 일치."""
    return float(np.linalg.norm(np.asarray(state[6:9], dtype=np.float64)))


# ── HP 누적 재구성기 (상태 보존) ──────────────────────────────────────────────

class StateReconstructor:
    """RL-step 단위로 damage 를 누적해 양측 HP 를 추정한다.

    advance() 는 한 RL-step(=STEP_RATIO 프레임) 당 정확히 한 번만 호출해야 한다.
    build_observation 은 누적값을 읽기만 하고 advance 하지 않는다(빈도 일관성).
    """

    def __init__(self, dt_per_step: float = DT_PER_STEP):
        self.dt = float(dt_per_step)
        self._geo = GeometryInfo()
        self.reset()

    def reset(self) -> None:
        self.hp_own = 1.0
        self.hp_tgt = 1.0
        self.t_sec = 0.0            # episode 경과 시간 (시간 게이팅용)
        self.last_dmg_dealt = 0.0   # 이번 step 내가 표적에 가하는 damage rate
        self.last_dmg_taken = 0.0   # 이번 step 내가 받는 damage rate
        self.last_r_ft = 0.0
        self.last_ata_own = 180.0
        self.last_ata_tgt = 180.0

    def advance(self, own_state, tgt_state) -> None:
        own = np.asarray(own_state, dtype=np.float64)
        tgt = np.asarray(tgt_state, dtype=np.float64)
        r_ft = self._geo._get_distance(own, tgt) * METER_TO_FEET
        ata_own = self._geo._get_antenna_train_angle(own, tgt, False)  # 내 기수→표적
        ata_tgt = self._geo._get_antenna_train_angle(tgt, own, False)  # 표적 기수→나

        rate_dealt = damage_rate(r_ft, ata_own, self.t_sec)   # 초당 rate (시간게이팅 반영)
        rate_taken = damage_rate(r_ft, ata_tgt, self.t_sec)
        # HP -= rate * dt (env 와 동일한 시간 적분: env 는 계수*delta_t 를 sub-step 마다).
        self.hp_tgt = max(0.0, self.hp_tgt - rate_dealt * self.dt)
        self.hp_own = max(0.0, self.hp_own - rate_taken * self.dt)

        self.last_dmg_dealt = rate_dealt
        self.last_dmg_taken = rate_taken
        self.last_r_ft = r_ft
        self.last_ata_own = ata_own
        self.last_ata_tgt = ata_tgt
        self.t_sec += self.dt   # 다음 step 을 위한 시간 진행


# ── 모듈 싱글톤 (학습/추론이 공유) ────────────────────────────────────────────

_RECON = StateReconstructor()


def get_reconstructor() -> StateReconstructor:
    return _RECON


def reset_reconstructor() -> None:
    """에피소드 시작 시 호출 (HP=1 로 초기화)."""
    _RECON.reset()


def advance_reconstructor(own_state, tgt_state) -> None:
    """RL-step 당 1회 호출 (HP 누적). 학습=ppo.py, 추론=MLPActionProvider 에서 호출."""
    _RECON.advance(own_state, tgt_state)


# ── 관측 벡터 (읽기 전용) ─────────────────────────────────────────────────────

def build_observation(ownship_state, target_state, geo_info, wez_config=None,
                      reconstructor=None) -> np.ndarray:
    """16-D 관측. 학습/검증/제출에서 동일하게 동작하도록 재구성값을 사용한다.

    추론에서 0이 되는 항목(속도·고도·HP·WEZ)을 위치·자세·속도와 누적 HP로 복원하므로
    16개 feature 가 학습과 추론에서 모두 유효하다. reconstructor 를 주면 그 HP/damage 를
    쓰고(self-play 상대처럼 별도 관점일 때), None 이면 전역 싱글톤 _RECON 을 쓴다.
    """
    rec = reconstructor if reconstructor is not None else _RECON
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    delta = np.asarray(target_state[:3], dtype=np.float64) - np.asarray(ownship_state[:3], dtype=np.float64)
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    # 재구성값 (무상태)
    speed = reconstruct_speed(ownship_state)
    altitude = reconstruct_altitude(ownship_state)
    # 재구성값 (상태 보존 HP — 해당 reconstructor 에서 읽기만)
    hp_own = rec.hp_own
    hp_tgt = rec.hp_tgt
    dmg_dealt = rec.last_dmg_dealt   # 내가 표적에 가하는 damage rate [0,~1]

    # 0~2: 자세 (원시, 추론에서도 유효)
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]), -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]), -90.0, 90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]), 0.0, 360.0)
    # 3~5: 속도/고도/내 HP (재구성)
    obs[3] = normalize(speed, 0.0, 600.0)
    obs[4] = normalize(altitude, 0.0, 15000.0)
    obs[5] = normalize(hp_own, 0.0, 1.0)
    # 6~8: 상대 위치 Δ (원시)
    obs[6] = normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = normalize(float(delta[2]), -8000.0, 8000.0)
    # 9~12: 기하 (원시 계산, 추론에서도 유효)
    obs[9] = normalize(float(ata), -180.0, 180.0)
    obs[10] = normalize(float(aa), -180.0, 180.0)
    obs[11] = normalize(float(az), -180.0, 180.0)
    obs[12] = normalize(float(el), -90.0, 90.0)
    # 13: 표적 HP (재구성)
    obs[13] = normalize(hp_tgt, 0.0, 1.0)
    # 14: 순간 damage rate(내가 표적에 가하는, 초당 [0,1]) — 기존 binary WEZ 대체, 재구성
    obs[14] = float(np.clip(2.0 * dmg_dealt - 1.0, -1.0, 1.0))
    # 15: 추격 점수 (원시)
    ata_factor = max(0.0, 1.0 - abs(float(ata)) / 30.0)
    range_factor = max(0.0, 1.0 - distance / 3000.0)
    obs[15] = 2.0 * (ata_factor * range_factor) - 1.0
    return obs


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "roll", "pitch", "yaw", "speed(TAS,재구성)", "altitude(재구성)",
            "own_hp(재구성)", "delta_n", "delta_e", "delta_d", "ata", "aa",
            "los_az", "los_el", "target_hp(재구성)", "damage_rate_dealt(재구성)",
            "pursuit_score",
        ],
        "description": "claude_code 16-D, 제출환경 미제공 항목(속도/고도/HP/WEZ)을 재구성.",
    }


__all__ = [
    "OBSERVATION_MODE", "OBSERVATION_SIZE", "OBSERVATION_LOW", "OBSERVATION_HIGH",
    "build_observation", "describe_observation",
    "StateReconstructor", "damage_rate",
    "reconstruct_altitude", "reconstruct_speed",
    "get_reconstructor", "reset_reconstructor", "advance_reconstructor",
]
