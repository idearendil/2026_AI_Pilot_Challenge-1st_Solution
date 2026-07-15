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
  - build_observation  : 위 재구성 값으로 34-D 관측 생성 (학습·검증·제출 동일 함수)
                         각도는 sin/cos 분리, delta 는 signed-log, 속도는 스칼라+3축(NED).
                         3축 속도는 raw 성분(학습=body/추론=world 프레임 불일치) 대신
                         속력+자세로 NED 속도를 재구성해 train/test 를 일치시킨다.
                         damage 조준 보강: 상대속도(NED)+명시거리+클로저레이트+aim_sharp.

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

# WEZ 콘 반각(도) — tier1/2/3 = 대결 서버 실제값으로 고정(1/2/3°). env.update_damage 와
# 재구성기(StateReconstructor)가 둘 다 damage_rate 를 통해 이 값을 읽는다. (커리큘럼 없음.)
_WEZ_CONE_DEG = (1.0, 2.0, 3.0)

OBSERVATION_MODE = "claude34r"   # r = reconstruction-aware
OBSERVATION_SIZE = 34
# 대부분 feature 는 [-1,1] 이지만 delta_n/e/d 는 sign(x)*ln(|x|+1) (정규화 안 함)이라
# |x|<=22025 까지 [-10,10] 안에 들어온다. 어차피 downstream RunningMeanStd 가 다시
# 정규화하므로 box 경계는 학습에 영향 없음 (clip 안 함). 넉넉히 ±10 으로 둔다.
OBSERVATION_LOW = -10.0
OBSERVATION_HIGH = 10.0

# 추론 환경의 고도 부호 규약. 학습은 NED(D=아래 양수)라 고도=-state[2].
# 실제 서버가 z를 "고도(위 양수)"로 주면 +1 로 바꾸세요(라이브 서버에서 확인 필요).
ALT_SIGN = -1.0


# ── damage 공식 (대결 서버 규칙) ──────────────────────────────────────────────

def damage_rate(r_ft: float, theta_deg: float, t_sec: float) -> float:
    """공격자 기준 damage '율'(per second). r=거리[ft], theta=ATA(공격자→피격자)[deg],
    t_sec=episode 경과 시간[s]. 시간 게이팅: tier2 는 100s, tier3 는 150s 부터 활성화.
    콘 반각은 _WEZ_CONE_DEG(1/2/3° 고정).
    실제 HP 감소는 advance() 에서 rate * DT_PER_STEP 으로 적분한다."""
    a = abs(theta_deg)
    c1, c2, c3 = _WEZ_CONE_DEG
    if 500.0 <= r_ft <= 3000.0 and a < c1:
        return 1.0 * (3000.0 - r_ft) / 2500.0
    if t_sec >= TIER2_START_SEC and 500.0 <= r_ft <= 3500.0 and a < c2:
        return 0.3 * (3500.0 - r_ft) / 3000.0
    if t_sec >= TIER3_START_SEC and 500.0 <= r_ft <= 4000.0 and a < c3:
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


# ── feature 변환 헬퍼 ─────────────────────────────────────────────────────────

def _sincos(angle_deg: float) -> tuple[float, float]:
    """각도(deg) → (sin, cos). 순환성(예: yaw 359°≈1°)을 모델이 학습하도록 분리한다.
    pitch/los_el(-90~90)처럼 순환하지 않는 각도도 sin 은 단조·cos 은 크기로 유효한 인코딩."""
    r = np.radians(float(angle_deg))
    return float(np.sin(r)), float(np.cos(r))


def _signed_log(x: float) -> float:
    """sign(x)*ln(|x|+1). 부호 보존 + 큰 값 압축(로그 스케일). 정규화하지 않는다."""
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1.0))


def _ned_velocity(speed: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """속력 + 자세(pitch/yaw)로 NED 속도 벡터를 재구성한다(기수 정렬 가정, AoA/sideslip≈0).
    raw body 속도 성분(학습)과 world 속도(추론)의 프레임 불일치를 피하려고 norm+자세로 복원."""
    p = np.radians(float(pitch_deg))
    y = np.radians(float(yaw_deg))
    cp = np.cos(p)
    return np.array([speed * cp * np.cos(y),    # North
                     speed * cp * np.sin(y),    # East
                     -speed * np.sin(p)],       # Down (상승=음수)
                    dtype=np.float64)


# ── 관측 벡터 (읽기 전용) ─────────────────────────────────────────────────────

def build_observation(ownship_state, target_state, geo_info, wez_config=None,
                      reconstructor=None) -> np.ndarray:
    """27-D 관측. 학습/검증/제출에서 동일하게 동작하도록 재구성값을 사용한다.

    추론에서 0이 되는 항목(속도·고도·HP·WEZ)을 위치·자세·속도와 누적 HP로 복원하므로
    모든 feature 가 학습과 추론에서 유효하다. reconstructor 를 주면 그 HP/damage 를
    쓰고(self-play 상대처럼 별도 관점일 때), None 이면 전역 싱글톤 _RECON 을 쓴다.

    각도 feature(roll/pitch/yaw/ata/aa/los_az/los_el)는 (sin, cos) 두 값으로 분리해
    순환성을 보존한다. delta_n/e/d 는 sign(x)*ln(|x|+1) (정규화 없음). 속도는 스칼라
    속력 + 3축 속도(NED, 속력·자세로 재구성) 를 모두 제공한다. damage 조준 보강용으로
    상대 속도(속력+NED 3축), 명시적 슬랜트 레인지, 클로저 레이트, 고분해능 조준점수(aim_sharp)도
    포함한다(총 34-D).
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
    dmg_taken = rec.last_dmg_taken   # 내가 받는 damage rate [0,~1]

    # 0~5: 내 자세 각도(roll/pitch/yaw) → 각 (sin, cos)
    obs[0], obs[1] = _sincos(ownship_state[StateIndex.ROLL])
    obs[2], obs[3] = _sincos(ownship_state[StateIndex.PITCH])
    obs[4], obs[5] = _sincos(ownship_state[StateIndex.YAW])
    # 6: 스칼라 속력(TAS, 재구성), 7~9: 3축 속도(NED, 재구성)
    # 주의: 학습 state[6:9]는 body 프레임(u,v,w)인데 추론(plane_info.velocity)은 world/NED
    # 프레임이라 raw 성분은 train/test 가 불일치한다(스칼라 속력=norm 만 프레임 무관해 일치).
    # 그래서 3축 속도는 속력 + 자세(pitch/yaw)로 NED 속도를 재구성한다(기수 정렬 가정
    # =AoA/sideslip≈0). 속력·자세 모두 두 경로에서 동일하므로 결과가 완전히 일치한다.
    own_vel = _ned_velocity(speed, ownship_state[StateIndex.PITCH], ownship_state[StateIndex.YAW])
    obs[6] = normalize(speed, 0.0, 600.0)
    obs[7] = normalize(float(own_vel[0]), -600.0, 600.0)
    obs[8] = normalize(float(own_vel[1]), -600.0, 600.0)
    obs[9] = normalize(float(own_vel[2]), -600.0, 600.0)
    # 10: 고도(재구성, 1000m 미만은 게임 종료라 1000~15000 정규화), 11: 내 HP(재구성)
    obs[10] = normalize(altitude, 1000.0, 15000.0)
    obs[11] = normalize(hp_own, 0.0, 1.0)
    # 12~14: 상대 위치 Δ → sign(x)*ln(|x|+1) (정규화 없음)
    obs[12] = _signed_log(delta[0])
    obs[13] = _signed_log(delta[1])
    obs[14] = _signed_log(delta[2])
    # 15~22: 기하 각도(ata/aa/los_az/los_el) → 각 (sin, cos)
    obs[15], obs[16] = _sincos(ata)
    obs[17], obs[18] = _sincos(aa)
    obs[19], obs[20] = _sincos(az)
    obs[21], obs[22] = _sincos(el)
    # 23: 표적 HP(재구성)
    obs[23] = normalize(hp_tgt, 0.0, 1.0)
    # 24: 내가 표적에 가하는 damage rate, 25: 내가 받는 damage rate (둘 다 초당 [0,1] → [-1,1])
    obs[24] = float(np.clip(2.0 * dmg_dealt - 1.0, -1.0, 1.0))
    obs[25] = float(np.clip(2.0 * dmg_taken - 1.0, -1.0, 1.0))
    # 26: 추격 점수 (원시)
    ata_factor = max(0.0, 1.0 - abs(float(ata)) / 30.0)
    range_factor = max(0.0, 1.0 - distance / 3000.0)
    obs[26] = 2.0 * (ata_factor * range_factor) - 1.0

    # ── damage 조준 보강 feature (1~4) ────────────────────────────────────────
    # ①상대 속도: 상대 속력 + NED 3축(상대 속력+자세로 재구성, train/test 일치). lead 조준용.
    tgt_speed = reconstruct_speed(target_state)
    tgt_vel = _ned_velocity(tgt_speed, target_state[StateIndex.PITCH], target_state[StateIndex.YAW])
    obs[27] = normalize(tgt_speed, 0.0, 600.0)
    obs[28] = normalize(float(tgt_vel[0]), -600.0, 600.0)
    obs[29] = normalize(float(tgt_vel[1]), -600.0, 600.0)
    obs[30] = normalize(float(tgt_vel[2]), -600.0, 600.0)
    # ②명시적 슬랜트 레인지(WEZ 스케일). 0~2000m 로 정규화해 WEZ 밴드(~152~914m)를 잘 분해.
    obs[31] = normalize(float(distance), 0.0, 2000.0)
    # ③클로저 레이트: LOS 단위벡터에 (내 속도−상대 속도) 투영. 양수=접근, 음수=이탈.
    dist_norm = float(np.linalg.norm(delta))
    if dist_norm > 1e-6:
        los_unit = delta / dist_norm
        closure = float(np.dot(own_vel - tgt_vel, los_unit))
    else:
        closure = 0.0
    obs[32] = normalize(closure, -1000.0, 1000.0)
    # ④고분해능 조준 점수: exp(-(ATA/σ)²), σ=3° → 1~3° 콘에서 또렷한 그래디언트. [-1,1] 매핑.
    aim_sharp = float(np.exp(-((float(ata) / 3.0) ** 2)))
    obs[33] = 2.0 * aim_sharp - 1.0
    return obs


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "roll_sin", "roll_cos", "pitch_sin", "pitch_cos", "yaw_sin", "yaw_cos",
            "speed(TAS,재구성)", "vel_n(NED,재구성)", "vel_e(NED,재구성)", "vel_d(NED,재구성)",
            "altitude(재구성,1000~15000)", "own_hp(재구성)",
            "delta_n(signed_log)", "delta_e(signed_log)", "delta_d(signed_log)",
            "ata_sin", "ata_cos", "aa_sin", "aa_cos",
            "los_az_sin", "los_az_cos", "los_el_sin", "los_el_cos",
            "target_hp(재구성)", "damage_rate_dealt(재구성)", "damage_rate_taken(재구성)",
            "pursuit_score",
            "tgt_speed(재구성)", "tgt_vel_n(NED,재구성)", "tgt_vel_e(NED,재구성)",
            "tgt_vel_d(NED,재구성)", "slant_range(0~2000m)", "closure_rate(접근+)",
            "aim_sharp(exp,-σ3°)",
        ],
        "description": "claude_code 34-D, 각도 sin/cos, delta signed-log, 속도 스칼라+3축, "
                       "상대속도+명시거리+클로저레이트+고분해능 조준점수 추가(damage 조준 보강), "
                       "제출환경 미제공 항목(속도/고도/HP/WEZ)을 재구성.",
    }


__all__ = [
    "OBSERVATION_MODE", "OBSERVATION_SIZE", "OBSERVATION_LOW", "OBSERVATION_HIGH",
    "build_observation", "describe_observation",
    "StateReconstructor", "damage_rate",
    "reconstruct_altitude", "reconstruct_speed",
    "get_reconstructor", "reset_reconstructor", "advance_reconstructor",
]
