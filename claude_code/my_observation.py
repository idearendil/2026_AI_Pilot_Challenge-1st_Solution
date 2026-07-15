# -*- coding: utf-8 -*-
"""claude_code 관측 모듈.

이 파일은 학습, 평가, self-play, 제출 코드가 사용하는 외부 계약을 유지한다.

  - StateReconstructor가 episode 중 재구성 상태를 저장한다.
  - _RECON은 모듈 전역 singleton이다.
  - reset_reconstructor()/advance_reconstructor()는 RL step마다 한 번만 호출한다.
  - build_observation()은 reconstructor 값을 읽기만 한다.

state 입력 계약
---------------
교전 서버와 JSBSim 경로는 동일한 최소 9차원 운동 상태를 보낸다고 본다.

  state[0] = N, North 위치 [m]
  state[1] = E, East 위치 [m]
  state[2] = D, Down 위치 [m]
  state[3] = roll [deg]
  state[4] = pitch [deg]
  state[5] = yaw [deg]
  state[6] = body x 속도 u [m/s], 기수 전방 양수
  state[7] = body y 속도 v [m/s], 오른쪽 날개 방향 양수
  state[8] = body z 속도 w [m/s], 아래 방향 양수

좌표계:
  NED frame  : N+ 북쪽, E+ 동쪽, D+ 아래
  Body frame : x+ 기수 전방, y+ 오른쪽 날개, z+ 아래

중요: state[6:9]는 body-frame 속도 u/v/w로 직접 사용한다. 예전처럼 speed,
pitch, yaw로 NED 속도를 재구성하는 경로는 의도적으로 쓰지 않는다. NED 속도가
필요할 때만 rotation matrix로 body 속도를 NED로 변환한다.

damage 재구성
-------------
서버가 HP를 직접 주지 않아도 된다. StateReconstructor는 현재 거리, ATA, episode
시간을 사용해 같은 3-tier cone-damage rule을 적분한다. 하위 tier가 우선이다.
예를 들어 150초 이후라도 더 좁은 tier1 거리/각도 조건을 만족하면 tier1 damage를
먼저 적용한다.

ownship p/q/r 재구성
--------------------
ownship p/q/r만 추정한다. StateReconstructor.advance()에서 직전 자세와 현재 자세를
사용한다. 추정은 상대 rotation의 SO(3) log를 사용하므로 yaw가 179 deg -> -179 deg로
wrap되어도 인위적인 spike가 생기지 않는다. target p/q/r은 이 42차원 observation에
포함하지 않는다.
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

FEET_TO_METER = 0.3048
METER_TO_FEET = 3.28084
G = 9.80665

# RL action step 시간 간격. env._delta_t = 1/SIM_HZ이고, RL step 1회는
# STEP_RATIO개의 simulator frame을 진행한다.
SIM_HZ = 60
STEP_RATIO = 6
DT_PER_STEP = STEP_RATIO / SIM_HZ

MAX_SPEED = 600.0
BODY_VEL_MIN = -600.0
BODY_VEL_MAX = 600.0
REL_VEL_MIN = -600.0
REL_VEL_MAX = 600.0
MAX_RANGE_M = 2500.0
MAX_CLOSURE_SPEED = 1000.0
VERTICAL_SPEED_SCALE = 100.0
PQR_SCALE_RAD_S = 4.0
AOA_SCALE_DEG = 30.0
SIDESLIP_SCALE_DEG = 15.0
MIN_ALTITUDE_M = 300.0
ALTITUDE_DANGER_SCALE_M = 300.0
ENERGY_ADVANTAGE_SCALE_M = 5000.0
REL_POS_SCALE_M = 1000.0

TIER2_START_SEC = 100.0
TIER3_START_SEC = 150.0
TIER1_CONE_DEG = 1.0
TIER2_CONE_DEG = 2.0
TIER3_CONE_DEG = 3.0
MIN_DAMAGE_RANGE_FT = 500.0
TIER1_MAX_RANGE_FT = 3000.0
TIER2_MAX_RANGE_FT = 3500.0
TIER3_MAX_RANGE_FT = 4000.0
EPISODE_MAX_TIME_SEC = 200.0

OBSERVATION_MODE = "claude42r"
OBSERVATION_SIZE = 42

# ── 42-D observation feature 요약 (build_observation 참고) ────────────────────
# 모든 값은 대략 [-1,1] 범위로 scaling되며, 마지막에 전체 clip은 하지 않는다.
# 표현 기준: body = ownship body frame(x 기수전방/y 오른쪽날개/z 아래), NED = 월드.
#
#   idx  feature                     설명
#   ---  --------------------------  -----------------------------------------
#    0   gravity_body_x              중력벡터를 ownship body frame으로 표현(어디가
#    1   gravity_body_y              아래인가 → 자세를 절대 기준으로 인지). 단위벡터.
#    2   gravity_body_z
#    3   own_speed_norm              내 속력 |u,v,w| / 600
#    4   own_vel_body_x_norm         내 body 속도 u(기수 전방) / 600
#    5   own_vel_body_y_norm         내 body 속도 v(오른쪽 날개) / 600
#    6   own_vel_body_z_norm         내 body 속도 w(아래) / 600
#    7   own_p_est_tanh              내 롤레이트 p (자세 history의 SO(3) log 추정) tanh(/4)
#    8   own_q_est_tanh              내 피치레이트 q, 〃
#    9   own_r_est_tanh              내 요레이트 r, 〃
#   10   AoA_tanh                    받음각 arctan2(w,u) tanh(/30°)
#   11   sideslip_tanh               옆미끄럼각 arctan2(v,·) tanh(/15°)
#   12   altitude_margin_low_tanh    최소고도(300m) 마진 tanh. 0이면 정확히 hard-deck.
#   13   vertical_speed_norm         상승률(NED, 상승=+) / 100
#   14   own_hp_norm                 내 HP(재구성) [0,1]
#   15   target_speed_norm           표적 속력 / 600
#   16   target_hp_norm              표적 HP(재구성) [0,1]
#   17   hp_diff                     내 HP - 표적 HP  (우세 +, 열세 -)
#   18   energy_advantage_softsign   에너지고도차(고도+v²/2g) softsign(/5000m)
#   19   rel_pos_body_x_softsign     표적 상대위치를 body frame으로 → 앞/뒤 softsign(/1000m)
#   20   rel_pos_body_y_softsign     〃 좌/우
#   21   rel_pos_body_z_softsign     〃 위/아래
#   22   rel_vel_body_x_norm         표적 상대속도(body frame) 앞/뒤 / 600
#   23   rel_vel_body_y_norm         〃 좌/우
#   24   rel_vel_body_z_norm         〃 위/아래
#   25   slant_range_norm            표적까지 거리 / 2500m
#   26   closure_rate_norm           접근율(LOS 투영, 접근=+) / 1000
#   27   sin_ATA \                    ATA(내 기수→표적 이탈각) sin/cos
#   28   cos_ATA /
#   29   sin_AA  \                    AA(표적 기준 내 aspect angle) sin/cos
#   30   cos_AA  /
#   31   sin_LOS_azimuth \            LOS 방위각 sin/cos
#   32   cos_LOS_azimuth /
#   33   sin_LOS_elevation \          LOS 고각 sin/cos
#   34   cos_LOS_elevation /
#   35   aim_sharp                   내 조준 예리도 exp(-(ATA/3°)²) → [-1,1] (표적이 내 콘 안?)
#   36   aim_margin_active_tanh      현재 활성 tier 콘 기준 조준 여유 tanh (양수=콘 안)
#   37   enemy_aim_sharp             적의 조준 예리도 exp(-(적ATA/3°)²) → 피격 위험 인지
#   38   enemy_aim_margin_active_tanh 적 콘 기준 내 피격 여유 tanh (양수=적 콘 안=위험)
#   39   range_margin_near_tanh      활성 damage 밴드 near-edge 여유 tanh (너무 가까운가)
#   40   range_margin_far_active_tanh 활성 damage 밴드 far-edge 여유 tanh (너무 먼가)
#   41   time_norm                   episode 경과시간 / 200s (0s→-1, 200s→+1)
#
# 시간 게이팅(tier2 100s, tier3 150s)에 따라 35~40의 active envelope가 넓어진다.

# Gym/model metadata에 기록할 observation space bound다.
# 전체 observation을 마지막에 clip하는 용도로 사용하지 않는다.
OBSERVATION_LOW = -10.0
OBSERVATION_HIGH = 10.0


def damage_rate(r_ft: float, theta_deg: float, t_sec: float) -> float:
    """초당 cone-damage rate를 반환한다.

    r_ft는 feet 단위 slant range다. theta_deg는 degree 단위 ATA다.
    t_sec는 episode 경과 시간이다. 낮은 tier 조건을 먼저 검사하므로,
    tier2/tier3가 활성화된 뒤에도 tier1 조건을 만족하면 tier1이 우선한다.
    """
    r_ft = float(r_ft)
    a = abs(float(theta_deg))
    t_sec = float(t_sec)

    if MIN_DAMAGE_RANGE_FT <= r_ft <= TIER1_MAX_RANGE_FT and a < TIER1_CONE_DEG:
        return 1.0 * (TIER1_MAX_RANGE_FT - r_ft) / (
            TIER1_MAX_RANGE_FT - MIN_DAMAGE_RANGE_FT
        )
    if (
        t_sec >= TIER2_START_SEC
        and MIN_DAMAGE_RANGE_FT <= r_ft <= TIER2_MAX_RANGE_FT
        and a < TIER2_CONE_DEG
    ):
        return 0.3 * (TIER2_MAX_RANGE_FT - r_ft) / (
            TIER2_MAX_RANGE_FT - MIN_DAMAGE_RANGE_FT
        )
    if (
        t_sec >= TIER3_START_SEC
        and MIN_DAMAGE_RANGE_FT <= r_ft <= TIER3_MAX_RANGE_FT
        and a < TIER3_CONE_DEG
    ):
        return 0.1 * (TIER3_MAX_RANGE_FT - r_ft) / (
            TIER3_MAX_RANGE_FT - MIN_DAMAGE_RANGE_FT
        )
    return 0.0


def reconstruct_altitude(state) -> float:
    """NED D에서 고도[m]를 재구성한다. D는 아래 방향 양수라서 altitude = -D다."""
    return -float(state[StateIndex.D])


def reconstruct_speed(state) -> float:
    """body-frame 속도 u/v/w = state[6:9]에서 TAS와 유사한 속력[m/s]을 구한다."""
    return float(np.linalg.norm(np.asarray(state[6:9], dtype=np.float64)))


def _sincos(angle_deg: float) -> tuple[float, float]:
    r = np.radians(float(angle_deg))
    return float(np.sin(r)), float(np.cos(r))


def _tanh_scale(x: float, scale: float) -> float:
    return float(np.tanh(float(x) / float(scale)))


def _softsign_scale(x: float, scale: float) -> float:
    x = float(x)
    return float(x / (abs(x) + float(scale)))


def _active_damage_envelope(t_sec: float) -> tuple[float, float]:
    if float(t_sec) >= TIER3_START_SEC:
        return TIER3_CONE_DEG, TIER3_MAX_RANGE_FT
    if float(t_sec) >= TIER2_START_SEC:
        return TIER2_CONE_DEG, TIER2_MAX_RANGE_FT
    return TIER1_CONE_DEG, TIER1_MAX_RANGE_FT


def _ned_to_body_matrix(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(float(roll_deg))
    p = np.radians(float(pitch_deg))
    y = np.radians(float(yaw_deg))
    tx = np.array([
        [1, 0, 0],
        [0, np.cos(r), np.sin(r)],
        [0, -np.sin(r), np.cos(r)],
    ], dtype=np.float64)
    ty = np.array([
        [np.cos(p), 0, -np.sin(p)],
        [0, 1, 0],
        [np.sin(p), 0, np.cos(p)],
    ], dtype=np.float64)
    tz = np.array([
        [np.cos(y), np.sin(y), 0],
        [-np.sin(y), np.cos(y), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return tx @ ty @ tz


def _body_to_ned_matrix(roll_deg, pitch_deg, yaw_deg):
    return _ned_to_body_matrix(roll_deg, pitch_deg, yaw_deg).T


def _log_so3(r_mat):
    tr = float(np.trace(r_mat))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return np.array([
        r_mat[2, 1] - r_mat[1, 2],
        r_mat[0, 2] - r_mat[2, 0],
        r_mat[1, 0] - r_mat[0, 1],
    ], dtype=np.float64) * (theta / denom)


class StateReconstructor:
    """RL step마다 정확히 한 번 갱신되는 상태 재구성기.

    advance()는 HP damage를 적분하고, episode 시간을 진행시키며, ownship 자세
    history에서 body p/q/r을 추정한다. build_observation()은 이 값들을 읽기만 해야
    한다. RL step 하나에서 advance()가 두 번 호출되면 damage와 시간이 중복 누적된다.
    """

    def __init__(self, dt_per_step: float = DT_PER_STEP):
        self.dt = float(dt_per_step)
        self._geo = GeometryInfo()
        self.reset()

    def reset(self) -> None:
        self.hp_own = 1.0
        self.hp_tgt = 1.0
        self.t_sec = 0.0
        self.last_dmg_dealt = 0.0
        self.last_dmg_taken = 0.0
        self.last_r_ft = 0.0
        self.last_ata_own = 180.0
        self.last_ata_tgt = 180.0
        self.prev_own_att = None
        self.own_pqr_est = np.zeros(3, dtype=np.float64)

    def advance(self, own_state, tgt_state) -> None:
        own = np.asarray(own_state, dtype=np.float64)
        tgt = np.asarray(tgt_state, dtype=np.float64)

        r_ft = self._geo._get_distance(own, tgt) * METER_TO_FEET
        ata_own = self._geo._get_antenna_train_angle(own, tgt, False)
        ata_tgt = self._geo._get_antenna_train_angle(tgt, own, False)

        rate_dealt = damage_rate(r_ft, ata_own, self.t_sec)
        rate_taken = damage_rate(r_ft, ata_tgt, self.t_sec)

        self.hp_tgt = max(0.0, self.hp_tgt - rate_dealt * self.dt)
        self.hp_own = max(0.0, self.hp_own - rate_taken * self.dt)

        self.last_dmg_dealt = rate_dealt
        self.last_dmg_taken = rate_taken
        self.last_r_ft = r_ft
        self.last_ata_own = ata_own
        self.last_ata_tgt = ata_tgt

        curr_att = np.array([
            own[StateIndex.ROLL],
            own[StateIndex.PITCH],
            own[StateIndex.YAW],
        ], dtype=np.float64)
        if self.prev_own_att is None:
            self.own_pqr_est[:] = 0.0
        else:
            r_prev = _body_to_ned_matrix(
                self.prev_own_att[0],
                self.prev_own_att[1],
                self.prev_own_att[2],
            )
            r_curr = _body_to_ned_matrix(
                curr_att[0],
                curr_att[1],
                curr_att[2],
            )
            r_delta = r_prev.T @ r_curr
            rotvec = _log_so3(r_delta)
            self.own_pqr_est = rotvec / max(self.dt, 1e-8)
        self.prev_own_att = curr_att

        self.t_sec += self.dt


_RECON = StateReconstructor()


def get_reconstructor() -> StateReconstructor:
    return _RECON


def reset_reconstructor() -> None:
    """episode 시작 시 재구성 HP, 시간, 자세 history를 초기화한다."""
    _RECON.reset()


def advance_reconstructor(own_state, tgt_state) -> None:
    """모듈 singleton을 RL step마다 정확히 한 번 진행시킨다."""
    _RECON.advance(own_state, tgt_state)


def build_observation(ownship_state, target_state, geo_info, wez_config=None,
                      reconstructor=None) -> np.ndarray:
    """42차원 claude42r observation을 만든다.

    wez_config는 기존 외부 signature 호환을 위해 받지만 이 observation에서는 쓰지
    않는다. damage envelope feature는 고정 서버 rule과 재구성 episode 시간을 사용한다.
    """
    rec = reconstructor if reconstructor is not None else _RECON
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    own = np.asarray(ownship_state, dtype=np.float64)
    tgt = np.asarray(target_state, dtype=np.float64)

    own_pos_ned = own[:3]
    tgt_pos_ned = tgt[:3]
    delta_ned = tgt_pos_ned - own_pos_ned

    own_roll = own[StateIndex.ROLL]
    own_pitch = own[StateIndex.PITCH]
    own_yaw = own[StateIndex.YAW]
    tgt_roll = tgt[StateIndex.ROLL]
    tgt_pitch = tgt[StateIndex.PITCH]
    tgt_yaw = tgt[StateIndex.YAW]

    r_ned_to_body_own = _ned_to_body_matrix(own_roll, own_pitch, own_yaw)
    r_body_to_ned_own = r_ned_to_body_own.T
    r_body_to_ned_tgt = _body_to_ned_matrix(tgt_roll, tgt_pitch, tgt_yaw)

    own_vel_body = np.asarray(own[6:9], dtype=np.float64)
    tgt_vel_body = np.asarray(tgt[6:9], dtype=np.float64)
    own_vel_ned = r_body_to_ned_own @ own_vel_body
    tgt_vel_ned = r_body_to_ned_tgt @ tgt_vel_body

    own_speed = float(np.linalg.norm(own_vel_body))
    target_speed = float(np.linalg.norm(tgt_vel_body))
    own_alt = reconstruct_altitude(own)
    target_alt = reconstruct_altitude(tgt)

    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)
    enemy_ata = geo_info._get_antenna_train_angle(target_state, ownship_state, False)

    # 0-2: ownship body frame으로 표현한 gravity vector.
    gravity_body = r_ned_to_body_own @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    obs[0] = float(gravity_body[0])
    obs[1] = float(gravity_body[1])
    obs[2] = float(gravity_body[2])

    # 3-6: ownship 속력과 raw body-frame 속도 u/v/w.
    obs[3] = normalize(own_speed, 0.0, MAX_SPEED)
    obs[4] = normalize(float(own_vel_body[0]), BODY_VEL_MIN, BODY_VEL_MAX)
    obs[5] = normalize(float(own_vel_body[1]), BODY_VEL_MIN, BODY_VEL_MAX)
    obs[6] = normalize(float(own_vel_body[2]), BODY_VEL_MIN, BODY_VEL_MAX)

    # 7-9: StateReconstructor.advance()가 추정한 ownship p/q/r.
    own_pqr = np.asarray(rec.own_pqr_est, dtype=np.float64)
    obs[7] = _tanh_scale(float(own_pqr[0]), PQR_SCALE_RAD_S)
    obs[8] = _tanh_scale(float(own_pqr[1]), PQR_SCALE_RAD_S)
    obs[9] = _tanh_scale(float(own_pqr[2]), PQR_SCALE_RAD_S)

    # 10-11: body 속도에서 추정한 공력 각도.
    u, v, w = own_vel_body
    if own_speed < 1.0:
        aoa_deg = 0.0
        sideslip_deg = 0.0
    else:
        aoa_deg = float(np.degrees(np.arctan2(w, u)))
        sideslip_deg = float(np.degrees(np.arctan2(v, np.sqrt(u * u + w * w))))
    obs[10] = _tanh_scale(aoa_deg, AOA_SCALE_DEG)
    obs[11] = _tanh_scale(sideslip_deg, SIDESLIP_SCALE_DEG)

    # 12: hard-deck margin. 0이면 정확히 MIN_ALTITUDE_M에 있다는 뜻이다.
    obs[12] = float(np.tanh((own_alt - MIN_ALTITUDE_M) / ALTITUDE_DANGER_SCALE_M))

    # 13: NED 기준 vertical speed. 양수면 상승 중이다.
    vertical_speed = -float(own_vel_ned[2])
    obs[13] = normalize(vertical_speed, -VERTICAL_SPEED_SCALE, VERTICAL_SPEED_SCALE)

    # 14-18: HP, target 속력, energy-height 우세/열세.
    obs[14] = normalize(float(rec.hp_own), 0.0, 1.0)
    obs[15] = normalize(target_speed, 0.0, MAX_SPEED)
    obs[16] = normalize(float(rec.hp_tgt), 0.0, 1.0)
    obs[17] = float(rec.hp_own - rec.hp_tgt)
    own_energy_height = own_alt + own_speed ** 2 / (2.0 * G)
    target_energy_height = target_alt + target_speed ** 2 / (2.0 * G)
    energy_advantage = own_energy_height - target_energy_height
    obs[18] = _softsign_scale(energy_advantage, ENERGY_ADVANTAGE_SCALE_M)

    # 19-21: ownship body frame 기준 target 상대 위치.
    rel_pos_body = r_ned_to_body_own @ delta_ned
    obs[19] = _softsign_scale(float(rel_pos_body[0]), REL_POS_SCALE_M)
    obs[20] = _softsign_scale(float(rel_pos_body[1]), REL_POS_SCALE_M)
    obs[21] = _softsign_scale(float(rel_pos_body[2]), REL_POS_SCALE_M)

    # 22-24: ownship body frame 기준 상대 속도.
    rel_vel_ned = tgt_vel_ned - own_vel_ned
    rel_vel_body = r_ned_to_body_own @ rel_vel_ned
    obs[22] = normalize(float(rel_vel_body[0]), REL_VEL_MIN, REL_VEL_MAX)
    obs[23] = normalize(float(rel_vel_body[1]), REL_VEL_MIN, REL_VEL_MAX)
    obs[24] = normalize(float(rel_vel_body[2]), REL_VEL_MIN, REL_VEL_MAX)

    # 25-26: 거리와 closure. closure > 0이면 서로 접근 중이다.
    obs[25] = normalize(float(distance), 0.0, MAX_RANGE_M)
    dist_norm = float(np.linalg.norm(delta_ned))
    if dist_norm > 1e-6:
        los_unit_ned = delta_ned / dist_norm
        closure = float(np.dot(own_vel_ned - tgt_vel_ned, los_unit_ned))
    else:
        closure = 0.0
    obs[26] = normalize(closure, -MAX_CLOSURE_SPEED, MAX_CLOSURE_SPEED)

    # 27-34: 교전 기하각을 sin/cos pair로 표현.
    obs[27], obs[28] = _sincos(ata)
    obs[29], obs[30] = _sincos(aa)
    obs[31], obs[32] = _sincos(az)
    obs[33], obs[34] = _sincos(el)

    # 35-38: 현재 active damage cone 기준 조준 품질과 피격 위험.
    obs[35] = float(2.0 * np.exp(-((float(ata) / 3.0) ** 2)) - 1.0)
    active_cone_deg, active_max_range_ft = _active_damage_envelope(float(rec.t_sec))
    aim_margin_raw = (active_cone_deg - abs(float(ata))) / max(active_cone_deg, 1e-6)
    obs[36] = float(np.tanh(aim_margin_raw))
    obs[37] = float(2.0 * np.exp(-((float(enemy_ata) / 3.0) ** 2)) - 1.0)
    enemy_aim_margin_raw = (
        active_cone_deg - abs(float(enemy_ata))
    ) / max(active_cone_deg, 1e-6)
    obs[38] = float(np.tanh(enemy_aim_margin_raw))

    # 39-40: 현재 active damage envelope 기준 near/far range margin.
    min_damage_range_m = MIN_DAMAGE_RANGE_FT * FEET_TO_METER
    active_max_range_m = active_max_range_ft * FEET_TO_METER
    active_span_m = max(active_max_range_m - min_damage_range_m, 1e-6)
    range_margin_near_raw = (float(distance) - min_damage_range_m) / active_span_m
    range_margin_far_raw = (active_max_range_m - float(distance)) / active_span_m
    obs[39] = float(np.tanh(range_margin_near_raw))
    obs[40] = float(np.tanh(range_margin_far_raw))

    # 41: episode 시간. 0s -> -1, EPISODE_MAX_TIME_SEC -> +1.
    obs[41] = normalize(float(rec.t_sec), 0.0, EPISODE_MAX_TIME_SEC)

    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return obs.astype(np.float32)


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "gravity_body_x",
            "gravity_body_y",
            "gravity_body_z",
            "own_speed_norm",
            "own_vel_body_x_norm",
            "own_vel_body_y_norm",
            "own_vel_body_z_norm",
            "own_p_est_tanh",
            "own_q_est_tanh",
            "own_r_est_tanh",
            "AoA_tanh",
            "sideslip_tanh",
            "altitude_margin_low_tanh",
            "vertical_speed_norm",
            "own_hp_norm",
            "target_speed_norm",
            "target_hp_norm",
            "hp_diff",
            "energy_advantage_softsign",
            "rel_pos_body_x_softsign",
            "rel_pos_body_y_softsign",
            "rel_pos_body_z_softsign",
            "rel_vel_body_x_norm",
            "rel_vel_body_y_norm",
            "rel_vel_body_z_norm",
            "slant_range_norm",
            "closure_rate_norm",
            "sin_ATA",
            "cos_ATA",
            "sin_AA",
            "cos_AA",
            "sin_LOS_azimuth",
            "cos_LOS_azimuth",
            "sin_LOS_elevation",
            "cos_LOS_elevation",
            "aim_sharp",
            "aim_margin_active_tanh",
            "enemy_aim_sharp",
            "enemy_aim_margin_active_tanh",
            "range_margin_near_tanh",
            "range_margin_far_active_tanh",
            "time_norm",
        ],
        "description": (
            "claude42r 42-D observation. 현재 state[6:9]는 body-frame 속도 "
            "u/v/w로 처리한다. speed + attitude로 NED 속도를 재구성하지 않는다. "
            "ownship p/q/r은 StateReconstructor에서 SO(3) log로 추정한다. "
            "target p/q/r은 사용하지 않는다. 상대 위치는 ownship body frame에서 "
            "표현하고 softsign으로 scaling한다. energy advantage는 energy-height "
            "차이를 softsign으로 scaling한 값이다. 고도 feature는 300m hard-deck "
            "margin을 tanh scaling한 값이다. 조준/range margin은 episode 시간에 "
            "따른 active damage envelope을 사용한다. observation 전체에 대한 마지막 "
            "clip은 적용하지 않는다. OBSERVATION_LOW/HIGH는 observation space bound일 "
            "뿐 실제 clipping 값이 아니다."
        ),
    }


__all__ = [
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "OBSERVATION_LOW",
    "OBSERVATION_HIGH",
    "build_observation",
    "describe_observation",
    "StateReconstructor",
    "damage_rate",
    "reconstruct_altitude",
    "reconstruct_speed",
    "get_reconstructor",
    "reset_reconstructor",
    "advance_reconstructor",
]