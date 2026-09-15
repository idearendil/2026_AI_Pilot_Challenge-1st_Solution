# -*- coding: utf-8 -*-
"""claude_code 관측 모듈 (multi-frame).

이 파일은 학습, 평가, self-play, 제출 코드가 사용하는 외부 계약을 유지한다.

  - StateReconstructor가 episode 중 재구성 상태를 저장한다.
  - _RECON은 모듈 전역 singleton이다.
  - reset_reconstructor()/advance_reconstructor()는 RL step마다 한 번만 호출한다.
  - build_observation()은 reconstructor 값을 읽기만 한다.

관측 구조 (claude164r)
----------------------
관측은 두 블록으로 나뉜다.

  1) frame-invariant 스칼라 블록 (50개): 속력, HP, 에너지고도차, 슬랜트 거리,
     closure, 교전 기하각(ATA/AA/LOS az·el의 sin/cos), 조준/거리 margin, 시간,
     나·상대 절대 자세(roll/pitch/yaw sin·cos), 나·상대 속도축 뱅크각 μ(sin·cos),
     연료(나·상대), 순간 damage rate(가함·받음), pursuit_score, 절대 고도(나·상대,
     고고도까지 선형) 등. 좌표계와 무관해 한 번만 넣는다.

  2) frame-expressed 벡터 블록 (114개): 아래 7개 방향 벡터를 6개 좌표계 성분으로
     각각 표현한다. 퇴화(정보 없는) 조합은 제외한다.

방향 벡터 (7):
  gravity     : 월드 아래 방향(중력) 단위벡터  → 자세를 절대 기준으로 인지
  los         : 표적 상대위치(시선) 단위벡터
  own_vel     : 내 속도 벡터 (정규화 /600)
  tgt_vel     : 표적 속도 벡터 (정규화 /600)
  rel_vel     : 상대 속도 벡터 (정규화 /600)
  own_omega   : 내 각속도 ω (tanh /4)
  tgt_omega   : 표적 각속도 ω (tanh /4)

좌표계 (6):
  world   : 월드 절대 좌표계(NED)
  mybody  : 내 전투기 body frame (x 기수전방/y 오른쪽날개/z 아래)
  oppbody : 상대 전투기 body frame
  myvel   : 내 선속도 좌표계 (x=속도방향, roll은 중력으로 고정)
  oppvel  : 상대 선속도 좌표계
  los     : 두 기체를 잇는 직선을 x축으로 하는 좌표계 (roll은 중력으로 고정)

퇴화 조합(제외):
  gravity@world (=상수 [0,0,1]), los@los (=상수 [1,0,0]),
  own_vel@myvel (=(|v|,0,0)), tgt_vel@oppvel (=(|v|,0,0))

각속도를 좌표계로 만들 수는 없다(회전이 없으면 축이 정의되지 않고, 나머지 두 축을
고정할 기준이 없다). 대신 각속도 '벡터'를 위 6개 좌표계 성분으로 다시 표현한다.

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

중요: state[6:9]는 body-frame 속도 u/v/w로 직접 사용한다. NED 속도가 필요할 때만
rotation matrix로 body 속도를 NED로 변환한다.

damage 재구성
-------------
서버가 HP를 직접 주지 않아도 된다. StateReconstructor는 현재 거리, ATA, episode
시간을 사용해 같은 3-tier cone-damage rule을 적분한다. 하위 tier가 우선이다.

p/q/r 재구성
------------
ownship과 target 양쪽의 body p/q/r을 각각 추정한다. advance()에서 직전 자세와 현재
자세로 상대 rotation의 SO(3) log를 취해 yaw wrap에도 spike가 없다. own_pqr_est /
tgt_pqr_est는 각각 해당 기체의 body frame 성분이다.
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

# ── action history: 직전 5 RL-step 동안 '내가 결정한 action'(연속 [-1,1]^4)을 관측에 붙인다.
# 5 step × 4 dim = 20 차원. 에피소드 시작 전(0 step 이전)은 0 으로 채운다. actor/critic 이
# 같은 관측을 쓰므로 둘 다에 반영된다. 순서: [가장 최근, ..., 가장 오래된] × 4채널.
ACTION_HISTORY_LEN = 5
ACTION_DIM = 4
ACTION_HISTORY_SIZE = ACTION_HISTORY_LEN * ACTION_DIM   # 20

MAX_SPEED = 600.0
BODY_VEL_MIN = -600.0
BODY_VEL_MAX = 600.0
REL_VEL_MIN = -600.0
REL_VEL_MAX = 600.0
MAX_RANGE_M = 2500.0
MAX_CLOSURE_SPEED = 1000.0
VERTICAL_SPEED_SCALE = 100.0
PQR_SCALE_RAD_S = 4.0
# 가속도(선가속도) 정규화 상한[m/s^2]. accel = d(v_ned)/dt 의 step 차분. 고기동(≈9g≈88)+여유.
ACCEL_SCALE_M_S2 = 150.0
AOA_SCALE_DEG = 30.0
SIDESLIP_SCALE_DEG = 15.0
MIN_ALTITUDE_M = 1000.0 * FEET_TO_METER
ALTITUDE_DANGER_SCALE_M = 300.0
MAX_ALTITUDE_M = 15000.0            # 절대 고도 정규화 상한(고고도 해상도 보존, 선형)
ENERGY_ADVANTAGE_SCALE_M = 5000.0
REL_POS_SCALE_M = 1000.0
PURSUIT_ATA_SCALE_DEG = 30.0        # pursuit_score: |ATA| 이 이 값이면 조준 factor=0
PURSUIT_RANGE_M = 3000.0            # pursuit_score: 거리 이 값이면 거리 factor=0

# 연료 재구성(서버 추론에선 fuel=0 으로 들어오므로 HP 처럼 관측에 넣지 않고 재구성한다).
# 만탱크=1.0 에서 시작해 속도(≈throttle 대리)에 비례해 아주 천천히 감소한다. 학습/추론이
# 동일한 관측값(속도+시간)으로 계산하므로 train/test 가 일치한다. 실측 소모율(~300m/s
# 순항에서 20s 동안 약 0.16% 감소)에 맞춰 계수를 잡았다(1 episode 안에선 거의 상수).
FUEL_BURN_PER_SEC = 8.0e-5          # 기준 속도에서의 초당 연료 소모(비율)
FUEL_REF_SPEED = 300.0             # 연료 소모를 1.0 배로 보는 기준 속력[m/s]

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

# ── frame-invariant 스칼라 블록 (좌표계와 무관, 한 번만) ──────────────────────
SCALAR_FEATURE_NAMES = [
    "own_speed_norm",
    "target_speed_norm",
    "AoA_tanh",
    "sideslip_tanh",
    "altitude_margin_low_tanh",
    "vertical_speed_norm",
    "own_hp_norm",
    "target_hp_norm",
    "hp_diff",
    "energy_advantage_softsign",
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
    # ── 복원/추가 feature ──
    "own_roll_sin",
    "own_roll_cos",
    "own_pitch_sin",
    "own_pitch_cos",
    "own_yaw_sin",
    "own_yaw_cos",
    "target_roll_sin",
    "target_roll_cos",
    "target_pitch_sin",
    "target_pitch_cos",
    "target_yaw_sin",
    "target_yaw_cos",
    "own_vel_bank_sin",
    "own_vel_bank_cos",
    "target_vel_bank_sin",
    "target_vel_bank_cos",
    "own_fuel_norm",
    "target_fuel_norm",
    "damage_rate_dealt",
    "damage_rate_taken",
    "pursuit_score",
    "own_altitude_abs_norm",
    "target_altitude_abs_norm",
]

# ── frame-expressed 벡터 블록 ────────────────────────────────────────────────
# 6개 좌표계. 순서 고정(관측 index 규약).
FRAME_NAMES = ["world", "mybody", "oppbody", "myvel", "oppvel", "los"]

# (key, scale_kind, 제외할 frame 집합)
#   scale_kind: "unit"  → 단위벡터 성분 그대로
#               "vel"   → normalize(comp, -600, 600)
#               "omega" → tanh(comp / PQR_SCALE_RAD_S)
#               "accel" → normalize(comp, -ACCEL_SCALE_M_S2, +ACCEL_SCALE_M_S2)
VECTOR_SPECS = [
    ("gravity",   "unit",  frozenset({"world"})),   # world 에선 상수 [0,0,1]
    ("los",       "unit",  frozenset({"los"})),      # los frame 에선 상수 [1,0,0]
    ("own_vel",   "vel",   frozenset({"myvel"})),    # myvel frame 에선 (|v|,0,0)
    ("tgt_vel",   "vel",   frozenset({"oppvel"})),   # oppvel frame 에선 (|v|,0,0)
    ("rel_vel",   "vel",   frozenset()),
    ("own_omega", "omega", frozenset()),
    ("tgt_omega", "omega", frozenset()),
    # 선가속도(내/상대). 속도 항과 동일 패턴: 자기-속도정렬 frame 1개씩 제외(5 frame). accel
    # 은 속도축과 정렬돼 있지 않지만, 그 frame 성분은 다른 frame 에서 복원 가능하므로 제외해도
    # 정보 손실이 사실상 없다(속도 항 규약과 대칭 유지).
    ("own_accel", "accel", frozenset({"myvel"})),
    ("tgt_accel", "accel", frozenset({"oppvel"})),
]


def _vector_layout():
    """(key, scale_kind, frame) 리스트. build_observation 과 이름 생성이 공유한다."""
    layout = []
    for key, kind, skip in VECTOR_SPECS:
        for fr in FRAME_NAMES:
            if fr in skip:
                continue
            layout.append((key, kind, fr))
    return layout


_VEC_LAYOUT = _vector_layout()


def _all_feature_names():
    names = list(SCALAR_FEATURE_NAMES)
    for key, _kind, fr in _VEC_LAYOUT:
        for ax in ("x", "y", "z"):
            names.append(f"{key}__{fr}_{ax}")
    # action history 20개: act_hist_t{-1..-5}_{ch0..3} (가장 최근 → 오래된).
    for lag in range(1, ACTION_HISTORY_LEN + 1):
        for ch in range(ACTION_DIM):
            names.append(f"act_hist_t-{lag}_ch{ch}")
    return names


FEATURE_NAMES = _all_feature_names()

OBSERVATION_MODE = "claude164r"
OBSERVATION_SIZE = len(FEATURE_NAMES)

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


def _dir_frame(x_axis_ned):
    """방향 벡터 x_axis_ned를 x축으로, 중력을 기준으로 roll을 고정한 좌표계를 만든다.

    반환 R은 R_ned_to_frame (행 = frame 축의 NED 표현). frame_component = R @ v_ned.
    frame 규약은 body와 동일(x 전방, y 오른쪽, z 아래). x축이 중력과 거의 평행하면
    (수직 비행) 북 → 동 순으로 fallback 기준축을 쓴다. x축 크기가 0이면 항등.
    """
    x = np.asarray(x_axis_ned, dtype=np.float64)
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return np.eye(3, dtype=np.float64)
    x = x / nx
    down = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z = down - np.dot(down, x) * x
    if np.linalg.norm(z) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # north
        z = ref - np.dot(ref, x) * x
        if np.linalg.norm(z) < 1e-6:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # east
            z = ref - np.dot(ref, x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.vstack([x, y, z])


def _bank_about_dir(r_body_to_ned, dir_ned) -> tuple[float, float]:
    """방향벡터 dir_ned를 축으로 한 body 뱅크각 μ의 (sin, cos)를 반환한다.

    dir_ned로 만든 _dir_frame(roll은 중력으로 고정)에서 body y축(오른쪽 날개)이
    얼마나 기울어졌는지가 μ다. 날개 수평이면 μ=0, 오른쪽으로 뱅크하면 μ>0.
    dir_ned가 속도벡터면 μ는 공력 뱅크각(양력벡터 방향), LOS면 LOS축 기준 뱅크각이다.
    """
    R = _dir_frame(dir_ned)   # R_ned_to_frame (행 = frame 축)
    body_y_ned = np.asarray(r_body_to_ned, dtype=np.float64)[:, 1]
    yv = R @ body_y_ned       # body 오른날개를 frame 성분으로: [전방, 오른쪽, 아래]
    mu = float(np.arctan2(float(yv[2]), float(yv[1])))
    return float(np.sin(mu)), float(np.cos(mu))


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


def _estimate_pqr(prev_att, curr_att, dt):
    """직전/현재 자세(roll,pitch,yaw deg)에서 body p/q/r [rad/s]을 추정한다."""
    if prev_att is None:
        return np.zeros(3, dtype=np.float64)
    r_prev = _body_to_ned_matrix(prev_att[0], prev_att[1], prev_att[2])
    r_curr = _body_to_ned_matrix(curr_att[0], curr_att[1], curr_att[2])
    r_delta = r_prev.T @ r_curr
    return _log_so3(r_delta) / max(dt, 1e-8)


class StateReconstructor:
    """RL step마다 정확히 한 번 갱신되는 상태 재구성기.

    advance()는 HP damage를 적분하고, episode 시간을 진행시키며, ownship/target 자세
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
        self.fuel_own = 1.0
        self.fuel_tgt = 1.0
        self.t_sec = 0.0
        self.last_dmg_dealt = 0.0
        self.last_dmg_taken = 0.0
        self.last_r_ft = 0.0
        self.last_ata_own = 180.0
        self.last_ata_tgt = 180.0
        self.prev_own_att = None
        self.prev_tgt_att = None
        self.own_pqr_est = np.zeros(3, dtype=np.float64)
        self.tgt_pqr_est = np.zeros(3, dtype=np.float64)
        # 선가속도 추정용: 직전 step 의 NED 속도(내/상대). pqr 을 자세 차분으로 추정하는 것과
        # 같은 규약이라 prev_*_att 와 수명이 같다. 첫 step 은 직전 값이 없어 accel 0.
        self.prev_own_vel_ned = None
        self.prev_tgt_vel_ned = None
        self.own_accel_est = np.zeros(3, dtype=np.float64)
        self.tgt_accel_est = np.zeros(3, dtype=np.float64)
        # 직전 ACTION_HISTORY_LEN 개 action(각 ACTION_DIM 차원). row 0 = 가장 최근. 에피소드
        # 시작 시 전부 0(= 0 step 이전 action 없음). push_action 으로 갱신.
        self.action_history = np.zeros((ACTION_HISTORY_LEN, ACTION_DIM), dtype=np.float64)

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

        # 연료 재구성: 속도(≈throttle 대리)에 비례해 만탱크(1.0)에서 천천히 감소.
        own_speed = float(np.linalg.norm(own[6:9]))
        tgt_speed = float(np.linalg.norm(tgt[6:9]))
        burn_own = FUEL_BURN_PER_SEC * (own_speed / FUEL_REF_SPEED)
        burn_tgt = FUEL_BURN_PER_SEC * (tgt_speed / FUEL_REF_SPEED)
        self.fuel_own = max(0.0, self.fuel_own - burn_own * self.dt)
        self.fuel_tgt = max(0.0, self.fuel_tgt - burn_tgt * self.dt)

        self.last_dmg_dealt = rate_dealt
        self.last_dmg_taken = rate_taken
        self.last_r_ft = r_ft
        self.last_ata_own = ata_own
        self.last_ata_tgt = ata_tgt

        curr_own_att = np.array([
            own[StateIndex.ROLL], own[StateIndex.PITCH], own[StateIndex.YAW],
        ], dtype=np.float64)
        curr_tgt_att = np.array([
            tgt[StateIndex.ROLL], tgt[StateIndex.PITCH], tgt[StateIndex.YAW],
        ], dtype=np.float64)
        self.own_pqr_est = _estimate_pqr(self.prev_own_att, curr_own_att, self.dt)
        self.tgt_pqr_est = _estimate_pqr(self.prev_tgt_att, curr_tgt_att, self.dt)
        # 선가속도 추정: NED 속도의 step 차분(위 pqr 을 자세 차분으로 추정하는 것과 동일 패턴).
        # own[6:9]/tgt[6:9] 는 body 속도이므로 body->NED 로 돌려서 차분한다.
        own_vel_ned = _ned_to_body_matrix(
            own[StateIndex.ROLL], own[StateIndex.PITCH], own[StateIndex.YAW]).T @ own[6:9]
        tgt_vel_ned = _ned_to_body_matrix(
            tgt[StateIndex.ROLL], tgt[StateIndex.PITCH], tgt[StateIndex.YAW]).T @ tgt[6:9]
        self.own_accel_est = (np.zeros(3, dtype=np.float64) if self.prev_own_vel_ned is None
                              else (own_vel_ned - self.prev_own_vel_ned) / self.dt)
        self.tgt_accel_est = (np.zeros(3, dtype=np.float64) if self.prev_tgt_vel_ned is None
                              else (tgt_vel_ned - self.prev_tgt_vel_ned) / self.dt)
        self.prev_own_vel_ned = own_vel_ned
        self.prev_tgt_vel_ned = tgt_vel_ned

        self.prev_own_att = curr_own_att
        self.prev_tgt_att = curr_tgt_att

        self.t_sec += self.dt

    def push_action(self, action) -> None:
        """이번 RL-step 에 결정한 action(연속 [-1,1]^4)을 history 맨 앞에 넣고 한 칸 민다.

        관측 빌드 **전에** 호출해야 그 관측이 방금 결정한 action 을 포함한다(lag 없음).
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < ACTION_DIM:
            a = np.pad(a, (0, ACTION_DIM - a.size))
        else:
            a = a[:ACTION_DIM]
        self.action_history = np.roll(self.action_history, 1, axis=0)
        self.action_history[0] = a

    def action_history_flat(self) -> np.ndarray:
        """(ACTION_HISTORY_SIZE,) = [가장 최근 4채널, ..., 가장 오래된 4채널]."""
        return self.action_history.reshape(-1).astype(np.float32)


_RECON = StateReconstructor()


def get_reconstructor() -> StateReconstructor:
    return _RECON


def reset_reconstructor() -> None:
    """episode 시작 시 재구성 HP, 시간, 자세 history를 초기화한다."""
    _RECON.reset()


def advance_reconstructor(own_state, tgt_state) -> None:
    """모듈 singleton을 RL step마다 정확히 한 번 진행시킨다."""
    _RECON.advance(own_state, tgt_state)


def push_action_reconstructor(action) -> None:
    """모듈 singleton(전역 _RECON = ownship 관점)의 action history 를 RL step 당 한 번 갱신.

    관측이 방금 결정한 action 을 포함하도록, 그 관측을 빌드하기 **직전**에 호출한다
    (학습: env.step 전, 추론: action 결정 후 다음 관측 빌드 전).
    """
    _RECON.push_action(action)


def build_observation(ownship_state, target_state, geo_info, wez_config=None,
                      reconstructor=None) -> np.ndarray:
    """claude164r multi-frame observation을 만든다.

    wez_config는 기존 외부 signature 호환을 위해 받지만 이 observation에서는 쓰지
    않는다. damage envelope feature는 고정 서버 rule과 재구성 episode 시간을 사용한다.
    """
    rec = reconstructor if reconstructor is not None else _RECON

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
    r_ned_to_body_tgt = _ned_to_body_matrix(tgt_roll, tgt_pitch, tgt_yaw)
    r_body_to_ned_tgt = r_ned_to_body_tgt.T

    own_vel_body = np.asarray(own[6:9], dtype=np.float64)
    tgt_vel_body = np.asarray(tgt[6:9], dtype=np.float64)
    own_vel_ned = r_body_to_ned_own @ own_vel_body
    tgt_vel_ned = r_body_to_ned_tgt @ tgt_vel_body
    rel_vel_ned = tgt_vel_ned - own_vel_ned

    own_speed = float(np.linalg.norm(own_vel_body))
    target_speed = float(np.linalg.norm(tgt_vel_body))
    own_alt = reconstruct_altitude(own)
    target_alt = reconstruct_altitude(tgt)

    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)
    enemy_ata = geo_info._get_antenna_train_angle(target_state, ownship_state, False)

    # ── frame-invariant 스칼라 블록 (SCALAR_FEATURE_NAMES 순서와 정확히 일치) ──
    u, v, w = own_vel_body
    if own_speed < 1.0:
        aoa_deg = 0.0
        sideslip_deg = 0.0
    else:
        aoa_deg = float(np.degrees(np.arctan2(w, u)))
        sideslip_deg = float(np.degrees(np.arctan2(v, np.sqrt(u * u + w * w))))

    vertical_speed = -float(own_vel_ned[2])

    own_energy_height = own_alt + own_speed ** 2 / (2.0 * G)
    target_energy_height = target_alt + target_speed ** 2 / (2.0 * G)
    energy_advantage = own_energy_height - target_energy_height

    dist_norm = float(np.linalg.norm(delta_ned))
    if dist_norm > 1e-6:
        los_unit_ned = delta_ned / dist_norm
        closure = float(np.dot(own_vel_ned - tgt_vel_ned, los_unit_ned))
    else:
        los_unit_ned = np.zeros(3, dtype=np.float64)
        closure = 0.0

    sin_ata, cos_ata = _sincos(ata)
    sin_aa, cos_aa = _sincos(aa)
    sin_az, cos_az = _sincos(az)
    sin_el, cos_el = _sincos(el)

    aim_sharp = float(2.0 * np.exp(-((float(ata) / 3.0) ** 2)) - 1.0)
    active_cone_deg, active_max_range_ft = _active_damage_envelope(float(rec.t_sec))
    aim_margin_raw = (active_cone_deg - abs(float(ata))) / max(active_cone_deg, 1e-6)
    enemy_aim_sharp = float(2.0 * np.exp(-((float(enemy_ata) / 3.0) ** 2)) - 1.0)
    enemy_aim_margin_raw = (
        active_cone_deg - abs(float(enemy_ata))
    ) / max(active_cone_deg, 1e-6)

    min_damage_range_m = MIN_DAMAGE_RANGE_FT * FEET_TO_METER
    active_max_range_m = active_max_range_ft * FEET_TO_METER
    active_span_m = max(active_max_range_m - min_damage_range_m, 1e-6)
    range_margin_near_raw = (float(distance) - min_damage_range_m) / active_span_m
    range_margin_far_raw = (active_max_range_m - float(distance)) / active_span_m

    # ── 복원/추가 feature 계산 ──
    own_roll_sin, own_roll_cos = _sincos(own_roll)
    own_pitch_sin, own_pitch_cos = _sincos(own_pitch)
    own_yaw_sin, own_yaw_cos = _sincos(own_yaw)
    tgt_roll_sin, tgt_roll_cos = _sincos(tgt_roll)
    tgt_pitch_sin, tgt_pitch_cos = _sincos(tgt_pitch)
    tgt_yaw_sin, tgt_yaw_cos = _sincos(tgt_yaw)
    own_vel_bank_sin, own_vel_bank_cos = _bank_about_dir(r_body_to_ned_own, own_vel_ned)
    tgt_vel_bank_sin, tgt_vel_bank_cos = _bank_about_dir(r_body_to_ned_tgt, tgt_vel_ned)
    dmg_dealt = float(rec.last_dmg_dealt)   # 내가 상대에 가하는 초당 damage rate [0,~1]
    dmg_taken = float(rec.last_dmg_taken)   # 내가 받는 초당 damage rate [0,~1]
    pursuit_ata_factor = max(0.0, 1.0 - abs(float(ata)) / PURSUIT_ATA_SCALE_DEG)
    pursuit_range_factor = max(0.0, 1.0 - float(distance) / PURSUIT_RANGE_M)
    pursuit_score = 2.0 * (pursuit_ata_factor * pursuit_range_factor) - 1.0

    scalars = [
        normalize(own_speed, 0.0, MAX_SPEED),
        normalize(target_speed, 0.0, MAX_SPEED),
        _tanh_scale(aoa_deg, AOA_SCALE_DEG),
        _tanh_scale(sideslip_deg, SIDESLIP_SCALE_DEG),
        float(np.tanh((own_alt - MIN_ALTITUDE_M) / ALTITUDE_DANGER_SCALE_M)),
        normalize(vertical_speed, -VERTICAL_SPEED_SCALE, VERTICAL_SPEED_SCALE),
        normalize(float(rec.hp_own), 0.0, 1.0),
        normalize(float(rec.hp_tgt), 0.0, 1.0),
        float(rec.hp_own - rec.hp_tgt),
        _softsign_scale(energy_advantage, ENERGY_ADVANTAGE_SCALE_M),
        normalize(float(distance), 0.0, MAX_RANGE_M),
        normalize(closure, -MAX_CLOSURE_SPEED, MAX_CLOSURE_SPEED),
        sin_ata, cos_ata,
        sin_aa, cos_aa,
        sin_az, cos_az,
        sin_el, cos_el,
        aim_sharp,
        float(np.tanh(aim_margin_raw)),
        enemy_aim_sharp,
        float(np.tanh(enemy_aim_margin_raw)),
        float(np.tanh(range_margin_near_raw)),
        float(np.tanh(range_margin_far_raw)),
        normalize(float(rec.t_sec), 0.0, EPISODE_MAX_TIME_SEC),
        # ── 복원/추가 feature (SCALAR_FEATURE_NAMES 뒷부분과 정확히 일치) ──
        own_roll_sin, own_roll_cos,
        own_pitch_sin, own_pitch_cos,
        own_yaw_sin, own_yaw_cos,
        tgt_roll_sin, tgt_roll_cos,
        tgt_pitch_sin, tgt_pitch_cos,
        tgt_yaw_sin, tgt_yaw_cos,
        own_vel_bank_sin, own_vel_bank_cos,
        tgt_vel_bank_sin, tgt_vel_bank_cos,
        normalize(float(rec.fuel_own), 0.0, 1.0),
        normalize(float(rec.fuel_tgt), 0.0, 1.0),
        float(np.clip(2.0 * dmg_dealt - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * dmg_taken - 1.0, -1.0, 1.0)),
        float(pursuit_score),
        normalize(own_alt, 0.0, MAX_ALTITUDE_M),
        normalize(target_alt, 0.0, MAX_ALTITUDE_M),
    ]

    # ── frame-expressed 벡터 블록 (_VEC_LAYOUT 순서와 정확히 일치) ────────────
    own_omega_ned = r_body_to_ned_own @ np.asarray(rec.own_pqr_est, dtype=np.float64)
    tgt_omega_ned = r_body_to_ned_tgt @ np.asarray(rec.tgt_pqr_est, dtype=np.float64)

    frames = {
        "world": np.eye(3, dtype=np.float64),
        "mybody": r_ned_to_body_own,
        "oppbody": r_ned_to_body_tgt,
        "myvel": _dir_frame(own_vel_ned),
        "oppvel": _dir_frame(tgt_vel_ned),
        "los": _dir_frame(delta_ned),
    }
    ned_vecs = {
        "gravity": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "los": los_unit_ned,
        "own_vel": own_vel_ned,
        "tgt_vel": tgt_vel_ned,
        "rel_vel": rel_vel_ned,
        "own_omega": own_omega_ned,
        "tgt_omega": tgt_omega_ned,
        "own_accel": np.asarray(rec.own_accel_est, dtype=np.float64),
        "tgt_accel": np.asarray(rec.tgt_accel_est, dtype=np.float64),
    }

    vec_feats = []
    for key, kind, fr in _VEC_LAYOUT:
        comp = frames[fr] @ ned_vecs[key]
        if kind == "unit":
            vec_feats.extend((float(comp[0]), float(comp[1]), float(comp[2])))
        elif kind == "vel":
            vec_feats.extend(
                normalize(float(comp[i]), REL_VEL_MIN, REL_VEL_MAX) for i in range(3)
            )
        elif kind == "accel":
            vec_feats.extend(
                normalize(float(comp[i]), -ACCEL_SCALE_M_S2, ACCEL_SCALE_M_S2)
                for i in range(3)
            )
        else:  # omega
            vec_feats.extend(
                _tanh_scale(float(comp[i]), PQR_SCALE_RAD_S) for i in range(3)
            )

    # ── action history 20개(직전 5 RL-step 의 내 action, [-1,1]^4). reconstructor 소유. ──
    act_hist = list(rec.action_history_flat())

    obs = np.asarray(scalars + vec_feats + act_hist, dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return obs.astype(np.float32)


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "frames": list(FRAME_NAMES),
        "vectors": [k for k, _, _ in VECTOR_SPECS],
        "features": list(FEATURE_NAMES),
        "description": (
            "claude164r multi-frame observation. frame-invariant 스칼라 50개 "
            "(나·상대 절대 자세 sin/cos, 나·상대 속도축 뱅크각 μ sin/cos, 연료 나·상대, "
            "순간 damage rate 2개, pursuit_score, 절대 고도 나·상대 포함) + "
            "7개 방향 벡터(중력/LOS/내속도/표적속도/상대속도/내ω/표적ω)를 6개 "
            "좌표계(world/mybody/oppbody/myvel/oppvel/los)로 표현한 114개. "
            "퇴화 조합(gravity@world, los@los, own_vel@myvel, tgt_vel@oppvel)은 "
            "제외한다. 속도 좌표계와 LOS 좌표계는 x축을 해당 방향에 두고 roll은 "
            "중력으로 고정한다. 각속도는 좌표계로 만들 수 없어 각속도 벡터를 6개 "
            "좌표계 성분으로 재표현한다. ownship/target p/q/r은 자세 history의 SO(3) "
            "log로 각각 추정한다. observation 전체에 대한 마지막 clip은 적용하지 "
            "않는다. OBSERVATION_LOW/HIGH는 space bound일 뿐 실제 clipping 값이 아니다."
        ),
    }


__all__ = [
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "OBSERVATION_LOW",
    "OBSERVATION_HIGH",
    "FEATURE_NAMES",
    "FRAME_NAMES",
    "VECTOR_SPECS",
    "build_observation",
    "describe_observation",
    "StateReconstructor",
    "damage_rate",
    "reconstruct_altitude",
    "reconstruct_speed",
    "get_reconstructor",
    "reset_reconstructor",
    "advance_reconstructor",
    "push_action_reconstructor",
    "ACTION_HISTORY_LEN",
    "ACTION_DIM",
    "ACTION_HISTORY_SIZE",
]
