# -*- coding: utf-8 -*-
"""[편집 가능] claude_code 관측 벡터.

여기서 관측을 직접 설계한다. `train.py --observation-module claude_code.my_observation`
으로 활성화하면 학습/로컬검증/제출이 모두 이 함수를 사용한다.

계약(원본 프레임워크와 동일):
  - OBSERVATION_SIZE: int                 (build_observation 이 반환하는 길이와 일치)
  - build_observation(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray(float32)
  선택:
  - OBSERVATION_MODE: str, OBSERVATION_LOW/HIGH: scalar 또는 배열, describe_observation()

기본값은 원본 tactical16(16차원)과 동일하게 맞춰 두었으므로, 활성화만 해서는
결과가 바뀌지 않는다. 차원을 바꾸면 기존 학습 번들과 호환되지 않으니 재학습해야 한다.

주의(학습 ↔ 추론 입력 차이): 대결 서버 추론에서는 상태가 PlaneInfo(위치/자세/속도)만
들어와 KCAS·ALT·HEALTH·WEZ 항목이 기본값이 된다. 상대 기하(ATA/AA/LOS, 상대 위치)는
양쪽 모두 채워지므로, 추격 정책은 이 항목 위주로 설계하면 전이에 유리하다.
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


OBSERVATION_MODE = "claude16"
OBSERVATION_SIZE = 16
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


def build_observation(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """기본값: 원본 tactical16 과 동일한 16차원 관측 (모두 [-1, 1])."""
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    # ownship 자세/속도/고도/체력
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]), -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]), -90.0, 90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]), 0.0, 360.0)
    obs[3] = normalize(float(ownship_state[StateIndex.KCAS]), 0.0, 600.0)
    obs[4] = normalize(float(ownship_state[StateIndex.ALT]), 0.0, 15000.0)
    obs[5] = normalize(float(ownship_state[StateIndex.HEALTH]), 0.0, 1.0)
    # 상대 위치 delta
    obs[6] = normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = normalize(float(delta[2]), -8000.0, 8000.0)
    # 기하 (ATA/AA/LOS)
    obs[9] = normalize(float(ata), -180.0, 180.0)
    obs[10] = normalize(float(aa), -180.0, 180.0)
    obs[11] = normalize(float(az), -180.0, 180.0)
    obs[12] = normalize(float(el), -90.0, 90.0)
    # 표적 체력
    obs[13] = normalize(float(target_state[StateIndex.HEALTH]), 0.0, 1.0)
    # WEZ 진입 플래그 (+1 / -1)
    if wez_config is not None:
        in_wez = (
            wez_config["min_range_m"] <= distance <= wez_config["max_range_m"]
            and abs(float(ata)) <= wez_config["angle_deg"] / 2.0
        )
        obs[14] = 1.0 if in_wez else -1.0
    else:
        obs[14] = -1.0
    # 추격 점수 (ATA×range gradient → [-1, 1])
    ata_factor = max(0.0, 1.0 - abs(float(ata)) / 30.0)
    range_factor = max(0.0, 1.0 - distance / 3000.0)
    obs[15] = 2.0 * (ata_factor * range_factor) - 1.0
    return obs


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "ownship_roll", "ownship_pitch", "ownship_yaw", "ownship_kcas",
            "ownship_alt", "ownship_health", "delta_n", "delta_e", "delta_d",
            "ata", "aa", "los_az", "los_el", "target_health", "in_wez", "pursuit_score",
        ],
        "description": "claude_code 16-D observation (tactical16 동일 기본값).",
    }


__all__ = ["OBSERVATION_MODE", "OBSERVATION_SIZE", "OBSERVATION_LOW",
           "OBSERVATION_HIGH", "build_observation", "describe_observation"]
