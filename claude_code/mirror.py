# -*- coding: utf-8 -*-
"""좌우 대칭(mirror) 증강용 부호 마스크 / 이산 action 반전.

교전을 수직평면 기준으로 좌우 반전하면(예: East 축 부호 뒤집기) 물리와 보상이
보존되고, 최적 정책은 equivariant 하다 — 즉 "대칭된 관측 → 대칭된 action". claude164r
관측(``claude_code.my_observation``)은 좌우 반전 시 **permutation 없이** 각 성분이
불변(+1) 또는 부호반전(-1)만 하므로, 고정 ±1 부호 마스크 ``M`` 하나로 표현된다.
raw 관측이면 ``M ⊙ obs`` 가 곧 대칭 관측이다(정규화된 관측에 대한 보정은 호출측이
running mean/var 로 처리한다).

부호 규칙(build_observation 의 feature 유도와 정확히 일치)
--------------------------------------------------------
- 스칼라: 좌우로 방향이 갈리는 "sin" 성분만 반전(_FLIP_SCALARS). ATA 는 부호 없는
  크기라 sin_ATA 도 불변. 나머지(cos·크기·거리·에너지·고도·시간 등)는 전부 불변.
- 벡터(7방향 × 6좌표계): **극(polar)벡터**(중력/LOS/속도들)는 어느 좌표계에서든 y(가운데)
  성분만 반전 → (+1,-1,+1). **축(axial=pseudo)벡터**(각속도 ω)는 x·z 반전, y 불변
  → (-1,+1,-1). (각속도가 pseudovector 라 부호 패턴이 반대다.)
- action history: roll/rudder 채널만 반전.

이산 action(bin) 반전
--------------------
action = [roll,pitch,rudder,throttle] 각 채널 num_bins 균등격자(중앙=0 대칭). 좌우반전은
roll/rudder 채널의 **bin 순서를 뒤집는다**(idx → num_bins-1-idx = 연속값 부호반전).

주의: 이 마스크는 my_observation 레이아웃 상수에서 직접 조립하므로, 레이아웃(스칼라
이름/벡터 spec/action history 길이)이 바뀌면 여기 규칙도 함께 갱신해야 한다. 길이
assert 로 1차 방어한다.
"""
from __future__ import annotations

import numpy as np

# 좌우반전 시 부호가 뒤집히는 스칼라(전부 "sin" 성분). 나머지 스칼라는 불변.
_FLIP_SCALARS = frozenset({
    "sideslip_tanh",
    "sin_AA",
    "sin_LOS_azimuth",
    "own_roll_sin",
    "own_yaw_sin",
    "target_roll_sin",
    "target_yaw_sin",
    "own_vel_bank_sin",
    "target_vel_bank_sin",
})

# 극(polar)벡터: 모든 좌표계에서 y(가운데) 성분만 반전 → (+1,-1,+1)
_POLAR_VECTORS = frozenset({"gravity", "los", "own_vel", "tgt_vel", "rel_vel"})
# 축(axial=pseudo)벡터(각속도): x·z 반전, y 불변 → (-1,+1,-1)
_AXIAL_VECTORS = frozenset({"own_omega", "tgt_omega"})

# action 채널 규약: 0=roll, 1=pitch, 2=rudder, 3=throttle. 좌우반전 시 roll/rudder 반전.
FLIP_ACTION_CHANNELS = (0, 2)


def build_obs_sign_mask() -> np.ndarray:
    """claude164r 관측 길이의 ±1 부호 마스크(np.float32) 반환.

    my_observation 의 실제 feature 배치(스칼라 → _VEC_LAYOUT 벡터 → action history)와
    정확히 대응한다. 길이가 OBSERVATION_SIZE 와 다르면 AssertionError.
    """
    from claude_code import my_observation as O

    mask: list[float] = []
    # 1) frame-invariant 스칼라
    for name in O.SCALAR_FEATURE_NAMES:
        mask.append(-1.0 if name in _FLIP_SCALARS else 1.0)
    # 2) frame-expressed 벡터: (key, kind, frame) 마다 x,y,z 3성분
    for key, _kind, _fr in O._VEC_LAYOUT:
        if key in _POLAR_VECTORS:
            mask.extend((1.0, -1.0, 1.0))
        elif key in _AXIAL_VECTORS:
            mask.extend((-1.0, 1.0, -1.0))
        else:
            raise ValueError(f"mirror: 알 수 없는 벡터 key {key!r} (POLAR/AXIAL 미분류)")
    # 3) action history: lag × ch, roll(0)/rudder(2) 반전
    for _lag in range(O.ACTION_HISTORY_LEN):
        for ch in range(O.ACTION_DIM):
            mask.append(-1.0 if ch in FLIP_ACTION_CHANNELS else 1.0)

    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape[0] != int(O.OBSERVATION_SIZE):
        raise AssertionError(
            f"mirror 마스크 길이 {arr.shape[0]} != OBSERVATION_SIZE {O.OBSERVATION_SIZE} "
            f"(my_observation 레이아웃 변경 시 claude_code/mirror.py 규칙 갱신 필요)")
    return arr


def mirror_action_indices(act_idx, num_bins: int) -> np.ndarray:
    """이산 action index(…, act_dim) 를 좌우반전. roll/rudder 채널만 bin 순서 반전.

    idx → (num_bins-1) - idx. 대칭격자에서 이는 연속 action 값의 부호반전과 동일하다.
    """
    out = np.asarray(act_idx, dtype=np.float32).copy()
    hi = float(int(num_bins) - 1)
    for ch in FLIP_ACTION_CHANNELS:
        out[..., ch] = hi - out[..., ch]
    return out


__all__ = ["build_obs_sign_mask", "mirror_action_indices", "FLIP_ACTION_CHANNELS"]
