# MPC 구조와 수학 계약

## 전체 계층

이 구현은 별도의 hand-coded lead/lag 전술 상태기를 두지 않습니다. 매 0.1초
현재 상태에서 가능한 조종 시퀀스를 직접 평가하는 receding-horizon 구조입니다.

```text
MPCCommandPolicy
  └─ 공개 UDP packet → 내부 state와 causal angular-rate 추정
MPCActionProvider
  ├─ 60 Hz 호출 중 10 Hz에서만 새 계획
  └─ 직전 action을 중간 5 frame 동안 유지
TargetMotionPredictor
  └─ 상대 velocity/acceleration/ground-track turn-rate 감쇠 외삽
MPCPlanner
  ├─ 이전 CEM 분포를 0.1초 이동해 warm start
  ├─ 대칭 Gaussian 후보 + 작은 일반 BFM maneuver library
  └─ elite로 mean/std를 2회 갱신
NativePredictor
  └─ 각 후보를 독립 6DoF 모델에서 60 Hz rollout
```

## 좌표계

위치와 world velocity는 NED를 사용합니다.

- `N`: 북쪽 양수
- `E`: 동쪽 양수
- `D`: 아래쪽 양수

패킷 velocity는 body frame입니다.

- body `x/u`: 기수 전방
- body `y/v`: 오른쪽
- body `z/w`: 아래쪽

`src/mpc/transforms.py`가 Euler attitude로 body velocity를 NED velocity로
변환합니다. 네트워크의 Euler 각도는 degree, 내부 각속도와 강체동역학은
radian 단위를 사용합니다.

Euler wrap에서 `179° → -179°`가 거대한 각속도로 보이지 않도록 ownship과
target의 `p/q/r`은 SO(3) relative rotation으로 추정합니다.

## 상대 궤적 예측

상대 예측은 현재와 직전 공개 표본만 사용합니다.

1. NED velocity 차분으로 acceleration을 추정하고 80 m/s²로 제한합니다.
2. 수평 ground track의 wrapped heading 차분으로 turn rate를 추정하고
   ±1.5 rad/s로 제한합니다.
3. acceleration과 turn rate에 각각 저역통과 smoothing을 적용합니다.
4. 미래로 갈수록 time constant 1.25 s / 1.5 s로 지수 감쇠합니다.
5. 60 Hz로 2초 trajectory를 만듭니다.

상대 정책, 상대 controller, 미래 action은 읽거나 가정하지 않습니다.

## CEM 최적화

Action 한 knot는 다음 4개입니다.

```text
roll, pitch, rudder, throttle
```

현재 설정은 0.5초 knot 4개이므로 후보 하나의 shape은 `[4, 4]`입니다.
CEM은 48개 후보를 두 번 평가하고 상위 1/6로 mean과 std를 갱신합니다.

무작위 후보는 `noise`와 `-noise`를 쌍으로 만들어 좌우 표본 편향을 줄입니다.
추가로 다음과 같은 일반 후보만 넣습니다.

- 직전 최적 mean
- 직전 실제 action 유지
- 수평 pull
- 완화·감속
- 좌우 대칭 bank-then-pull 세기 3종

특정 상대의 행동을 감지해 분기하는 후보는 없습니다.

## 독립 C++ predictor

`native/reduced_predictor`는 대회 DLL을 링크하거나 호출하지 않습니다.

- F-16 XML에서 생성한 aerodynamic force/moment 식과 표
- 질량·관성·형상 계수
- F100 추력 표
- ISA atmosphere
- quaternion rigid-body dynamics
- midpoint integration
- FCS PID/actuator/engine spool 근사

를 자체 구현합니다. XML 두 파일은 계수의 provenance 확인과 runtime asset
계약을 위해 함께 둡니다.

## Cost 함수

각 60 Hz 예측 step의 utility는 개념적으로 다음과 같습니다.

```text
dt × (
  + damage_dealt_weight × predicted_damage_dealt
  - damage_taken_weight × predicted_damage_taken
  + attack_geometry
  + control_zone
  + closure_utility
  + nose_advantage
  - enemy_threat
  - far_range
  - overshoot
  - ground_risk
  - envelope_risk
)
- control_slew
```

Horizon 마지막에는 attack, control-zone, nose advantage, closure, threat를
조합한 terminal geometry를 더합니다.

핵심 항의 의미는 다음과 같습니다.

- `attack_geometry`: ATA 0°가 유일한 최대인 smooth potential
- `control_zone`: 작은 ATA와 현재 phase의 유효 사거리 안쪽을 동시에 선호
- `nose_advantage`: `enemy ATA - own ATA`의 상대 우위
- `threat_geometry`: 상대 ATA가 작고 12,000 ft 안쪽일수록 큰 위험
- `closure`: 원거리 접근, 근거리 약 20 m/s closure 선호
- `overshoot`: 1,800 ft 안에서 과도한 closure 억제
- `ground/envelope`: 저고도, sink, 저속·과속, 큰 AoA·beta 억제

정확한 식은 `reduced_predictor.cpp`의 `damageRate`, `attackPotential`,
`threatPotential`, `controlZonePotential`, `closureUtility`, `scoreStep`,
`terminalScore`에 있습니다.

## 적용과 fallback

최고 후보의 첫 knot 명령만 0.1초 적용하고 다시 계획합니다. Native 계산이
실패하거나 action이 non-finite이면 직전 유효 action을 유지하며
`PlanResult.fallback=True`를 기록합니다. 정상 검증에서는 fallback이 0회였습니다.
