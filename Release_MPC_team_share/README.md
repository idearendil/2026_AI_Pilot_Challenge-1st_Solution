# Release_MPC 팀 공유본

이 압축파일은 공중전 대회의 **MPC guidance + 직접 조종명령 제어기**만
분리한 최소 개발 번들입니다. 강화학습 모델이나 상대별 분기 규칙은 없으며,
공개 패킷의 ownship/target 상태만 사용합니다.

현재 기본 버전은 `v7`입니다.

- 시뮬레이션: 60 Hz
- MPC 재계획: 10 Hz
- 예측 horizon: 2.0 s
- control knot: 0.5 s × 4개
- 탐색: CEM 48 candidates × 2 iterations
- 출력: roll / pitch / rudder / throttle

## 포함된 것

```text
configs/mpc.yaml                 최종 MPC 설정과 cost weight
student/my_submission.py         대회 UDP 실행 진입점
src/mpc/                         Python MPC, CEM, target predictor
src/dogfight/                    필요한 최소 UDP/action interface
native/reduced_predictor/        독립 C++ F-16 predictor 소스
runtime/predictor/Release/       바로 실행 가능한 predictor DLL
aircraft/f16/f16.xml             모델 계수 출처 XML
engine/F100-PW-229.xml           엔진 계수 출처 XML
tests/test_mpc_core.py            최소 회귀 테스트
tools/build_native.cmd           C++ predictor 재빌드
docs/                            구조, 빌드, 검증 설명
```

다음은 의도적으로 제외했습니다.

- 학습 코드, PPO checkpoint, W&B 파일
- 평가 로그와 튜닝 후보 설정
- pure-pursuit Baseline DLL
- 대회 공식 JSBSim DLL
- CMake build tree와 Python cache
- 중단된 benchmark 산출물

따라서 이 번들만으로 공식 로컬 대전 환경을 재현할 수는 없지만, MPC 소스를
읽고 수정하거나 predictor를 빌드하고 단위 테스트하며 대회 서버에 접속할 수
있습니다.

## 1. 빠른 시작

Windows PowerShell에서 압축을 푼 폴더로 이동합니다.

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python student\my_submission.py --help
```

대회 서버 실행 예시는 다음과 같습니다.

```powershell
python student\my_submission.py `
  --server-ip <SERVER_IP> `
  --server-port 9999 `
  --team-name <TEAM_NAME> `
  --config configs\mpc.yaml
```

동일 값은 환경변수 `DOGFIGHT_SERVER_IP`, `DOGFIGHT_SERVER_PORT`,
`DOGFIGHT_TEAM_NAME`으로도 지정할 수 있습니다.

## 2. 입력과 출력 계약

대회 네트워크에서 각 항공기에 대해 받는 공개 상태는 9개입니다.

```text
[north_m, east_m, down_m,
 roll_deg, pitch_deg, yaw_deg,
 body_u_mps, body_v_mps, body_w_mps]
```

두 항공기를 합쳐 18개 공개 값만 직접 사용합니다. 각속도와 상대 가속도는
과거 공개 프레임으로 causal하게 추정합니다.

출력은 다음 순서입니다.

```text
[roll_cmd, pitch_cmd, rudder_cmd, throttle_cmd]
[-1, 1], [-1, 1], [-1, 1], [0, 1]
```

프로토콜 필드 이름은 `yaw_cmd`지만 물리적으로는 rudder 명령입니다. 별도의
yaw-angle 추종 명령이 아닙니다.

## 3. 실행 흐름

```text
공개 18D 상태
  → Euler history로 own/target p,q,r 추정
  → target의 2초 궤적 예측
  → CEM으로 48개 조종 시퀀스 생성
  → 독립 C++ predictor에서 60 Hz rollout 및 cost 계산
  → 최고 시퀀스의 첫 0.1초 명령 적용
  → 다음 공개 상태에서 다시 계획
```

상세 수식과 좌표계는 [ARCHITECTURE.md](docs/ARCHITECTURE.md), 빌드와 수정
방법은 [BUILD_AND_INTEGRATION.md](docs/BUILD_AND_INTEGRATION.md), 성능 수치와
한계는 [VALIDATION.md](docs/VALIDATION.md)를 참고하십시오.

## 4. 주로 수정할 파일

- Cost와 horizon: `configs/mpc.yaml`
- CEM 후보 생성·warm start: `src/mpc/planner.py`
- 상대 궤적 예측: `src/mpc/target_prediction.py`
- 패킷/명령 연결: `src/mpc/command_policy.py`
- 6DoF와 FCS 근사·cost 수식: `native/reduced_predictor/reduced_predictor.cpp`

C++ 구조체의 필드 순서가 바뀌면 `reduced_predictor.h`와
`src/mpc/native.py`를 반드시 함께 수정해야 합니다.

## 5. 중요한 한계

- Predictor는 대회 JSBSim DLL 복제본이 아니라 XML 계수를 사용한 독립 근사
  모델입니다.
- 서버 내부 actuator, PID integral, engine spool state는 직접 관측하지
  못하므로 명령 이력과 reset 가정으로 추정합니다.
- 상대의 미래 조종입력은 알 수 없어 최근 운동을 감쇠 외삽합니다.
- 긴 open-loop 예측은 오차가 누적됩니다. 현재 실험에서는 4초보다 2초
  horizon이 훨씬 안정적이었습니다.
- 특정 benchmark에서 나온 승률을 임의의 상대에 대한 보장으로 해석하면 안
  됩니다.

## 6. 무결성

압축파일의 `MANIFEST_SHA256.txt`에는 포함된 모든 파일의 SHA-256이 들어
있습니다. 공유 후 파일 누락이나 변조 여부를 확인할 때 사용하십시오.
