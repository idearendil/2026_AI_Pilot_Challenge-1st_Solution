# Stable_MPC 팀 공유본

이 ZIP은 2026-08-05에 동결한 **native CEM MPC + predictive safety**
컨트롤러의 실행용 최소 번들입니다. `release_mpc_team_share.zip`의 기본 MPC는
그대로 유지하고, 검증된 `lazy + predictive` 안전 선택기를 최종 행동 선택에
추가했습니다.

실험용 robust ensemble, DP/value guidance, score residual, stratified sampling,
평가 로그, GitHub Actions 파일, 테스트와 빌드 산출물은 포함하지 않습니다.

## 동결 설정

| 항목 | 값 |
|---|---:|
| 시뮬레이션 | 60 Hz |
| 재계획 | 10 Hz |
| 예측 horizon | 2.0 s |
| control knot | 0.5 s × 4 |
| CEM | 48 candidates × 2 iterations |
| 상대 예측 | 관측 가속도·선회율 감쇠 외삽 |
| 안전 검사 | `lazy` + `predictive` |
| 최소 예측 여유고도 | 380 m |
| pull-out 가속도 가정 | 28 m/s² |
| robust / DP / residual | OFF |

`configs/mpc.yaml`과 `configs/safe_mpc.yaml`이 이 값을 명시합니다. 팀 공유 후
성능을 재현하려면 두 파일을 변경하지 마십시오.

## 빠른 실행

필수 환경은 Windows x64, Python 3.10 이상입니다. 압축을 푼 폴더에서:

```powershell
python -m pip install -r requirements.txt
python student\my_submission.py --help
```

대회 서버 접속:

```powershell
python student\my_submission.py `
  --server-ip <SERVER_IP> `
  --server-port 9999 `
  --team-name <TEAM_NAME>
```

필요하면 설정 경로를 명시할 수 있습니다.

```powershell
python student\my_submission.py `
  --mpc-config configs\mpc.yaml `
  --safety-config configs\safe_mpc.yaml
```

서버 주소·포트·팀명은 `DOGFIGHT_SERVER_IP`, `DOGFIGHT_SERVER_PORT`,
`DOGFIGHT_TEAM_NAME` 환경변수로도 지정할 수 있습니다.

## 입력과 출력

각 항공기의 공개 상태 9개만 사용합니다.

```text
[north_m, east_m, down_m,
 roll_deg, pitch_deg, yaw_deg,
 body_u_mps, body_v_mps, body_w_mps]
```

각속도는 공개 Euler 이력에서 causal하게 추정합니다. 출력은 다음 순서입니다.

```text
[roll_cmd, pitch_cmd, rudder_cmd, throttle_cmd]
[-1, 1], [-1, 1], [-1, 1], [0, 1]
```

프로토콜의 `yaw_cmd` 필드는 yaw-angle 목표가 아니라 rudder 명령입니다.

## 실행 흐름

1. 공개 ownship/target 상태와 직전 명령을 구성합니다.
2. 상대의 2초 궤적을 관측 이력만으로 예측합니다.
3. native predictor에서 48개 후보를 2회 평가합니다.
4. CEM elite·mean·std는 기본 MPC와 똑같이 갱신합니다.
5. 점수가 높은 후보부터 380 m 고도 조건과 종단 pull-out 가능성을 검사합니다.
6. 첫 안전 후보의 첫 0.1초 명령을 적용합니다.
7. 안전 후보가 하나도 없으면 날개 수평화 또는 최대 pull-up 복구 명령을 냅니다.

안전 검사는 CEM 표본, elite, warm start와 RNG를 바꾸지 않습니다. 최종 후보만
후처리합니다.

## 파일 구성

```text
configs/                 동결된 MPC·safety 설정
student/                 대회 UDP 실행 진입점
src/mpc/                 기존 native CEM MPC
src/safe_mpc/            predictive safety와 UDP 연결
src/dogfight/            필요한 최소 action/UDP 인터페이스
runtime/predictor/       실행용 Windows x64 predictor DLL
aircraft/, engine/       predictor runtime 자산
CHANGES_FROM_RELEASE_MPC_V7.md
MANIFEST_SHA256.txt      파일 무결성 목록
```

## 검증 결과와 해석

- seed 0–8, 양 진영 18경기에서 기존 native MPC 상대 `11승 7패`, 평균 체력차
  `+0.0592`
- 우리 기체 추락 0회, planner fallback 0회
- 최종 동결 재현 4경기 `3승 1패`, 평균 체력차 `+0.3408`
- 평균 계획시간 `21.0–27.2 ms`
- seed 0 target에서 OS/runner성 단발 최대 지연 `682.7 ms` 1회 관측

18경기는 여전히 작은 표본이며, 특정 상대·CPU·서버 지연에 대한 승률 보장은
아닙니다. 첫 ZIP과의 상세 차이는
`CHANGES_FROM_RELEASE_MPC_V7.md`를 참고하십시오.

## 무결성

PowerShell에서 다음 명령으로 개별 파일 해시를 확인할 수 있습니다.

```powershell
Get-FileHash -Algorithm SHA256 .\runtime\predictor\Release\MPCJSBSim.dll
```

전체 기준값은 `MANIFEST_SHA256.txt`에 있습니다.
