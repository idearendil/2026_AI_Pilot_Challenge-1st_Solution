# `release_mpc_team_share.zip` 대비 변경점

비교 기준은 사용자가 제공한 2026-08-03 `Release_MPC_team_share.zip` 34개
파일과, 2026-08-05 동결 안정판(`228cc6aa0db82af56926ab9c3e630c2b8ad7809f`)
의 활성 설정입니다.

## 1. 실제 정책 동작의 변경

기본 MPC 자체는 바꾸지 않았습니다.

| 구성 | 첫 Release_MPC v7 | 현재 안정판 |
|---|---|---|
| CEM | 48후보 × 2회 | 동일 |
| 재계획 / horizon | 10 Hz / 2.0 s | 동일 |
| knot | 0.5 s × 4 | 동일 |
| 상대 예측 | 감쇠 가속도·선회율 | 동일 |
| cost weight 13개 | v7 값 | 전부 동일 |
| RNG seed | `20260802` | 동일 |
| CEM 결과 사용 | native 최고점 후보 | native 후보 중 첫 안전 후보 |
| 안전 후처리 | 없음 | `lazy + predictive` |

추가된 안전 판정은 다음과 같습니다.

1. native rollout이 valid이고 점수가 finite여야 합니다.
2. 2초 rollout 중 최소 고도가 380 m 이상이어야 합니다.
3. 2초 끝 상태에서 아래 pull-out margin이 0 이상이어야 합니다.

```text
margin = altitude
       - 380 m
       - sink_speed × roll_level_time
       - 0.5 × g × roll_level_time²
       - sink_speed² / (2 × 28 m/s²)

roll_level_time
  = max(0, |wrapped_roll| - 45°) / 1.6 rad/s
```

후보는 native 점수 내림차순으로 검사하고 첫 안전 후보에서 멈춥니다. 이것이
`lazy`의 의미입니다. 96개 후보를 모두 안전 rollout하던 초기 구현과 선택 행동,
CEM elite, mean/std는 같지만 평균 계획시간은 `37.37 ms → 21.01 ms`로
줄었습니다.

안전 후보가 없을 때의 복구 명령도 새로 추가됐습니다.

- bank가 45°보다 크면: 반대 roll `±1`, pitch `+0.28`, throttle `1.0`
- 그 외: roll `0`, pitch `-0.98`, throttle `1.0`

대회 프로토콜의 pitch 부호 때문에 `-0.98`이 강한 pull-up 명령입니다.

## 2. 바뀌지 않은 것

- 공개 ownship/target 상태만 사용합니다.
- 상대 ID, 상대별 규칙, 미래 상대 명령을 사용하지 않습니다.
- CEM 후보 생성·대칭 표본·structured 후보·elite 갱신·warm start를 안전
  판정이 건드리지 않습니다.
- 기본 공격/방어 cost weight는 모두 v7과 같습니다.
- 실행 DLL은 첫 ZIP과 동결 커밋에서 Git blob SHA
  `e2806c4354971aa71eab5edabae6edeee2afb7ee`로 완전히 같습니다.
- robust response, DP/Q value, residual correction, stratified CEM은 최종 정책에서
  모두 꺼져 있습니다.

따라서 안정판은 “새 MPC”라기보다 **v7 MPC의 최종 행동에 predictive safety
shield를 붙인 버전**입니다.

## 3. 성능 비교

직접 비교 가능한 native-MPC 대전은 seed 0–8, 양 진영 18경기입니다.

| 평가 | 안정판 결과 |
|---|---:|
| 전적 | `11승 7패` |
| 평균 체력차 | `+0.0592` |
| 우리 추락 | `0회` |
| planner fallback | `0회` |

초기 seed 0–1의 4경기는 `3승 1패, +0.3408`이었지만, seed를 0–8로 늘리자
우위가 크게 줄었습니다. 안정판이 첫 모델보다 약간 나은 결과이지만 18경기만으로
통계적으로 강한 우위를 단정할 수는 없습니다.

안전 판정이 모든 판을 개선하는 것도 아닙니다. 특정 동일 seed/진영에서는
safety OFF가 직접승 `+0.5909`, safety ON이 timeout 패 `-0.2081`이었습니다.
안전 shield는 단순 추락 방지기가 아니라 초기 궤적을 바꾸는 정책 요소입니다.

첫 ZIP 문서의 RL champion `9-10-1`과 pure-pursuit baseline `13-1-2`는 상대와
seed bank가 달라 현재 안정판 수치와 직접 비교하면 안 됩니다.

## 4. 개발 브랜치에 추가됐지만 채택되지 않은 코드

동결 브랜치에는 실험을 검증하기 위한 다음 기반 코드도 누적됐습니다.

- target/ownship prediction shadow 계측
- native batch/trajectory/debug rollout ABI
- score residual 필드와 stratified CEM sampler
- 좌·우 상대 반응 robust reranker
- safety full/emergency/altitude/risk-priced 변형
- candidate-selection ablation
- value-guidance와 6초 terminal DP
- 실험 workflow, 테스트, 결과 telemetry

이 경로들은 기본 설정에서 비활성이며, 실제 대전에서 안정판을 넘지 못했습니다.
현재 팀 공유 ZIP은 동결된 활성 경로만 남기기 위해 관련 모듈·설정·결과 파일을
제외했습니다.

## 5. 공유 ZIP 구조 변경

첫 ZIP의 34개 파일 중 실행에 필요하지 않은 다음 항목은 제거했습니다.

- `docs/ARCHITECTURE.md`, `BUILD_AND_INTEGRATION.md`, `VALIDATION.md`
  — 현재 README와 이 변경 문서로 통합
- `native/reduced_predictor/` C++ 빌드 소스 4개
  — 실행은 검증된 prebuilt DLL 사용
- `tests/test_mpc_core.py`
- `tools/build_native.cmd`, `tools/package_team_share.py`
- plain-MPC 전용 `src/mpc/provider.py`

다음은 새로 추가하거나 교체했습니다.

- `configs/safe_mpc.yaml`
- `src/safe_mpc/config.py`, `planner.py`, `provider.py`, `command_policy.py`
- `student/my_submission.py`를 안정판 safety 설정 두 개를 읽도록 변경
- `requirements.txt`에서 실행에 불필요한 `pytest` 제거
- `README.md`, `VERSION.txt`, `MANIFEST_SHA256.txt` 갱신

native C++ 소스를 수정하거나 DLL을 다시 빌드해야 하는 개발 작업은 전체 TopGun
저장소의 동결 브랜치에서 진행하고, 이 ZIP은 팀원의 실행·검토용으로 사용합니다.
