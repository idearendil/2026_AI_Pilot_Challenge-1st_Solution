# 검증 결과와 해석 한계

## 현재 선택값

최종 v7은 generic defensive cost와 2초 horizon을 사용합니다. 특정 상대의
ID나 행동 패턴을 조건문으로 넣지 않았습니다.

```yaml
horizon_seconds: 2.0
damage_dealt: 140.0
damage_taken: 180.0
attack_geometry: 3.5
control_zone: 9.0
closure: 1.0
threat_geometry: 12.0
nose_advantage: 9.0
far_range: 4.0
overshoot: 6.0
ground: 120.0
envelope: 6.0
terminal_geometry: 28.0
control_slew: 0.03
```

## RL champion 상대

상대는 고정 deterministic PPO checkpoint였고 모든 평가는 공식 3-9의
North/South mirror를 동일 seed로 paired 실행했습니다.

완전히 사용하지 않은 20-game holdout 결과:

| 설정 | W-L-D | score rate | 평균 HP margin | 평균 plan p95 |
|---|---:|---:|---:|---:|
| 이전 v6, 3초 | 8-11-1 | 42.5% | -0.214 | 35.3 ms |
| 현재 v7, 2초 | **9-10-1** | **47.5%** | -0.281 | **29.8 ms** |

현재 설정이 한 게임 앞섰지만 차이는 작고 HP margin은 더 나빴습니다. 따라서
챔피언에 대한 명확한 우위를 입증했다고 말할 수 없습니다.

Tuning에 관여한 두 seed bank까지 합친 60전에서는 v7이 32-23-5, 이전 v6가
24-31-5였지만 이 수치는 비편향 holdout 승률이 아닙니다.

또한 holdout의 v7 승리 9회 중 8회는 상대 저고도 이탈, 1회만 직접 격추였습니다.
대회 승리 조건에는 유효하지만 순수 무장 운용 우위로 해석하면 안 됩니다.

## Pure-pursuit Baseline 회귀

사용하지 않은 seed 8개를 양쪽 mirror로 실행한 16전 결과:

```text
13승 1패 2무
score rate 87.5%
평균 damage margin +0.602
직접 target destruction 10회
planner fallback 0회
```

RL champion 튜닝 때문에 기존 pure-pursuit Baseline 성능이 붕괴한 증거는
없었습니다.

## Horizon 실험

동일 cost와 동일 20게임에서:

| Horizon | W-L-D | score rate | HP margin | plan p95 |
|---:|---:|---:|---:|---:|
| 2초 | 11-8-1 | 57.5% | -0.066 | 27.4 ms |
| 3초 | 11-8-1 | 57.5% | -0.161 | 37.6 ms |
| 4초 | 3-16-1 | 17.5% | -0.584 | 45.3 ms |

4초의 실패는 더 먼 미래 자체가 나빠서라기보다, 같은 48후보/2회 CEM으로
8개 knot를 찾는 난이도와 reduced-model 오차 누적이 합쳐진 결과로 보는 것이
타당합니다.

## 공유 시 주의사항

- 결과는 한 RL checkpoint와 한 Baseline 구현에 대한 유한 표본입니다.
- 다른 상대 pool, 서버 지연, 다른 CPU에서는 다시 측정해야 합니다.
- `compute_budget_ms=80`은 hard real-time 중단이 아니라 계획시간 profile 감시
  값입니다. 현재 fixed candidate profile은 자동으로 후보 수를 줄이지 않습니다.
- 공식 simulator DLL은 이 번들에 없으므로 predictor parity/공식 로컬 교전
  평가는 원래 개발 환경에서 수행해야 합니다.
