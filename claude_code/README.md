# claude_code — 독립형 PPO 학습 & 제출

원본 DogFight 프레임워크와 **완전히 동일한 환경**(`DogFightWrapper`)을 그대로
사용하되, RLlib 의존성 없이 **순수 PyTorch로 PPO를 직접 구현**한 패키지입니다.
다양한 강화학습 아이디어를 빠르게 실험할 수 있도록 학습 루프 전체가 읽기 쉬운
한 곳에 모여 있습니다.

학습 결과는 원본과 같은 **2-파일 번들**(`metadata.json` + `policy_weights.pkl.gz`)로
저장되며, `submission.py`가 원본 `student/my_submission.py`와 동일한 UDP 클라이언트
경로로 대결 서버에 연결합니다. 즉 대결 서버 입장에서는 원본 RL 제출과 완전히
동일하게 동작합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `env_utils.py` | 원본 YAML(`student_ppo_mlp.yaml`)과 동일한 환경 세팅으로 `DogFightWrapper` 생성 + 보상/관측 모듈 hook |
| `my_reward.py` | **[편집] 보상 함수** (`MY_REWARD_CONFIG`, `compute_reward`) |
| `my_observation.py` | **[편집] 관측 벡터** (`OBSERVATION_SIZE`, `build_observation`) |
| `model.py` | MLP actor-critic(가우시안 정책) + 2-파일 번들 저장/로드 + action 변환 |
| `ppo.py` | GAE + clipped surrogate 기반 순수 PyTorch PPO 트레이너 |
| `train.py` | 학습 entrypoint (CLI). 종료 시 번들 저장 |
| `action_provider.py` | MLP 정책을 원본 `ActionProvider` 계약으로 감싼 어댑터 |
| `submission.py` | 대결 서버 연결 (원본 my_submission.py 와 동일한 경로) |
| `evaluate.py` | 학습한 번들을 로컬 환경에서 검증 |

## 환경 세팅 (원본과 동일)

`env_utils.STANDARD_ENV_CONFIG` 는 `experiments/student_ppo_mlp.yaml` 의
`env` / `env_config` 와 동일합니다.

- 관측: `tactical16` (16차원, `[-1, 1]` 정규화)
- 행동: `Box([-1, 1]^4)` = roll, pitch, rudder, throttle
- `step_ratio: 6`, `max_engage_time: 60s`, `episode_step_limit: 3600`
- 보상/종료: 원본 `src/dogfight/envs/reward.py`, `termination.py` 그대로 사용

동역학·보상·종료·관측 파이프라인이 RLlib 경로와 100% 동일하고, **유일한 차이는
학습 루프(RLlib → claude_code PPO)** 입니다.

### 표적 모드 (`--target-mode`)

`train.py` 의 학습 데모 기본값은 **`loiter`** 입니다. 이유:

| 표적 | 특성 | 학습 시연 적합성 |
|---|---|---|
| `loiter` | 선회하며 고도 유지(자기파괴 없음) | **적합(기본).** episode 가 timeout 으로 끝나(terminal=0) return 이 ownship 의 추격/사격 성과로만 결정됨 |
| `fixed` / `autopilot` | 스스로 하강·추락 | 부적합. 매 episode 표적이 추락해 무승부(-30)로 끝나 return 이 정책 품질과 무관하게 ~-30 에 고정 |
| `behavior_tree` | 실제 대회형 강한 상대 | 고난도. 처음부터 학습은 매우 어려움 |

> 참고: `STANDARD_ENV_CONFIG` 자체는 원본 YAML 과 동일하게 `target_mode: fixed` 를
> 담지만, `train.py` 는 학습 신호가 깨끗한 `loiter` 를 기본으로 사용합니다.
> `--target-mode` 로 언제든 바꿀 수 있습니다.

## 1) 학습

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\train.py `
  --iterations 150 `
  --rollout-steps 2048 `
  --target-mode loiter `
  --output-name team01 `
  --output-tag ppo_mlp_v1
```

주요 옵션: `--lr`(기본 3e-4), `--gamma`, `--gae-lambda`, `--clip-coef`,
`--update-epochs`, `--minibatch-size`, `--ent-coef`, `--hidden 256,256`,
`--activation tanh|relu|elu`, `--log-std-init`, `--target-mode`,
`--eval-interval`(기본 5), `--eval-episodes`(기본 2).

> 학습률은 3e-4 부근이 안정적입니다. 1e-3 이상은 이 환경에서 policy 가
> trust region 을 크게 벗어나(approx_kl 폭증) 붕괴합니다.

학습 로그: `artifacts/logs/<name>/<tag>/ppo_training_log.csv`
번들: `artifacts/models/<name>/<tag>/{metadata.json, policy_weights.pkl.gz}`

iteration 마다 다음이 출력됩니다:

- `return`: 평균 episode 합산 보상 (탐험 on 상태의 rollout 기준)
- `len`: 평균 episode 길이 (생존을 배우면 timeout=600 에 수렴)
- `pursuit` / `damage`: 추격/사격 보상 성분 (추격을 배우면 상승)
- `ev`: value function explained variance (가치함수 학습 지표)
- `EVAL`: `--eval-interval` 마다 **탐험을 끈 결정론적 정책**으로 평가한 return/length

### 최고 성능 정책 저장 (best-model saving)

학습 후반에는 탐험 noise 가 커지며 *rollout* return(탐험 on)이 출렁일 수 있습니다.
실제 제출에 쓰는 것은 **탐험을 끈 결정론적 정책**이므로, 본 트레이너는
`--eval-interval` 마다 결정론적 평가를 수행해 **그 시점까지의 최고 정책만 번들로
저장**합니다. 따라서 마지막 iteration 이 출렁여도 저장된 번들은 항상 최고 성능
스냅샷입니다. (환경 초기 상태가 결정적이라 적은 episode 로도 정책 품질을 대표합니다.)

### 검증된 학습 결과 (loiter, 120 iters, seed 0)

이 환경에서 PPO 가 실제로 학습됨을 확인했습니다 (CPU, 약 13분):

| 지표 | 학습 전(랜덤 정책) | 학습 후(저장된 best 정책) |
|---|---|---|
| 결정론적 평가 outcome | crash (471 step) | **timeout (600 step, 추락 없음)** |
| 결정론적 평가 return | ≈ **-29.9** | ≈ **+3.0** |
| value EV | ~0 | **0.95 ~ 0.98** |
| `pursuit` 성분 | ~0 | **+5 ~ +6 (양수)** |

즉 에이전트는 (1) 추락하지 않고 비행하는 법, (2) episode 끝까지 생존하는 법,
(3) 표적을 추격하는 법을 스스로 학습했습니다.

**학습 ↔ 제출 경로 일치 검증:** 학습 중 결정론적 평가 return 과
`evaluate.py`(= `MLPActionProvider` 제출 경로, `action_repeat=step_ratio`)의 return 이
**+3.004 로 정확히 일치**합니다. 즉 학습 때와 대결 서버 추론 때 정책이 동일하게
동작합니다.

> 제어 주기 주의: 정책은 RL action 1회를 `step_ratio(=6)` sim step 동안 유지하도록
> 학습됩니다. 제출(`submission.py`)은 `ProviderCommandPolicy(action_repeat=6)`,
> 로컬 검증(`evaluate.py`)은 내부 `_ActionRepeatProvider(repeat=6)` 로 이 주기를
> 맞춥니다. 이 값을 어기면(매 sim step 마다 정책 호출) 동작이 달라져 추락합니다.

### 빠른 수렴을 위한 정규화

순수 PPO 는 정규화 없이는 매우 느리게 학습합니다. 본 구현은 아래 표준 기법을 기본으로
켜며, 필요하면 `--no-normalize-obs` / `--no-scale-reward` / `--no-anneal-lr` 로 끌 수
있습니다:

- 관측 running mean/std 정규화 (통계는 번들에 저장 → 추론에서 동일 적용)
- 할인 누적 보상 std 기반 보상 스케일링 (학습 전용)
- 학습률 선형 감쇠
- advantage 정규화, clipped value loss, `approx_kl` 기반 epoch 조기 종료

### 보상/관측을 claude_code 에서 직접 정의

보상·관측은 환경(`src/dogfight/`)이 계산하지만, **claude_code 안에서 바로 바꿀 수**
있습니다. 두 편집 파일이 있고(`claude_code/my_reward.py`, `claude_code/my_observation.py`),
모듈 경로로 활성화합니다(기본값은 두 파일 모두 원본 동작과 동일하게 맞춰져 있어,
활성화만 해서는 결과가 바뀌지 않습니다):

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\train.py `
  --reward-module claude_code.my_reward `
  --observation-module claude_code.my_observation `
  --output-name team01 --output-tag custom_v1
```

- **보상**(`my_reward.py`): `MY_REWARD_CONFIG`(계수) + `compute_reward(...) -> (total, components)`.
  학습에만 영향(추론/제출에는 무관). 계수만 바꾸려면 `env_utils.STANDARD_ENV_CONFIG["reward"]`
  를 수정해도 됩니다.
- **관측**(`my_observation.py`): `OBSERVATION_SIZE` + `build_observation(...)`. 학습·로컬검증·
  제출이 **모두** 같은 함수를 사용합니다. 사용한 모듈 경로는 번들 `metadata.json` 에
  기록되고, `evaluate.py`/`submission.py` 가 이를 읽어 동일 관측을 재구성하므로 학습과
  추론 입력이 일치합니다. 관측 **차원**을 바꾸면 기존 번들과 호환되지 않으니 재학습해야
  합니다.

> 검증: custom 보상+관측으로 학습한 번들에 대해, 학습 중 결정론적 평가 return 과
> `evaluate.py` return 이 동일하게 나오는 것을 확인했습니다(학습↔추론 관측 일치).

## 2) 로컬 검증

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\evaluate.py `
  --bundle-dir artifacts\models\team01\ppo_mlp_v1 `
  --episodes 5 --target-mode fixed
```

## 3) 대결 서버 제출

`submission.py` 상단의 `TEAM_NAME`, `SERVER_IP`, `BUNDLE_DIR` 를 설정 후:

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\submission.py
```

`ACTION_REPEAT=6` 은 학습 `step_ratio=6` 과 맞춘 값으로, 원본 제출과 동일하게
6개 PlaneInfo pair마다 정책을 한 번 호출합니다.

## 번들 형식

```text
artifacts/models/<name>/<tag>/
├── metadata.json          # 관측 모드, MLP 구조(hidden/activation), throttle 변환 규칙 등
└── policy_weights.pkl.gz  # gzip 압축된 MLPActorCritic.state_dict()
```

원본 RLlib 번들은 `policy_weights.pkl.gz` 에 RLModule state 를 담지만, claude_code
번들은 PyTorch `state_dict` 를 담습니다. `submission.py` 가 이 형식을 직접
읽으므로 대결 서버 연결에 RLlib 가 전혀 필요 없습니다. (따라서 원본
`run_unreal_inference.py` / `run_local_dogfight.py` 의 RLlib backend 로는 직접
로드되지 않으며, claude_code 의 `submission.py` / `evaluate.py` 를 사용합니다.)

## action 변환 (학습 ↔ 추론 일치)

정책은 4차원 raw 값을 출력하고, 환경과 추론 양쪽에서 동일하게 변환됩니다
(`model.policy_action_to_command`, 원본 `DogFightEnv._to_sim_action` 와 동일):

- roll, pitch, rudder → `[-1, 1]` clip
- throttle → `(a + 1) / 2` 로 `[0, 1]` 변환

## 알아둘 점: 학습 ↔ 추론 관측 차이 (프레임워크 공통)

이 차이는 claude_code 만의 문제가 아니라 **원본 RL 제출도 동일하게 겪는**
프레임워크 고유 특성입니다.

- 학습 시 관측은 JSBSim 전체 상태(속도 KCAS, 고도, Health, WEZ 등 포함)로
  `tactical16` 을 만듭니다.
- 대결 서버 추론 시 관측은 `PlaneInfo`(위치/자세/속도 9개 값)만으로 구성되므로,
  `tactical16` 중 KCAS·ALT·HEALTH·target HEALTH·WEZ 플래그 항목은 기본값(정규화 후
  대부분 `-1`)이 됩니다. 이는 원본 `ProviderCommandPolicy` + `plane_info_to_state`
  경로를 그대로 사용하기 때문입니다.

상대 기하(ATA/AA/LOS, 상대 위치)는 양쪽에서 동일하게 채워지므로 추적 정책의 핵심
신호는 보존됩니다. 이 항목들에 과도하게 의존하지 않는 관측/정책 설계가 전이에
유리합니다.
