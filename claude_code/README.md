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
| `my_observation.py` | **[편집] 관측 벡터** + 제출환경 state 재구성(`StateReconstructor`) |
| `verify_reconstruction.py` | 학습 env 에서 재구성값 vs 실제 state 검증 |
| `verify_tier_damage.py` | 학습 env 의 시간 게이팅 3-tier damage 적용 검증 |
| `model.py` | MLP actor-critic(가우시안 정책) + 2-파일 번들 저장/로드 + action 변환 |
| `ppo.py` | GAE + clipped surrogate 기반 순수 PyTorch PPO 트레이너 |
| `parallel.py` | Ray 기반 병렬 rollout 수집 (`ParallelPPOTrainer`, `physical_cpu_count`) |
| `train.py` | 학습 entrypoint (CLI). 종료 시 번들 저장 |
| `action_provider.py` | MLP 정책을 원본 `ActionProvider` 계약으로 감싼 어댑터 |
| `self_play.py` | 상대를 같은 actor network 로 조종하는 `SelfPlayProvider` |
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

### 상대(self-play) — 기본값

`train.py` 는 **기본적으로 self-play** 입니다: 상대 전투기를 **학습 중인 같은 actor
network** 로 조종합니다(`self_play.SelfPlayProvider`). 정책이 향상되면 상대도 같이
강해지는 완전한 self-learning 입니다.

- 상대는 자신의 `StateReconstructor`(상대 관점 HP)로 **대칭 관측**을 만들고, model 을
  통과시켜 action 을 냅니다. `action_repeat=step_ratio` 로 본 기체와 동일한 제어 주기.
- 상대는 결정론적(평균 action)으로 동작합니다(`SelfPlayProvider(explore=True)` 로 변경 가능).
- **시작 위치 랜덤화**: `STANDARD_ENV_CONFIG["randomize_start_side"]=True`(기본) — 매 episode
  학습 agent 시작 위치를 두 위치(config `ownship`/`target`) 중 랜덤 선택(좌우 교대)해
  한쪽에 과적합하지 않게 합니다. 끄려면 `randomize_start_side=False`.
- 끄려면 `--no-self-play` (그러면 아래 `--target-mode` 스크립트 상대 사용).

```powershell
... claude_code\train.py --output-name team01 --output-tag selfplay_v1     # 기본 self-play
... claude_code\train.py --no-self-play --target-mode loiter ...           # 스크립트 상대
```

### 스크립트 상대 모드 (`--target-mode`, `--no-self-play` 일 때만)

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

### 병렬 데이터 수집 (Ray) — 기본값

큰 모델 + GPU 학습을 대비해, CPU-bound 한 env stepping 을 여러 프로세스(Ray actor)로
병렬 수집합니다. **worker 수 기본값 = 물리 CPU 코어 수**(논리 코어 아님; psutil 없으면
OS 조회 → 마지막엔 logical//2). 각 worker 는 자신의 env(self-play 포함)+로컬 model 을
갖고, 매 iteration 마다 driver 가 weights/obs_rms 를 broadcast → 병렬 수집 → driver 에서
update(GPU 가능).

```powershell
... claude_code\train.py --num-workers 6 ...      # 명시 (기본은 물리 코어 수)
... claude_code\train.py --num-workers 1 ...      # 단일 프로세스(Ray 미사용)
... claude_code\train.py --device cuda ...         # 큰 모델: driver update 를 GPU 로
```

측정(이 머신, 물리 6코어, 3072 step/iter, self-play+claude16): 단일 **~15.5s/iter** →
6-worker **~3.4s/iter (약 4.5×)**. worker 는 CPU 추론, driver 만 `--device cuda` 로 GPU
update. `--num-workers 1` 이면 Ray 없이 단일 프로세스 경로를 씁니다.

> 주의: obs_rms 는 driver 가 authoritative 로 관리(worker batch 통계를 병합), 각 worker 는
> broadcast 된 obs_rms 로 정규화. reward 스케일러는 worker 별로 유지됩니다(분포 동일해 수렴).

### 보상/관측을 claude_code 에서 직접 정의 (기본값)

`train.py` 는 **기본적으로 `claude_code/my_reward.py` 와 `claude_code/my_observation.py`
를 사용**합니다(플래그 불필요). 이 두 파일을 편집하면 학습 보상/관측이 바뀝니다.

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\train.py `
  --output-name team01 --output-tag v1          # 그냥 실행하면 my_reward + my_observation 사용
```

프레임워크 기본 보상(`src/dogfight/envs/reward.py`)·tactical16 관측을 쓰려면 빈 값을 줍니다:

```powershell
... claude_code\train.py --reward-module "" --observation-module "" ...
```

> 현재 `my_reward.py`:
> - **종료**: 상대 격추/고도이탈 → +10, 내 격추/고도이탈 → −10
> - **보조(매 step, 양측 생존 중)**: `(상대 HP감소 − 내 HP감소) × 10` (피해 유도 shaping)
>
> 긴 episode(200s) 에서는 rollout 당 episode 수가 적어지므로 `--rollout-steps` 를
> 크게(예: 4096) 두는 것을 권장합니다.

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

### 제출환경 state 재구성 (HP·고도·속도)

대결 서버 추론에서는 위치·자세·속도(state 0~8)만 들어오고 HP·고도·속도(KCAS)·WEZ 는
0 입니다. `my_observation.py` 의 `StateReconstructor` 가 이를 복원합니다:

- **고도** = `-D` (= `-state[2]`)  ← 위치
- **속도(TAS)** = `||(u,v,w)||` (= `||state[6:9]||`)  ← 속도
- **거리/ATA/AA/LOS** = `geo_info` 계산  ← 위치·자세
- **HP** = 매 RL-step `rate(r,theta,t) * DT_PER_STEP` 누적  ← 대결 서버 damage 공식
  (`damage_rate()`. r=거리[ft], theta=ATA)

**시간 게이팅** (대결 서버 규칙, `damage_rate` 에 반영):

| tier | 범위 | 각도 | 계수 | 활성화 |
|---|---|---|---|---|
| 1 | 500~3000 ft | ±1° | 1.0·(3000−r)/2500 | 0s~ (항상) |
| 2 | 500~3500 ft | ±2° | 0.3·(3500−r)/3000 | **100s 부터** |
| 3 | 500~4000 ft | ±3° | 0.1·(4000−r)/3500 | **150s 부터** |

**delta_t 보정**: 학습 env 의 `update_damage` 는 `계수 × env._delta_t(=1/60)` 를 매 sim
sub-step 적용하고, RL-step = `step_ratio(=6)` sub-step 이므로 **`DT_PER_STEP = 6/60 =
0.1 s`** (코드+실측으로 추출). 재구성도 `HP -= rate × 0.1` 로 동일하게 시간적분하므로,
tier-1 구간에서 **재구성 HP 가 env real HP 와 근접**(검증: 평균차 ≈ 0.044).

### 학습 env 의 시간 게이팅 3-tier damage (`TierGatedDogFightEnv`)

**원본 프레임워크 env**(`single_agent_env.py`)는 `wez` 가 `__init__` 에서 한 번 설정 후
안 바뀌어 **항상 tier-1 만** 적용합니다(시간 게이팅 없음 — 코드 + 실측으로 확인).
매뉴얼의 시간 게이팅(tier2 100s, tier3 150s)은 대결 서버 규칙입니다.

그래서 `claude_code/env_utils.py` 의 **`make_env` 는 기본적으로 `TierGatedDogFightEnv`**
(원본을 상속해 `update_damage` 만 오버라이드)를 사용해, 대결 서버와 동일한 **시간 게이팅
3-tier damage** 를 학습 env 에 적용합니다(`my_observation.damage_rate` 와 동일 공식).
`max_engage_time=200s` 라 episode 가 100s/150s 를 넘어 tier-2/3 가 실제로 발생합니다.
원본 단일-tier 로 비교하려면 `make_env(..., time_gated_damage=False)`.

검증(`verify_tier_damage.py`, 결정론적): tier2 기하(r=3200ft,ATA=1.5°)는 t=50s→0,
t=120s→0.0005; tier3 기하(r=3800ft,ATA=2.5°)는 t=120s→0, t=160s→0.000095. base env 는
이 기하에서 항상 0(단일-tier). tier env 값은 공식 × delta_t 와 정확히 일치.

`build_observation` 은 이 재구성값으로 16-D 관측을 만들어, 학습·검증·제출에서 16개
feature 가 **모두 유효**합니다(기존엔 5개가 추론에서 -1 고정이었음).

HP 누적은 **RL-step 당 정확히 1회** 갱신되어야 하므로(`advance_reconstructor`), 학습은
`ppo.py`, 추론은 `MLPActionProvider.compute_action` 에서 호출하고 `build_observation`
은 읽기만 합니다. 따라서 학습·평가·제출의 누적 빈도가 일치합니다(검증: recon 관측
번들의 학습 중 결정론적 평가 return 과 `evaluate.py` return 이 동일).

**검증 명령**:

```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\verify_reconstruction.py
```

학습 env 실측 결과: 고도 평균오차 ≈ **2.3 m**(7000m 중), 속도 평균오차 ≈ **0.03 m/s**,
표적 HP 재구성 vs env real HP 평균차 ≈ **0.044** → 위치·자세·속도만으로 거의 정확히
복원됨(`DT_PER_STEP` 보정으로 tier-1 HP 도 근접). 제출 서버는 시간 게이팅된 3-tier 라
재구성값이 그쪽과 맞습니다.

> 좌표계 주의: 고도는 학습 NED(D=아래 양수) 기준 `-D` 입니다. 라이브 서버가 z 를
> "고도(위 양수)"로 준다면 `my_observation.ALT_SIGN` 을 `+1.0` 으로 바꾸세요.

학습:
```powershell
D:\other_programs\anaconda3\envs\aip\python.exe claude_code\train.py `
  --observation-module claude_code.my_observation `
  --output-name team01 --output-tag recon_v1
```

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
