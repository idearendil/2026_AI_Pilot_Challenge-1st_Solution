# model2_gylee — 학습 계보와 실행 계약

## 1. 이 모델의 정확한 의미

`model2_gylee`는 2026-08-13 패키징 시점에 활성 `ver09` 학습에서 가장 최근에
완전히 저장되어 있던 `iter_1415.pt`입니다.

| 항목 | 값 |
|---|---|
| checkpoint | `model/iter_1415.pt` |
| global environment steps | 141,500,000 |
| SHA-256 | `6A12ECD587A51627C0A67504579092382CC3FFE00C7F554345D825035C0E16CF` |
| observation RMS count | 149,000,000.0001 |
| 상태 | 활성 학습의 durable snapshot |
| final-best 선정 여부 | 선정하지 않음 |

패키징 당시 학습은 iter1418까지 완료되어 iter1419 rollout을 계속하고 있었고,
checkpoint는 5 iteration 간격이므로 iter1415가 마지막으로 완전히 저장된 파일이었습니다.
학습 프로세스를 중단하거나 변경하지 않고 immutable한 iter1415 파일을 복사했습니다.

`model1_gylee`의 iter1220은 ver05 후보 리그로 선정된 champion입니다. 반면 이 파일은
다른 계보인 ver09의 더 최근 learner이므로 “iteration 숫자가 더 크다 = 검증된 best”라고
해석하면 안 됩니다. 둘을 pool에 함께 두면 stable champion과 recent learner라는 서로
다른 역할을 갖습니다.

## 2. 전체 학습 계보

```text
ver01 Phase1 iter75 (7-bin)
  → ver06 curriculum, local iter1~380 (7-bin)
  → ver08 iter380에서 7→19-bin 보존 변환
  → ver08 broad/HP-conditioned ATA60 학습
  → ver08 durable iter815
  → ver09 outcome-first branch, critic 1회 reset
  → iter880에서 draw -40→-80
  → iter1075에서 past20 dynamic league
  → iter1220에서 category hard-PFSP
  → ver09 iter1415 (이 패키지)
```

### 2.1 ver06 시작점과 curriculum

ver06은 `ver01 Phase1 iter75`의 7-bin actor, critic과 observation RMS를 이어받고
optimizer/RNG는 새로 시작했습니다.

- observation: 47D
- actor/critic: 분리 512-512-512 tanh MLP
- action: roll/pitch/rudder/throttle 4축 × 7 bins
- PPO: gamma 0.999, GAE 0.97, LR `1e-4`, entropy `2e-4`
- epochs 4, minibatch 512, target KL 0.05
- official 3-9, 10 Hz policy, 8 rollout workers

Phase2는 BT40%와 자체 league60%, Phase3는 BT30%/MPC30%/자체 league40%, Phase4는
ver05 best1220 60%/MPC30%/BT10% 구조였습니다. 각 gate와 내부 opponent 선택은 draw를
승리로 세지 않는 actual-win hard PFSP를 사용했습니다. ver06 iter380 champion이 다음
계보의 seed가 되었습니다.

### 2.2 ver08: 19 bins와 ATA60 실험

ver06 iter380의 기존 7개 action 위치를 정확히 유지하는 nested-grid 방식으로
19 bins로 확장했습니다. Actor/critic과 RMS는 보존했습니다.

ver08은 60° cone의 기하 shaping을 사용했고, iter620부터 HP 우열에 따라 공격
shaping을 조정했습니다. iter815에서는 HP 동률 또는 열세일 때 ownship이 lead를
잡도록 positive attack scale을 2배로 높였습니다. 이 단계까지는 outcome reward 외에도
비교적 강한 ATA guidance가 있었습니다.

Opponent pool은 BT/best/MPC의 staged fixed schedule을 사용했고, 최종 단계에서 frozen
ver05 best1220을 외부 기준 상대 중 하나로 사용했습니다. ver08 iter815가 ver09의
immutable seed입니다.

### 2.3 ver09: outcome-first 전환

ver08 iter815에서 19-bin actor, observation RMS, actor optimizer, RNG와 opponent-pool
상태를 보존했습니다. Reward scale과 value target 의미가 크게 달라져 critic과 critic
optimizer만 분기 시작 시 정확히 한 번 reset했습니다. 이후 resume에서는 critic도
정상적으로 계속 이어졌습니다.

PPO 설정:

| 항목 | 값 |
|---|---:|
| gamma | 0.999 |
| GAE lambda | 0.97 |
| learning rate | `1e-4` |
| entropy coefficient | `2e-4` |
| epochs | 4 |
| minibatch | 512 |
| action bins | 19 per axis |
| rollout workers | 8 |
| scenario | official 3-9 |

10 Hz에서 gamma 0.999의 e-folding horizon은 약 100초이며, 200초 뒤 reward도 약
13.5%가 남습니다.

## 3. ver09 reward

Reward의 목적은 dense damage farming보다 실제 대회 결과를 지배적으로 만드는 것입니다.

### 3.1 Outcome

- terminal 또는 timeout win: `+100`
- terminal 또는 timeout loss: `-100`
- draw: iter880까지 `-40`, iter881 이후 `-80`
- target ground/FDM forced loss: `+100`
- ownship ground loss: predictive safety와 terminal remainder를 합쳐 총 `-100`
- 직접 damage reward: 0
- destruction 별도 reward: 0
- distance reward: 0

따라서 iter1415는 draw `-80` 체제에서 학습된 모델입니다.

### 3.2 제한된 guidance

Sparse outcome만으로 credit assignment가 지나치게 어려워지는 것을 막기 위해 두 개의
bounded guidance만 남겼습니다.

1. HP-state hold: 200초 episode 최대 크기 ±5. 엄격한 HP 우세는 우세량과 무관하게
   같은 positive score를 갖고, 열세는 `-tanh(deficit_points/15)`로 완만히 벌점.
2. 대칭 ATA potential:

```text
q(a) = clip(1 - |a|/60°, 0, 1)
Phi = 5 × range_gate × (q(own ATA) - q(enemy ATA))
r = gamma × Phi(next) - Phi(previous)
```

Terminal에서는 Phi를 0으로 강제합니다. 보유 상태를 매 step 반복 보상하거나 기하 cycle로
reward를 수확하기보다, geometry 변화에 대한 potential-based guidance를 제공합니다.
Range gate는 실제 시간별 weapon range, 500 ft 최소거리와 1,000 ft 바깥 buffer를
사용합니다.

Predictive ground-risk는 1,750 ft부터 3초 예상고도를 사용하며 최대 -30입니다. 회복에
양의 보상을 지급하지 않습니다.

## 4. Opponent 분포 변화

### iter815~1075

Staged fixed pool을 사용했고 frozen best1220, MPC v7과 BT들을 상대했습니다.

### iter1076~1220

과적합 완화를 위해 다음 고정 category 분포로 변경했습니다.

```text
frozen selected best1220 50%
Release MPC v7           20%
BT 3종                   10%
learner history past     20%
```

Past는 recent 3개와 frontier 3개, 총 6 slot이었습니다. 20 iteration마다 현재 후보와
중간 checkpoint를 추가할 수 있고, recent는 최신 모델로 교체되며 frontier는 어려운
과거 상대를 hard PFSP로 보존했습니다. best1220은 promotion을 금지해 외부 기준점으로
고정했습니다.

### iter1221~1415

네 category 전체를 최근 20 iteration actual win rate로 hard PFSP했습니다.

```text
priority = 0.10 + 1 - actual_win_rate
actual_win_rate = W / (W + L + D)
```

- draw는 0승
- 5 iteration마다 갱신
- best1220/MPC/past/BT 중 어려운 category 비율을 즉시 높임
- BT 내부와 past frontier 내부 hard PFSP도 유지
- frozen best1220은 여전히 교체하지 않음

iter1415 update 후 기록된 다음 sampling weight는 대략 다음과 같습니다.

```text
best1220 25.3% / past 31.3% / BT 16.4% / MPC 26.9%
```

iter1415 rollout은 mixed stochastic training batch 53 episodes에서 W/L/D 35/12/6을
기록했습니다. 이 수치는 네 category가 섞인 단일 iteration 표본이지, deterministic
official 독립평가 승률이나 최종 best 선정 결과가 아닙니다.

## 5. 47D observation 계약

각 항공기의 공개 최소 9D state를 입력으로 사용합니다.

```text
[N, E, D, roll_deg, pitch_deg, yaw_deg, body_u, body_v, body_w]
```

- 좌표: NED(North+, East+, Down+)
- body: x 전방+, y 오른쪽+, z 아래+
- p/q/r: 연속 attitude의 SO(3) 차분으로 추정
- HP: 공개 기하와 time-gated damage rule을 episode 동안 적분한 추정값
- 각 environment/opponent마다 독립 StateReconstructor 필요

Feature 순서:

| index | 내용 |
|---:|---|
| 0–2 | own gravity body x/y/z |
| 3–6 | own speed, body u/v/w |
| 7–9 | estimated own p/q/r |
| 10–13 | AoA, sideslip, low-altitude margin, vertical speed |
| 14–18 | own HP, target speed/HP, HP difference, energy advantage |
| 19–21 | relative position in own body frame |
| 22–24 | relative velocity in own body frame |
| 25–26 | slant range, closure |
| 27–30 | sin/cos ATA and aspect angle |
| 31–34 | sin/cos LOS azimuth/elevation |
| 35–38 | own/enemy aim sharpness and active margins |
| 39–41 | near/far range margins, episode time |
| 42–44 | estimated target p/q/r |
| 45–46 | LOS azimuth/elevation rates |

Checkpoint의 running mean/variance를 반드시 적용합니다.

```text
z = (raw_obs - mean) / sqrt(var + 1e-8)
z = clip(z, -10, 10)
```

팀원의 learner RMS를 대신 적용하면 모델 계약이 깨집니다.

## 6. Actor/action 계약

```text
47D normalized observation
→ 512 tanh → 512 tanh → 512 tanh
→ 4 × 19 logits
→ roll/pitch/rudder/throttle 축별 categorical
```

각 축의 19 bins는 `[-1,1]` 균등 격자입니다. 선택된 index를 연속값으로 바꾼 뒤:

- roll/pitch/rudder: `[-1,1]`
- throttle: policy `[-1,1]`을 `(a+1)/2`로 `[0,1]` 변환

`explore=False`는 축별 argmax, `explore=True`는 categorical sample입니다. 고정 비교
opponent는 deterministic이 해석하기 쉽고, 학습 pool 다양성이 목적이면 stochastic을
선택할 수 있습니다.

## 7. 설치와 검증

ZIP을 팀원 DogFightEnv `Release` 루트에 해제합니다.

```text
Release/
├── src/dogfight/...
├── GeoMathUtil.py
└── model2_gylee/
```

```powershell
python -m model2_gylee.verify
```

검증기는 SHA-256, checkpoint 구조, 47D RMS, strict state load와 deterministic command
범위를 검사합니다. `.pt`는 pickle 기반이므로 신뢰할 수 없는 파일에는
`weights_only=False`를 사용하지 마십시오.

## 8. Opponent pool 연결

```python
from model2_gylee import make_opponent_provider

provider = make_opponent_provider(
    step_ratio=6,
    device="cpu",
    explore=False,
)
env._target_action_provider = provider
```

병렬 환경에서는 각 worker/env마다 factory를 별도로 호출해야 합니다. Episode reset마다
provider의 `reset()`이 호출되어야 하며, context에서는 opponent 자신의 state가
`ownship_state`, 학습 agent state가 `target_state`여야 합니다.

같은 프로젝트 snapshot loader를 이미 쓰는 경우 `model/iter_1415.pt`만 등록할 수도
있습니다. 다만 자체 provider가 47D feature 순서, iter1415 RMS, throttle remap,
action repeat 6과 상대 관점 reconstructor를 모두 동일하게 구현해야 합니다.

## 9. 패키지에서 제외한 것

- 실제 reward source와 iter1415 활성 설정은 재현용으로 `training_reward/`에 포함
- PPO optimizer/update 구현: 고정 opponent 행동 생성에는 불필요
- curriculum/pool coordinator: 팀원 자신의 pool을 사용
- JSBSim/DLL/XML: 팀원이 가진 동일 환경 사용
- W&B, 로그, MPC/BT, 과거 checkpoint: inference에 불필요

원본 `.pt` 안에는 critic, optimizer와 RNG 학습 상태도 보존되어 있지만, 제공 loader는
actor, model kwargs와 observation RMS만 사용합니다.

### 포함된 reward 원본을 사용할 때

```text
training_reward/
├── gyLee_reward.py
├── reward_config_ver09_iter1415.json
└── README.md
```

`gyLee_reward.py`는 ver09 source snapshot과 byte-identical하며 SHA-256은
`BF99FAA904B0ABBC2634E837C3A9BDFEA00722A6E2B528220A5B5C4763D8A603`입니다.
파일에는 이전 실험의 phase1/phase2도 함께 보존되어 있고 module-level 기본값은 기존
호환성을 위해 phase1입니다. 따라서 iter1415 재학습에서는 파일만 import하고 기본값을
쓰면 안 되며, 반드시 `reward_profile="phase3"` 또는 함께 제공한 JSON의 전체 설정을
학습 launcher에 전달해야 합니다.

원본 import 경로는 `claude_code.my_observation`입니다. 이는 실제 학습 당시 계약을
보존하기 위해 수정하지 않았습니다. 팀원의 환경에서 그대로 실행하려면 reward 파일을
팀원의 `claude_code` source 위치에 두거나, hook loader가 이 파일을 불러올 때 동일한
47D `my_observation.py`가 `claude_code.my_observation`으로 import 가능해야 합니다.
Opponent로만 사용할 때는 이 작업이 전혀 필요하지 않습니다.

## 10. 통합 오류 점검 순서

성능이 원본과 다르면 다음을 확인하십시오.

1. checkpoint SHA-256
2. `explore` 설정
3. iter1415 RMS 적용 여부
4. 47D feature 순서
5. opponent 관점 own/target state 순서
6. episode별 reconstructor reset
7. worker별 독립 provider
8. action repeat 6
9. 19-bin `[-1,1]` 매핑
10. throttle `(a+1)/2`
