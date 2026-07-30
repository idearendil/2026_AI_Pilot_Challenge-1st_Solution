# claude_code — DogFight 1v1 PPO 학습 패키지 (팀 매뉴얼)

원본 DogFight 프레임워크의 **환경(`DogFightWrapper` / `TierGatedDogFightEnv`)은 그대로**
쓰되, RLlib 없이 **순수 PyTorch 로 PPO 를 직접 구현**한 독립 패키지입니다. 이 문서는
"어떻게 학습을 돌리고, 학습한 모델끼리 붙여서 리플레이 로그를 뽑아 보는가"를 처음
보는 사람이 따라 할 수 있게 정리한 매뉴얼입니다.

> 모든 명령은 **Release 디렉토리(=이 폴더의 부모)** 에서 실행합니다.

---

## 1. 빠른 시작

```
# (1)-1 phase 1 학습(reward 식에 distance 항 추가됨)  — 결과 번들이 artifacts/models/team01/ppo_phase1 에 저장됨
python claude_code/train.py --output-name ppo --output-tag pbrs2
# (1)-2 phase 2 학습(reward 식에 distance 항 제거됨)  — 결과 번들이 artifacts/models/team01/ppo_phase2 에 저장됨
(아직 phase2 학습을 하고 나서도 서로 damage를 잘 주진 못함...)
python claude_code/train.py --output-name team01 --output-tag ppo_phase2 --resume-from claude_code/models/team01/ppo_phase1/iter_0150.pt --distance-reward-scale 0
# (1)-3 last 모델이 아닌 특정 iteration의 모델을 번들로 저장하고 싶으면
python claude_code/snapshot_to_bundle.py --snapshot claude_code/models/latest/iter_0754.pt --output-dir artifacts/models/team01/basic

# (2) 두 모델 대결 + 리플레이 로그 저장
(rl 모델과 bt 모델을 사용하는 경우)
python claude_code/run_local_dogfight.py --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic --target-backend bt --target-bt-dll Lee_BT1.dll --max-engage-time 200 --episode-step-limit 12000 --save-log --seed 0
(rl 모델과 rl 모델을 사용하는 경우)
python claude_code/run_local_dogfight.py --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic --target-backend rl --target-bundle-dir artifacts/models/team01/basic --max-engage-time 200 --episode-step-limit 12000 --save-log --seed 0

# (2)-2 어느 모델이 강한지 통계로 판정 (리플레이 없이 100판 병렬, rl=항상 stochastic, 판마다 랜덤 시드)
python claude_code/power_test.py --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic --target-backend bt --target-bt-dll Shin_BT1.dll --games 100
(번들 vs 번들)
python claude_code/power_test.py --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic --target-backend rl --target-bundle-dir artifacts/models/team01/old --games 100

# (3) 리플레이 로그는 artifacts/logs/ 아래 *_ownship_*.csv / *_target_*.csv / *_summary.json 로 저장됨 (Tacview 포맷)
=> 리플레이 로그 파일 3개를 모두 logs/ 폴더로 옮기고 아래 명령어 실행 후 브라우저에 접속해서 확인
python tools/web_log_viewer.py --logdir logs --port 7870
```

---

## 2. 파일 구성 — 어떤 게 무슨 역할인가

| 파일 | 역할 |
|---|---|
| **`train.py`** | **PPO 학습 entrypoint (여기서 학습을 시작).** CLI 옵션 파싱 → 학습 → 번들 저장 |
| `ppo.py` | 단일 프로세스 PPO 트레이너 (GAE + clipped surrogate) |
| `parallel.py` | Ray 기반 **병렬 rollout** 트레이너/워커 (`--num-workers>1` 일 때 사용) |
| `model.py` | **정책 네트워크**(`MLPDiscreteActorCritic`) + 2-파일 번들 저장/로드 |
| **`my_reward.py`** | **[편집] 보상 함수** (`MY_REWARD_CONFIG`, `compute_reward`) |
| **`my_observation.py`** | **[편집] 관측 벡터(34-D)** + 제출환경 state 재구성(`StateReconstructor`) |
| `env_utils.py` | 원본과 동일한 환경 생성(`make_env`, `TierGatedDogFightEnv`) + 보상/관측 hook |
| `self_play.py` | 상대를 (다른) actor network 로 조종하는 `SelfPlayProvider` |
| `action_provider.py` | 학습한 정책을 원본 `ActionProvider` 계약으로 감싼 추론 어댑터 |
| `normalizers.py` | 관측 running mean/std 정규화(`RunningMeanStd`) |
| `evaluation.py` | snapshot 저장/로드 + past-self 대결 평가(`play_games`) |
| **`run_local_dogfight.py`** | **모델 vs (모델·스크립트·DLL) 로컬 대결 + 리플레이 로그 생성** |
| **`power_test.py`** | **두 모델을 N판(기본 100) 병렬 대결시켜 어느 쪽이 강한지 통계 판정**(Wilson CI + 이항검정, 좌우 시작 50/50 교대, rl=항상 stochastic·판별 랜덤 시드) |
| `bt_rule.py` | BT DLL 의 rule XML(`AIP_RULE_XML`) 지정/검증 — **다른 claude_code import 보다 먼저** 써야 함 |
| **`snapshot_to_bundle.py`** | 매 iter snapshot(`iter_NNNN.pt`) → 제출용 2-파일 번들 변환 |
| `evaluate.py` | 학습한 번들을 로컬에서 간단 검증 |
| `submission.py` | 대결 서버 UDP 제출 (원본 `my_submission.py` 와 동일 경로) |
| `verify_reconstruction.py` / `verify_tier_damage.py` | 재구성값·tier damage 검증 스크립트 |

**학습 결과물 위치**
- 번들(제출용): `artifacts/models/<name>/<tag>/{metadata.json, policy_weights.pkl.gz}`
  - `<tag>` = past-self 승률 > 0.5 인 **best iteration**, `<tag>_final` = **맨 마지막 iteration**
- 학습 로그: `artifacts/logs/<name>/<tag>/ppo_training_log.csv`
- 매 iter snapshot: `claude_code/models/<name>/<tag>/iter_NNNN.pt` (재현·리플레이 상대용)

---

---

## 3. 바꿀 수 있는 주요 옵션 (train.py)

| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--iterations` | 150 | 학습 iteration 수 |
| `--rollout-steps` | 100000 | iteration 당 수집 step 수(전체 worker 합) |
| `--lr` | 1e-4 | actor 학습률 (1e-3 이상은 이 환경에서 붕괴 위험) |
| `--gamma` | 0.97 | 할인율 |
| `--gae-lambda` | 0.95 | GAE λ |
| `--clip-coef` | 0.2 | PPO 클립 계수 |
| `--update-epochs` | 4 | 수집 batch 당 업데이트 epoch 수 |
| `--minibatch-size` | 512 | 미니배치 크기 |
| `--ent-coef` | 0.0001 | 엔트로피 보너스(탐험) |
| `--target-kl` | 0.05 | approx_kl 초과 시 epoch 조기 종료 |
| `--hidden` | 512,512,512 | actor MLP hidden 크기 |
| `--activation` | tanh | 활성화(`tanh`/`relu`/`elu`) |
| `--action-bins` | 7 | 행동 채널당 이산 카테고리 수(균등 분할) |
| `--critic-hidden` / `--critic-activation` / `--critic-lr` | (actor 와 동일) | critic 전용 구조·학습률(별개 네트워크) |
| `--critic-epochs` | (`--update-epochs` 와 동일) | critic 전용 epoch 수. **critic 루프는 actor 루프와 분리**돼 `--target-kl` 조기 종료의 영향을 받지 않는다 |
| `--num-workers` | 물리 코어 수 | Ray 병렬 worker 수. 1 이면 단일 프로세스 |
| `--device` | cpu | driver update 디바이스(큰 모델은 `cuda`, worker 는 항상 CPU) |
| `--no-normalize-obs` | (off) | 관측 정규화 끄기 |
| **보상/관측** | | |
| `--reward-module` | claude_code.my_reward | 보상 모듈(빈 값이면 프레임워크 기본 보상) |
| `--observation-module` | claude_code.my_observation | 관측 모듈(빈 값이면 tactical16) |
| `--shaping-reward-scale` | None(→0.0001) | 거리·조준 통합 포텐셜 shaping 계수. 0 이면 끔 |
| **상대/phase** | | |
| `--self-play` / `--no-self-play` | self-play on | 상대를 학습 중 정책으로 / 스크립트 상대로 |
| `--target-mode` | loiter | `--no-self-play` 일 때 스크립트 상대(`loiter`/`fixed`/`behavior_tree`/`autopilot`) |
| `--resume-from` | "" | snapshot(.pt) 에서 actor+critic+obs_rms 이어받기(phase2) |
| `--frozen-opponent` | (off) | self-play snapshot 상대를 학습 시작 시점 정책으로 **고정** |
| `--pool-size` | 6 | opponent pool 최대 크기(**BT 포함**). 기본 = BT 1 + snapshot 5 |
| `--selfplay-gate-threshold` | 0.6 | **snapshot 후보들**의 min-EMA 가 이 값 이상이면 현재 정책을 pool 에 추가 |
| **평가/저장** | | |
| `--eval-interval` | 5 | 평가 주기(iter). N iter 전 self 상대 승률>0.5 면 best 번들 저장 |
| `--eval-games` | 20 | 평가 1회당 대결 판 수 |
| `--output-name` / `--output-tag` | team01 / ppo_mlp_v1 | 번들·로그 저장 경로 이름 |

> `loiter`(선회·고도유지, 자기파괴 없음)가 학습 신호가 깨끗해 스크립트 상대 기본값입니다.
> `fixed`/`autopilot` 은 표적이 스스로 추락해 무승부로 끝나므로 학습 시연엔 부적합합니다.

---

## 4. 현재 보상 식 (`my_reward.py`)

한 step 의 총 보상 = **damage + distance + aim + terminal** 4개 성분의 합입니다.

**계수(`MY_REWARD_CONFIG`)**: `win_reward=+10`, `loss_reward=-10`, `damage_scale=10`,
`distance_reward_scale=0.001`, `aim_reward_scale=0.0`(→ CLI 기본 0.5), `aim_range_m=3000`.

1. **damage (주 보상)** — 양측 HP>0 인 매 step:
   `r_damage = (표적 HP감소 × 1.0 − 내 HP감소 × 0.0) × 10`
   → 사실상 **표적에 입힌 damage rate × 10** (내 피해는 가중치 0). step 당 대략 0 ~ +1.0.

2. **distance (거리 접근 shaping)** — 직전 RL-step 대비 거리가 `d[m]` 줄면:
   `r_distance = d × 0.001` (멀어지면 음수). phase2 에서 `--distance-reward-scale 0` 로 끔.

3. **aim (조준 dense shaping, potential-based)** — `--aim-reward-scale` 로 켬(기본 0.5):
   포텐셜 `Φ = (1+cos ATA)/2 × clip(1 − dist/3000, 0, 1)` 로 두고
   `r_aim = (Φ_now − Φ_prev) × aim_scale`.
   **텔레스코핑**이라 episode 총합이 `±aim_scale` 로 bounded → damage 보다 항상 작은
   **보조 보상**으로만 작용(최적 정책을 바꾸지 않음, reward hacking 방지).

4. **terminal (종료)** — `terminated` 일 때:
   상대 격추/최소고도 이탈 → `+10`, 내 격추/최소고도 이탈 → `−10`.

> **HP-tiebreak 는 보상이 아니라 평가에서만**: 무승부(timeout) 시 HP 가 낮은 쪽이 패로
> 판정됩니다(`evaluation.py`). 보상 식에는 들어가지 않습니다.

---

## 5. 관측 벡터 (34-D, `my_observation.py` / `claude34r`)

각도는 **sin/cos 분리**, 상대 위치 Δ는 **signed-log**(`sign(x)·ln(|x|+1)`, 정규화 없음),
속도는 **스칼라 속력 + 3축(NED)** 를 함께 제공. 나머지는 대체로 `[-1,1]` 정규화.

| idx | feature | idx | feature |
|---|---|---|---|
| 0–1 | roll sin/cos | 17–18 | aspect angle sin/cos |
| 2–3 | pitch sin/cos | 19–20 | LOS azimuth sin/cos |
| 4–5 | yaw sin/cos | 21–22 | LOS elevation sin/cos |
| 6 | speed (TAS, 0–600) | 23 | target HP |
| 7–9 | 내 속도 vel N/E/D (NED 재구성) | 24 | 내가 입히는 damage rate |
| 10 | altitude (1000–15000m) | 25 | 내가 받는 damage rate |
| 11 | 내 HP | 26 | pursuit score |
| 12–14 | 상대위치 ΔN/E/D (signed-log) | 27 | 표적 speed |
| 15–16 | ATA(antenna train angle) sin/cos | 28–30 | 표적 속도 N/E/D (NED 재구성) |
| | | 31 | slant range (0–2000m) |
| | | 32 | closure rate (접근 +, 이탈 −) |
| | | 33 | aim_sharp = `exp(−(ATA/3°)²)` (1–3° 콘 고분해능) |

> **핵심**: 대결 서버 추론에서는 위치·자세·속도(state 0~8)만 들어오고 속도(KCAS)·고도·HP·
> WEZ 는 0 입니다. `StateReconstructor` 가 고도=`−D`, 속력=`‖u,v,w‖`, HP=damage 공식 누적,
> 기하=`geo_info` 로 **복원**하므로 학습·추론의 관측이 일치합니다. 3축 속도는 raw 성분
> (학습=body / 추론=world 프레임 불일치) 대신 **속력+자세로 NED 재구성**해 train/test 를 맞춥니다.

---

## 6. 모델 구조 (`MLPDiscreteActorCritic`)

- **이산(discrete) 정책**: 행동 4채널(roll / pitch / yaw · rudder / throttle) 각각을
  `num_bins`(기본 7)개 균등 분할 카테고리로 두고, **채널별 독립 Categorical** 로 샘플.
  (최대 엔트로피 = 4·ln 7 ≈ 7.78.)
- **actor**: `obs(34) → MLP(512,512,512, tanh) → 4×7 로짓`.
- **critic**: **actor 와 파라미터를 공유하지 않는 별개 MLP** `obs(34) → … → 1`,
  **별도 optimizer** 로 독립 업데이트. 구조·학습률도 `--critic-*` 로 따로 설정 가능.
- 이산 카테고리 index → `[-1,1]` 연속 행동값으로 변환 후, roll/pitch/rudder 는 clip,
  throttle 은 `(a+1)/2` 로 `[0,1]` 변환(원본 `DogFightEnv._to_sim_action` 과 동일).
- 저장은 원본과 같은 **2-파일 번들**: `metadata.json`(구조·관측모듈·정규화 통계) +
  `policy_weights.pkl.gz`(state_dict).

---

## 7. 학습 로그 읽는 법

`ppo_training_log.csv` 컬럼 + iteration 출력:

- `return` / `len` : 평균 episode 합산 보상 / 길이(탐험 on rollout 기준)
- `damage` / `dist` / `aim` : 보상 성분별 평균(어느 신호로 배우는지 확인)
- `ent` : 정책 엔트로피(탐험 정도), `kl` : approx_kl(trust region)
- `ev` : value function explained variance(가치함수 학습 지표; 보상이 희소하면 낮음)
- `ep aN/M cK/L` : 이번 iter 의 update 가 실제로 돈 epoch 수. `a`=actor(`N`/상한 `M=--update-epochs`),
  `c`=critic(`K`/상한 `L=--critic-epochs`, 미지정이면 `M`). actor 쪽 `*` 는
  `approx_kl > --target-kl` 로 **조기 종료**됐다는 뜻이다.
  **actor 와 critic 은 별도 루프**라 critic 은 KL 조기 종료와 무관하게 항상 `L` 번 다 돈다.
  CSV 컬럼 `update_epochs`(actor) / `critic_epochs` / `update_early_stop`,
  wandb `update/actor_epochs`, `update/critic_epochs`, `update/actor_epochs_frac`,
  `update/kl_early_stop`.
- `EVAL vs iterN` : `--eval-interval` 마다 **탐험 끈 결정론적 정책**으로 N iter 전 self 와 대결한 승률/전적. **승률>0.5 면 best 번들로 저장**.

---

## 8. 지금까지 적용한 아이디어 정리

**관측 설계**
- 각도 **sin/cos 분리**(순환성 보존), 상대위치 Δ **signed-log**(정규화 없이 부호+로그 압축)
- **NED 속도 재구성**(속력+자세) — 학습(body)/추론(world) 프레임 불일치 제거 → train/test 일치
- 제출환경 미제공 항목(속도·고도·HP·WEZ) **StateReconstructor 로 복원** → 34-D 전부 유효
- damage 조준 보강 feature: 표적 속도, 명시 slant range, closure rate, 고분해능 `aim_sharp`

**보상 설계**
- 표적에 입힌 damage 중심(내 피해 가중치 0), 종료 ±10
- **거리 접근 shaping**(phase1) + **potential-based 조준 shaping**(텔레스코핑 → damage 에 종속, 최적 정책 불변)
- HP-tiebreak 는 보상이 아니라 **평가 판정**에서만

**학습 구조**
- 순수 PyTorch PPO(GAE + clipped surrogate), **actor/critic 완전 분리**
  (별 네트워크·별 optimizer·**별 update 루프**). `--target-kl` 조기 종료는 actor 에만 적용되고
  critic 은 `--critic-epochs` 만큼 항상 다 돈다(가치 회귀 타깃은 rollout 시점에 고정돼 있어
  정책의 trust region 과 무관하기 때문).
- **이산 정책**(채널별 Categorical, 7 bins) — 연속 가우시안 대비 안정적 탐험
- 관측 running mean/std 정규화(통계 번들 저장 → 추론 동일 적용), per-minibatch advantage 정규화, approx_kl 조기 종료
- **Ray 병렬 rollout**(worker=물리 코어 수, driver 가 weights/obs_rms broadcast)
- **opponent pool self-play**(기본, 최대 6): slot0 = **baseline BT(`AIP_DCS_baseline.dll`) 고정 후보**,
  slot1.. = 학습 snapshot(오래된→최신). 후보별 EMA 승률을 각각 관리하고 EMA 가 낮은 후보를
  softmax(-ema/τ) 로 더 자주 샘플링한다. pool 이 가득 차면 **가장 오래된 snapshot** 을 제거하고
  BT 는 절대 evict 하지 않는다. pool 추가 게이트(min-EMA)는 **snapshot 후보만** 보고 판단한다
  (BT 는 매우 강해 게이트에 넣으면 세대 진행이 영구히 멈춤). BT DLL/rule 은
  `bt_rule.DEFAULT_BT_DLL`/`BT_RULE_DEFAULTS` 로 고정(CLI 옵션 없음).
  - ⚠️ **rule XML 캐싱 gotcha**: BT DLL 이 읽는 `AIP_RULE_XML` 은 `JSBSimAIPLib.dll` 이
    로드되는 시점(= `claude_code.env_utils` import 체인)에 **한 번만** 캐싱된다. 그래서
    `train.py` 는 다른 claude_code import 보다 **먼저** `bt_rule.apply_rule_env()` 를 호출하고,
    Ray worker 에는 `ray.init(runtime_env={"env_vars": ...})` 로 **프로세스 시작 시점에** 주입한다.
    늦게 세팅하면 DLL 이 `Rule_forTraining.xml`(트리 = `Task_Empty`)로 폴백해 **표적이 조종을
    전혀 안 하고 직진만 한다** → BT 상대 승률이 거짓으로 100% 가까이 찍힌다.
    `make_bt_provider()` 가 `bt_rule.check_rule_applied()` 로 이 상황을 즉시 에러로 잡는다.
- `--frozen-opponent` 로 snapshot 상대 고정
- **2-phase 학습**: phase1(거리 shaping) → phase2(`--resume-from` 이어받아 거리 off + 고정 상대)
- **WEZ 3-tier 시간 게이팅**(tier1 항상 / tier2 100s / tier3 150s, 콘 1°/2°/3° 고정)을 학습 env(`TierGatedDogFightEnv`)에도 반영해 대결 서버 규칙과 일치
- best-model saving: 결정론적 평가 승률>0.5 iteration 만 번들로 저장(마지막 출렁임 방지)

**제거/정리한 것** (실험 후 불필요/해로워서 삭제): LR 선형 감쇠, WEZ 콘 커리큘럼,
value-function 클리핑, 보상 스케일링(RewardScaler), env 의 max_separation 무승부/리셋.
