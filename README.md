# DogFight PPO — F-16 1v1 공중전 강화학습

JSBSim/DLL 기반 F-16 1v1 공중전 환경에서, **RLlib 없이 순수 PyTorch 로 PPO 를 직접 구현**한
self-play 학습 패키지입니다. 학습 코드는 모두 [`claude_code/`](claude_code/) 안에 있고,
환경(`DogFightWrapper` / `TierGatedDogFightEnv`)은 원본 프레임워크를 그대로 씁니다.

> 모든 명령은 이 저장소 루트(`Release/`)에서 실행합니다. 런타임 자산(DLL·XML·`aircraft/`·
> `engine/`·JSBSim script)은 이름 변경·이동·삭제하지 마세요.
>
> 이전 RLlib 기반 학습 프레임워크(`train_rllib.py`, `experiments/*.yaml`, `student/`) 매뉴얼은
> [`docs/rllib-framework.md`](docs/rllib-framework.md) 로 옮겨 보존했습니다. 더 자세한 팀
> 매뉴얼은 [`claude_code/README.md`](claude_code/README.md) 를 참고하세요.

---

## 1. 빠른 시작

### 학습
```bash
# 기본 학습(무한 iteration · self-play · Ray 병렬 · wandb 로깅). Ctrl-C 로 중단 가능.
python claude_code/train.py --output-name team01 --output-tag basic

# GPU 가 없으면 driver update 를 CPU 로
python claude_code/train.py --output-name team01 --output-tag basic --device cpu

# wandb 를 끄려면
python claude_code/train.py --output-name team01 --output-tag basic --no-wandb
```
- 학습은 **무한 반복**하며(`--iterations 0` 기본), 매 iteration 끝에 체크포인트를 저장하므로
  언제든 멈췄다가 이어서 학습할 수 있습니다(→ [§8 중단·재개](#8-중단재개)).
- 결과 번들은 `artifacts/models/<name>/<tag>/` 에 저장됩니다(제출·추론용 2-파일 번들).

### 대결·검증 (로컬)
```bash
# 두 모델(또는 모델 vs BT) 로컬 대결 + Tacview 리플레이 로그 저장
python claude_code/run_local_dogfight.py \
  --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic \
  --target-backend bt --target-bt-dll Lee_BT1.dll \
  --save-log --seed 0

# 어느 모델이 강한지 100판 병렬 통계 판정(Wilson CI + 이항검정)
python claude_code/power_test.py \
  --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic \
  --target-backend bt --target-bt-dll Shin_BT_best.dll --games 100
```

### 대결 서버 제출 (Unreal)
`claude_code/submission_a.py` / `submission_b.py` 의 `TEAM_NAME`·`SERVER_IP`·`BUNDLE_DIR` 을 맞춘 뒤 실행합니다.
```bash
python claude_code/submission_a.py    # 인스턴스 A
python claude_code/submission_b.py    # 인스턴스 B (로컬 self 대전 테스트용)
```

---

## 2. 학습이 어떻게 이뤄지는가

한 문장 요약: **Ray 로 병렬 수집한 rollout 으로 PPO 업데이트를 돌리고, 상대는 opponent
pool 에서 뽑은 self-play 정책·BT·MPC·exploiter 로 계속 갈아 끼우며, 주기적으로 난이도·보상
스케줄을 올린다.**

### 2.1 PPO 코어
- 순수 PyTorch PPO(GAE + clipped surrogate). **actor/critic 완전 분리**(별 네트워크·별
  optimizer·별 update 루프). `--target-kl` 조기 종료는 actor 루프에만 적용되고, critic 은
  `--critic-epochs` 만큼 항상 다 돕니다.
- 관측 running mean/std 정규화(통계를 번들에 저장 → 추론에서 동일 적용), per-minibatch
  advantage 정규화.
- **Ray 병렬 rollout**: worker = 물리 코어 수(기본). driver 가 매 iteration 정책 weights·
  obs_rms·스케줄을 broadcast 하고, worker 가 환경 step + GAE 까지 계산해 batch 를 돌려줍니다.

### 2.2 Opponent pool (self-play)
매 게임 상대를 pool 에서 뽑아 대전합니다. 구성:

| 슬롯 | 상대 | evict |
|---|---|---|
| 2 | **BT** — `Lee_BT1`, `Shin_BT_best` | 고정 (never-evict) |
| 1 | **MPC** — 팀 공유 planner | 고정 (never-evict) |
| n | **Exploiter** — 점점 증가 (→ [§2.4](#24-exploiter)) | 고정 (never-evict) |
| ≤3 | **Self-play snapshot** — 과거 자기 정책 | 초과 시 가장 오래된 것 FIFO 제거 |

- **샘플링**: 후보별 EMA 승률에 `softmax(−ema/τ)` (τ=0.3) → 우리 승률이 낮은(어려운) 상대를
  더 자주 뽑습니다. 단 확률의 절반은 균등 분배(초반 강한 상대 과편중 방지).
- **snapshot 승격**: snapshot 후보들의 최소 EMA 가 `--selfplay-gate-threshold`(0.6) 이상이면
  현재 정책을 pool 에 새 snapshot 으로 추가합니다.
- BT 는 프로세스당 rule 1개 제약이 있어, 실제 사용 BT 수 = min(2, worker 수)로 워커별
  round-robin 배정됩니다. ⚠️ BT rule XML 캐싱 gotcha 는 [`claude_code/README.md`](claude_code/README.md) 참고.

### 2.3 초기 배치 (매 판 랜덤, A:B = 4:1)
매 episode 두 시나리오 중 하나로 시작합니다. 고도·속도는 두 기체가 항상 **동일**(대칭 시작),
roll/pitch = 0(수평).

| | 확률 | 배치 | 거리 | heading |
|---|---|---|---|---|
| **A** | 0.8 | 같은 남–북 직선 위 | {2000, 2500, 3000} ft 중 랜덤 | 직선에 **수직·반대**(90° / 270°) — 마주보지 않음 |
| **B** | 0.2 | 정면 head-on | **10000 ft** | 서로 **마주봄**(0° / 180°) |

- 공통 랜덤 범위: 고도 **2000–30000 ft**, 속도 **200–300 m/s**.
- 매 판 달라지는 것: 시나리오(A/B) · 고도 · 속도 · (A) 거리·북남·좌우방향 / (B) 북남 위치.

### 2.4 Exploiter
main 학습이 `--exploiter-period`(**500**) iter 진행될 때마다, main 을 잠깐 멈추고 **그 시점의
main 을 유일한 상대로 삼아 새 exploiter 를 scratch(random init)로 학습**합니다.
- 종료: vs-main 승률 ≥ **70%**(`--exploiter-win-target`) 또는 **500 iter**(`--exploiter-max-iters`).
- 하이퍼파라미터: rollout **80,000** / lr **1e-4** / ent **5e-5** / **clip 0.4**(빠른 수렴).
- 학습이 끝나면 그 exploiter 의 actor 를 **pool 에 never-evict 로 고정 추가**하고 main 재개.
- wandb 는 exploiter 세션마다 **별도 run**(같은 group)에 독립 로깅됩니다.
- exploiter 학습도 **매 exploiter-iter 마다 체크포인트**를 저장해, 도중에 끊겨도 그 지점부터
  이어서 학습합니다(→ [§8](#8-중단재개)).

### 2.5 스케줄 (무한 학습, 1000 iter마다 단계 전환)
`k = (iter − 1) // 1000` 일 때 단계별로 값이 바뀝니다:

| 항목 | 식 | 1–1000 (k=0) | 1001–2000 (k=1) | 2001–3000 (k=2) | 3001– (k≥3) |
|---|---|---|---|---|---|
| rollout steps | 80,000 + 20,000·k | 80,000 | 100,000 | 120,000 | 140,000+ |
| learning rate | 1e-4 × (1/3)ᵏ | 1e-4 | 3.3e-5 | 1.1e-5 | ↓ |
| ent coef | 1e-4 × (1/3)ᵏ | 1e-4 | 3.3e-5 | 1.1e-5 | ↓ |
| **gamma** | 사다리 | 0.98 | **0.99** | **0.995** | **0.999** |
| **내 damage 가중치** | k≥2 부터 1.0 | 0.5 | 0.5 | **1.0** | 1.0 |
| **shaping 크기** | k≥2 부터 ×0.5 | ×1.0 | ×1.0 | **×0.5** | ×0.5 |

- 고정 하이퍼파라미터: clip 0.2, GAE λ 0.95, update epochs 5, minibatch 512, target_kl 0.05.
- 스케줄은 iteration 번호로만 결정되므로 재개해도 정확한 단계가 복원됩니다.

---

## 3. 모델 구조 (`MLPDiscreteActorCritic`)

- **이산(discrete) 정책**: 행동 4채널(roll / pitch / rudder·yaw / throttle)을 각각
  **21개 균등 bin**(`--action-bins`, 홀수라 가운데 bin = 중립 0)으로 두고, **채널별 독립
  Categorical** 로 샘플합니다.
- **Actor**: `obs(184) → MLP(512, 512, 512, tanh) → 4×21 = 84 logit`.
- **Critic**: actor 와 파라미터를 공유하지 않는 **별개 MLP** `obs(184) → 512³ → 1`, 별도 optimizer.
- **총 파라미터 ≈ 1.28M** (actor 663,124 + critic 620,545 = 1,283,669).
- 이산 index → `[-1,1]` 연속값 변환 후 roll/pitch/rudder 는 clip, throttle 은 `(a+1)/2` 로
  `[0,1]` 변환(원본 `DogFightEnv._to_sim_action` 과 동일).
- 저장은 **2-파일 번들**: `metadata.json`(구조·관측모듈·정규화 통계) + `policy_weights.pkl.gz`.

---

## 4. 관측 · 행동 · 보상

### 4.1 관측 — 184차원 (`my_observation.py`, `claude164r`)
서버가 주지 않는 HP·연료·각속도 등은 위치·자세·속도로부터 **재구성**해 채웁니다
(`StateReconstructor`). 구성:

| 블록 | 개수 | 내용 |
|---|---|---|
| 스칼라 | 50 | 자세(sin/cos)·속도·고도·연료·damage rate·pursuit score·절대고도 등 |
| 벡터 성분 | 114 | 7종 벡터 × **6 좌표계**(world·mybody·oppbody·myvel·oppvel·los), 38 vec × 3축 |
| 행동 히스토리 | 20 | 최근 5스텝 × 4채널 |

### 4.2 행동 — `Box([-1,1]⁴)` → 21 bin 이산화
| 축 | 의미 |
|---|---|
| 0 | roll |
| 1 | pitch |
| 2 | rudder / yaw |
| 3 | throttle (내부에서 `[0,1]` 변환) |

### 4.3 보상 — `my_reward.py`, 3요소 합
한 step 총 보상 = **damage + shaping + terminal**.

1. **damage (매 step, 양측 생존 시)**:
   `(상대 HP감소 × 1.0 − 내 HP감소 × own_w) × 10`.
   `own_w` = 0.5 → 학습 단계 k≥2(2001 iter~)에서 **1.0**(스케줄, [§2.5](#25-스케줄-무한-학습-1000-iter마다-단계-전환)).
2. **shaping (거리·조준 포텐셜 차분)**: `(Φ_now − Φ_prev) × 5e-5`. 거리가 가깝고(WEZ 안),
   내가 잘 조준하고 상대는 못 조준할수록 Φ↑. 텔레스코핑이라 episode 총합 bounded. 고도
   1000~3000ft 하강 억제항 포함. k≥2(2001 iter~)에서 크기 **절반**(스케줄).
3. **terminal (종료 시)**: HP 승/패 = 0 / 0(현재 미사용), **내 고도 최소치(≈300m) 이하 종료
   = −20**, **상대 고도 최소치 이하 종료 = +5**.

---

## 5. 주요 학습 옵션 (`train.py`)

| 옵션 | 기본값 | 기능 |
|---|---|---|
| `--iterations` | 0 | 총 iteration(0 이하 = **무한 학습**) |
| `--rollout-steps` | 80000 | iteration 당 수집 step(전체 worker 합, 스케줄로 증가) |
| `--lr` / `--gamma` / `--ent-coef` | 1e-4 / 0.98 / 1e-4 | 스케줄 시작값(→ [§2.5](#25-스케줄-무한-학습-1000-iter마다-단계-전환)) |
| `--clip-coef` / `--gae-lambda` | 0.2 / 0.95 | PPO 클립 / GAE λ |
| `--update-epochs` / `--minibatch-size` / `--target-kl` | 5 / 512 / 0.05 | 업데이트 루프 |
| `--hidden` / `--activation` / `--action-bins` | 512,512,512 / tanh / 21 | 네트워크 구조 |
| `--critic-hidden` / `--critic-activation` / `--critic-lr` / `--critic-epochs` | (actor 와 동일) | critic 전용 설정 |
| `--num-workers` | 물리 코어 수 | Ray worker 수(1 이면 단일 프로세스) |
| `--device` | cuda | driver update 디바이스(worker 는 항상 CPU) |
| `--sched-period` / `--sched-rollout-increment` / `--sched-lr-decay` / `--sched-ent-decay` | 1000 / 20000 / ⅓ / ⅓ | 단계 스케줄 |
| `--exploiter-period` / `--exploiter-max-iters` / `--exploiter-win-target` | 500 / 500 / 0.7 | exploiter 주기·종료 |
| `--exploiter-rollout-steps` / `--exploiter-lr` / `--exploiter-ent-coef` / `--exploiter-clip-coef` | 80000 / 1e-4 / 5e-5 / 0.4 | exploiter 하이퍼파라미터 |
| `--max-self-snapshots` / `--pool-size` | 3 / 7 | self-play snapshot 정원 / pool 총 슬롯 |
| `--mpc-opponent` / `--no-mpc-opponent` | on | MPC 고정 상대(self-play + workers>1 에서만 활성) |
| `--reward-module` / `--observation-module` | claude_code.my_reward / claude_code.my_observation | 보상·관측 모듈 |
| `--resume-state` / `--auto-resume` | "" / off | 전체 학습 상태 재개(→ [§8](#8-중단재개)) |
| `--wandb` / `--no-wandb` / `--wandb-project` | on / — / "AIP contest" | wandb 로깅 |
| `--output-name` / `--output-tag` | team01 / ppo_mlp_v1 | 저장 경로 이름 |

> wandb API 키는 `WANDB_API_KEY` 환경변수로 넣는 것을 권장합니다.

---

## 6. 출력물

- **번들(제출·추론용)**: `artifacts/models/<name>/<tag>/{metadata.json, policy_weights.pkl.gz}`
  - `<tag>` = past-self 승률 기준 **best iteration**, `<tag>_final` = **맨 마지막 iteration**
- **학습 로그**: `artifacts/logs/<name>/<tag>/ppo_training_log.csv`
- **매 iter snapshot**: `claude_code/models/<name>/<tag>/iter_NNNN.pt` (재현·리플레이 상대용)
- **전체 학습 상태 체크포인트**: `claude_code/models/<name>/<tag>/train_state.pt`
- 특정 iteration snapshot 을 번들로:
  `python claude_code/snapshot_to_bundle.py --snapshot <iter_NNNN.pt> --output-dir artifacts/models/<name>/<tag>`

---

## 7. 리플레이 확인

`run_local_dogfight.py --save-log` 가 남긴 Tacview CSV 3종(`*_ownship_*.csv`, `*_target_*.csv`,
`*_summary.json`)을 `logs/` 로 옮긴 뒤:
```bash
python tools/web_log_viewer.py --logdir logs --port 7870
```
브라우저에서 `http://127.0.0.1:7870` 접속.

---

## 8. 중단·재개

- **자동 재시작**: `--supervise`(기본 on)면 학습이 자식 프로세스로 돌고, 크래시 시 마지막
  `train_state.pt` 에서 자동 재시작합니다(Windows CUDA+Ray 네이티브 크래시 우회).
- **메인 재개**: `train_state.pt` 는 **매 메인 iteration 끝**에 원자적으로 저장됩니다. 중단하면
  **진행 중이던 iteration 의 시작부터** 다시 이어집니다(`--auto-resume` 또는
  `--resume-state <train_state.pt>`).
- **Exploiter 재개**: exploiter 학습은 **매 exploiter-iter 마다** `exploiter_state.pt` 로 저장되어,
  도중에 끊겨도 그 exploiter 의 중단 지점 iteration 부터 이어서 학습합니다(원래 얼렸던 상대·
  optimizer·obs_rms 까지 복원). 정상 종료하면 이 파일은 삭제됩니다.
- 스케줄(gamma·lr·ent·rollout·damage 가중치·shaping)은 iteration 번호로 결정되므로 재개 시
  정확히 복원됩니다.

---

## 9. 파일 구성 (`claude_code/`)

| 파일 | 역할 |
|---|---|
| `train.py` | **학습 entrypoint** (CLI 파싱 → 학습 → 번들 저장) |
| `ppo.py` | 단일 프로세스 PPO 트레이너 + iteration 스케줄 |
| `parallel.py` | Ray 병렬 rollout 트레이너/워커 + exploiter |
| `model.py` | 정책 네트워크(`MLPDiscreteActorCritic`) + 2-파일 번들 |
| `my_reward.py` | **[편집] 보상 함수** |
| `my_observation.py` | **[편집] 184-D 관측 + 제출환경 state 재구성** |
| `env_utils.py` | 환경 생성(`make_env`, `TierGatedDogFightEnv`) + A/B 초기 배치 |
| `self_play.py` | 상대를 actor network 로 조종하는 provider |
| `bt_rule.py` | BT DLL rule XML 지정·검증 |
| `run_local_dogfight.py` / `power_test.py` | 로컬 대결 + 리플레이 / N판 통계 판정 |
| `submission_a.py` / `submission_b.py` | 대결 서버(Unreal) UDP 제출 |
| `snapshot_to_bundle.py` / `evaluate.py` | snapshot→번들 변환 / 번들 로컬 검증 |

더 자세한 설명은 [`claude_code/README.md`](claude_code/README.md) 를 참고하세요.
