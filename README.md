# DogFight RL 학습 환경 매뉴얼

JSBSim/DLL 기반 F-16 1v1 공중전 강화학습 환경입니다.  
현재 `Release/` 배포본은 학생 수정 파일을 얇은 템플릿으로 유지하고,
공통 학습 루프와 기록 저장은 본체 스크립트가 담당합니다.

## 1. 실행 환경

권장 Python:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe
```

설치:

```powershell
cd C:\Users\USER\workspace\DogFightEnv\DogFightEnv\Release
C:\Users\USER\anaconda3\envs\aip\python.exe -m pip install -r requirements.txt
```

런타임 자산은 `Release/` 루트에 있어야 합니다. DLL, XML, `aircraft/`, `engine/`,
JSBSim script XML은 이름 변경, 이동, 삭제하지 마세요. 기본 제출/BT Rule XML은
`Rule_forTraining.xml`이며, 팀별 Rule을 쓰는 경우 `Rule_team01.xml`처럼 별도 파일을
두고 `--bt-rule-xml` 또는 `student/my_submission.py`에서 파일명을 지정합니다.

## 2. 주요 파일

```text
Release/
├── train_rllib.py              # 단일 스테이지 PPO/SAC 학습
├── train_curriculum.py         # 단계형 커리큘럼 PPO/SAC 학습
├── scripts/run_experiment.py   # YAML 실험 실행기
├── experiments/                # 학생 수정용 YAML 템플릿
├── src/dogfight/               # 환경, 보상, 종료, curriculum, RLlib 공통 코드
└── student/                    # 학생 수정 파일
```

학생이 주로 수정할 파일:

| 파일 | 역할 |
|---|---|
| `student/my_reward.py` | `MY_REWARD_CONFIG`, `compute_reward()` 작성 |
| `student/my_observation.py` | 선택적 custom 관측 벡터 작성 |
| `student/my_train.py` | 선택형 간단 wrapper. 최신 권장 학습 경로는 YAML 실행 |
| `student/my_curriculum.py` | 선택적 custom curriculum stage 작성 |
| `experiments/*.yaml` | 실험 이름, 알고리즘, 환경, 모듈 경로 설정 |

## 3. YAML 실험 관리

실행 전 dry-run:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_ppo_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_lstm.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_mixed_initial_sac_mlp.yaml --dry-run
```

실제 실행 예:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml
```

지원되는 `script` 값:

| 값 | 실행 대상 | 용도 |
|---|---|---|
| `train_rllib` | `train_rllib.py` | 단일 1v1 PPO/SAC 학습 |
| `train_curriculum` | `train_curriculum.py` | 단계형 커리큘럼 학습 |
| `student/my_train` | `student/my_train.py` | 선택형 간단 wrapper 실행 |

YAML에서 `env.reward_module: student.my_reward`를 지정하면 학생 보상을 주입합니다.
YAML에서 `env.observation_module: student.my_observation`를 지정하면 학생 관측을
주입합니다.
현재 학생 배포 YAML 6개는 모두 `script: train_rllib`를 기본으로 사용합니다.
`student/my_train.py`는 간단한 wrapper가 필요할 때만 선택적으로 사용합니다.

### 판단 근거

- YAML은 실험 관리 단위이고, 보상/관측/커리큘럼 아이디어는 Python 모듈에 둡니다.
- custom 관측은 학습과 추론의 입력 차원을 바꾸므로 YAML에 module path를 남겨
  결과 기록과 재현성을 확보합니다.

## 4. 단일 1v1 학습

기본 보상:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --observation-mode tactical16 `
  --target-mode behavior_tree `
  --target-behavior-dll AIP_BASE_target.dll `
  --output-name team01 `
  --output-tag single_stage_v1
```

학생 보상:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --reward-module student.my_reward `
  --observation-mode tactical16 `
  --target-mode behavior_tree `
  --target-behavior-dll AIP_BASE_target.dll `
  --output-name team01 `
  --output-tag reward_v1
```

학생 관측:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --observation-mode custom `
  --observation-module student.my_observation `
  --reward-module student.my_reward `
  --output-name team01 `
  --output-tag observation_v1
```

선택형 간단 wrapper:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe student\my_train.py --iterations 50
```

일반 학생 시작점은 `student_sac_mlp.yaml` 또는 `student_ppo_mlp.yaml`입니다.
SAC LSTM은 RLlib 패치가 필요한 고급 경로로 분리해 사용합니다.

## 5. 커리큘럼 학습

기본 curriculum은 `src/dogfight/ai/curriculum.py`에 있으며, 다음 순서입니다.

| Stage | 이름 | 목적 |
|---:|---|---|
| 0 | `flight_survival` | 비행/스로틀 안정화 |
| 1 | `target_pursuit` | 고정 표적 추적 |
| 2 | `wez_approach` | loiter 표적 상대로 WEZ 진입 |
| 3 | `autopilot_pursuit` | 이동 표적 추적 |
| 4-13 | `two_circle_headon_a***` | alpha 투서클 헤드온 |
| 14 | `full_dogfight` | BT 상대 전면 교전 |

학생 curriculum 사용:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_curriculum.py `
  --algorithm sac `
  --reward-module student.my_reward `
  --stages-module student.my_curriculum `
  --output-name team01 `
  --output-tag curriculum_v1
```

## 6. 체크포인트 재개와 번들 재시작

Native checkpoint는 RLlib 전체 상태를 저장합니다. 정책 weight뿐 아니라 optimizer,
replay buffer 등 학습 상태를 이어받아 같은 학습을 계속할 때 사용합니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --lightweight-bundle-frequency 10 `
  --save-native-checkpoint `
  --native-checkpoint-frequency 10 `
  --output-name team01 `
  --output-tag v1

C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --restore-checkpoint artifacts\checkpoints\team01\v1\checkpoint_000010 `
  --output-name team01 `
  --output-tag v1_resume
```

저장 위치:

- lightweight bundle 최종본: `artifacts\models\<output-name>\<output-tag>`
- lightweight bundle 주기 저장: `artifacts\models\<output-name>\<output-tag>\bundle_000010`
- native checkpoint 최종본: `artifacts\checkpoints\<output-name>\<output-tag>\checkpoint_final`
- native checkpoint 주기 저장: `artifacts\checkpoints\<output-name>\<output-tag>\checkpoint_000010`

YAML에서는 `runtime` 아래에서 두 저장 정책을 따로 설정합니다.

```yaml
runtime:
  save_lightweight_bundle: true
  lightweight_bundle_frequency: 0
  save_native_checkpoint: true
  native_checkpoint_frequency: 10
```

`0` 주기는 최종본만 저장한다는 뜻입니다. 기존 `checkpoint_frequency`는 오래된
YAML/CLI 호환용으로 남아 있으며, 새 실험에서는 `native_checkpoint_frequency`를
사용합니다.

커리큘럼은 기존 중단 학습을 이어갈 때 `--resume`을 우선 사용합니다. 특정
checkpoint에서 새 커리큘럼 run을 시작하려면 `--restore-checkpoint`를 사용합니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_curriculum.py `
  --algorithm sac `
  --output-name team01 `
  --output-tag curriculum_v1 `
  --resume
```

Lightweight bundle은 `metadata.json`과 `policy_weights.pkl.gz`만 포함합니다.
optimizer와 replay buffer는 새로 시작하므로, 제출/추론용 정책을 초기 weight로 삼아
다른 실험을 시작할 때 사용합니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 50 `
  --init-bundle artifacts\models\team01\v1 `
  --output-name team01 `
  --output-tag bundle_restart_v1
```

YAML에서는 `runtime.restore_checkpoint` 또는 `runtime.init_bundle`을 사용합니다.
두 옵션은 동시에 지정하지 않습니다.

## 6-1. LSTM 및 네트워크 구조 템플릿

Ray 2.54.0 SAC에 LSTM을 적용하려면 프로젝트 옵션만으로는 부족하고, 워크스페이스
루트의 `RLLibLstm/` 패치가 현재 conda 환경의 RLlib에 적용되어 있어야 합니다.
학생 기본 실습은 일반 PPO/SAC로 진행하고, RNNSAC 비교나 장기 기억 정책 실험이
필요할 때만 사용합니다.

Release 배포본에는 학생용 YAML 6개가 제공됩니다. 일반 시작점은
`student_sac_mlp.yaml`, `student_ppo_mlp.yaml`이고, mixed initial scenario 예시는
`student_mixed_initial_sac_mlp.yaml`부터 확인합니다. SAC LSTM은 RLlib 패치가 필요한
고급 경로이고, PPO LSTM은 RLlib 기본
`DefaultModelConfig` 기반 recurrent actor/value 경로입니다. Ray 1.9.2
`RNNSAC_model.py` 형태의 layer sequence 자유도는 `algo.network.type: sequence_v1`로
관리합니다.

템플릿 dry-run:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_ppo_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_lstm.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_ppo_lstm.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_mixed_initial_sac_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_mixed_initial_sac_lstm.yaml --dry-run
```

학습 중 actor 출력 상태를 육안 확인하려면 YAML의 `policy_probe` 섹션을 사용합니다.
구조별 템플릿에는 기본으로 켜져 있으며, 출력은
`artifacts/logs/<output-name>/<output-tag>/policy_probe.csv`와
`policy_probe.jsonl`에 저장됩니다.

```yaml
policy_probe:
  enabled: true
  interval: 5
  steps: 4
  print: true
```

학습 중 실제 교전 궤적을 뷰어로 확인하려면 `engagement_log`를 켭니다. 이 기능은
현재 actor로 짧은 평가 교전을 별도 실행하고, 통합 대시보드의 `Replay` 탭이 읽는
Tacview CSV 쌍을 `artifacts/logs/<output-name>/<output-tag>/engagement_replays/`에
저장합니다.

```yaml
engagement_log:
  enabled: true
  interval: 10
  steps: 600
  episodes: 1
  print: true
```

저장된 replay는 다음처럼 통합 대시보드에서 확인합니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\dashboard.py `
  --default-tab replay `
  --logdir artifacts\logs\<output-name>\<output-tag>\engagement_replays `
  --port 7860
```

Actor만 recurrent로 쓸 때:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 10 `
  --use-lstm-sac `
  --lstm-cell-size 64 `
  --max-seq-len 8 `
  --debug-io `
  --output-name team01 `
  --output-tag sac_lstm_smoke
```

Ray 1.9.2 RNNSAC 모사형 recurrent actor-critic Q를 쓸 때:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe train_rllib.py `
  --algorithm sac `
  --iterations 30 `
  --train-batch-size 64 `
  --rollout-fragment-length 8 `
  --use-lstm-sac `
  --lstm-scope actor_critic `
  --lstm-cell-size 64 `
  --max-seq-len 8 `
  --debug-io `
  --output-name team01 `
  --output-tag sac_lstm_actor_critic
```

판단 근거:

- `actor_critic` scope는 Q/twin/target Q가 모두 `[obs, action] -> Q LSTM -> Q head`
  계약을 사용하므로 Ray 1.9.2 RNNSAC 구조를 Ray 2.54 New API에 맞춰 모사합니다.
- `actor_only` scope는 actor만 LSTM이고 Q/twin/target Q는 MLP입니다.
- PPO LSTM 템플릿은 `algo.lstm.enabled: true`를 `--use-lstm`으로 변환하며,
  SAC 전용 RLlib 패치 경로인 `--use-lstm-sac`를 사용하지 않습니다.
- `sequence_v1`은 YAML에서 `linear`, `activation`, `lstm` layer 순서를 지정합니다.
  SAC critic 입력은 항상 `[obs, action]` concat 순서를 사용하고, PPO는 actor section만
  사용합니다.
- `policy_probe`는 고정 observation probe에 대한 action, action 변화량, LSTM
  state norm을 기록하므로 reward가 드문 초반에도 행위 모델 출력 변화를 확인할 수
  있습니다.
- `engagement_log`는 실제 환경을 추가로 짧게 실행해 교전 CSV를 남깁니다. replay buffer
  샘플을 꺼내 쓰지 않으므로 학습 batch 계약은 건드리지 않지만, 환경 실행 비용과 디스크
  사용량이 늘어나므로 긴 학습에서는 interval을 크게 둡니다.
- `--debug-io`는 `seq_lens`, recurrent state, Q concat 순서를 확인하기 위한 단일
  공개 debug 옵션입니다.
- 긴 Ray 학습과 Unreal 제출 검증 전에는 `ray stop --force`로 잔존 Ray node를 정리하고,
  짧은 smoke에서 bundle 저장까지 확인하는 흐름을 권장합니다.
- LSTM smoke에서 `Reward=[nan]` 행이 보이더라도 해당 iteration에 새 episode completion
  metric이 없는 경우가 있으므로, actor/critic/alpha loss가 numeric인지 함께 봅니다.
- 장기 학습에서는 `--debug-io`를 끄고 별도 `output-tag`를 사용합니다. debug 출력은
  입출력 계약 확인용이며 로그가 커집니다.

## 7. 관측, 행동, 보상

권장 관측 모드는 `tactical16`입니다.

| 인덱스 | 의미 |
|---:|---|
| 0-5 | ownship 자세, 속도, 고도, 체력 |
| 6-8 | 상대 위치 delta N/E/D |
| 9 | ATA |
| 10 | AA |
| 11-12 | LOS 방위각/고각 |
| 13 | target 체력 |
| 14 | WEZ 진입 여부 |
| 15 | 추적 점수 |

행동 공간은 `Box([-1, 1]^4)`입니다.

| 축 | 의미 |
|---|---|
| 0 | roll |
| 1 | pitch |
| 2 | rudder/yaw |
| 3 | throttle, 내부에서 `[0, 1]`로 변환 |

기본 보상은 `src/dogfight/envs/reward.py`에 있습니다. 학생 보상은
`student/my_reward.py`의 `compute_reward()` 계약만 지키면 됩니다.

관측값을 직접 설계하려면 `student/my_observation.py`를 수정합니다. 필수 계약은
`OBSERVATION_SIZE`와 `build_observation(...)`이며, 학습 시
`--observation-mode custom --observation-module student.my_observation`을 사용합니다.
관측 차원을 바꾸면 기존 checkpoint/bundle과 호환되지 않습니다.

custom 관측을 사용한 policy는 로컬/Unreal 추론에서도 같은 옵션을 지정해야 합니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe run_local_dogfight.py `
  --ownship-backend rl `
  --ownship-bundle-dir artifacts\models\team01\observation_v1 `
  --target-backend bt `
  --observation-mode custom `
  --observation-module student.my_observation
```

## 8. 출력과 대시보드

단일 학습 출력:

```text
artifacts/
├── logs/<output-name>/<output-tag>/training_log.csv
├── models/<output-name>/<output-tag>/
├── records/<output-name>/<output-tag>/
└── dashboard/<output-name>_<output-tag>/
```

커리큘럼 출력:

```text
artifacts/curriculum/<output-name>/<output-tag>/
├── curriculum_state.json
├── training_log.csv
├── stage_*_*/checkpoints/
└── stage_*_*/final_bundle/
```

통합 대시보드:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\dashboard.py `
  --training-logdir artifacts\dashboard `
  --logdir logs `
  --port 7860
```

브라우저: `http://127.0.0.1:7860`

상단 탭에서 `Training`은 `metrics.jsonl` 학습 지표를, `Replay`는 Tacview CSV
교전 replay를 표시합니다. PyVista 뷰어는 활성 실행 경로에서 제거되었고,
브라우저 기반 Replay 탭이 표준 확인 경로입니다. 기존 호환 명령
`tools\training_dashboard\server.py`와 `tools\web_log_viewer.py`도 통합 서버를
실행합니다.

## 9. Smoke 검증

문법과 YAML 명령 생성:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe -m py_compile train_rllib.py train_curriculum.py scripts\run_experiment.py student\my_reward.py student\my_observation.py student\my_train.py student\my_curriculum.py
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_ppo_mlp.yaml --dry-run
```

짧은 학습 smoke:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml
```

## 10. 로컬 교전 검증

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe run_local_dogfight.py `
  --ownship-backend rl `
  --ownship-bundle-dir artifacts\models\<output-name>\<output-tag> `
  --target-backend bt `
  --save-log
```

`--save-log`를 켜면 정상 episode 종료 후 `logs/` 아래에 같은 timestamp의
ownship CSV, target CSV, summary JSON이 생성됩니다. CSV에는 Tacview/Replay 탭에서
읽는 시간, 위도/경도, 고도, 자세, Health가 들어가고, summary JSON에는
`end_condition`, `outcome`, 양측 Health가 들어갑니다.

이 로그는 제출 전 로컬 교전 복기용 표준 절차입니다. 다만 provider 예외,
`KeyboardInterrupt`, reset/build 실패처럼 정상 종료 블록에 도달하지 못한 경우까지
항상 보존하는 crash dump 기능은 아닙니다. NaN 조기 종료 경로도 마지막 실패 프레임이
완전히 남지 않을 수 있으므로, 이 경우에는 터미널 종료 사유와 함께 확인합니다.

판단 근거:

- `run_local_dogfight.py --save-log`는 episode loop가 `terminated` 또는 `truncated`로
  끝난 뒤 `env.make_tacviewLog()`를 호출합니다.
- 환경은 정상 `step()` 경로에서 매 step `ownship_log`와 `target_log`를 누적하고,
  `make_tacviewLog()`에서 두 CSV와 summary JSON을 씁니다.
- 종료 조건은 FDM update fail, 고도 하한, 기체 destroyed, fuel fail, timeout,
  episode step limit 등을 포함합니다.

## 11. Ray/RLlib와 ONNX 확장 구조

현재 기본 학습 경로는 Ray/RLlib가 `DogFightWrapper` 환경을 감싸는 구조입니다.
Ray/RLlib는 rollout 수집, 학습 루프, checkpoint, lightweight bundle 저장을 담당하고,
환경 자체는 `reset()`/`step()`과 observation/action 계약을 제공합니다.

따라서 강의 관점에서는 다음 구조로 분리해 설명할 수 있습니다.

1. `DogFightWrapper` 또는 동일한 환경 계약을 직접 호출해 PyTorch/TensorFlow 학습 루프를
   작성합니다.
2. 학습된 neural network를 ONNX로 export합니다.
3. ONNX Runtime 등으로 inference를 수행해 4차원 action `[roll, pitch, rudder, throttle]`
   을 반환합니다.
4. 그 inference 코드를 `ActionProvider.compute_action()` 계약으로 감싸면 local/Unreal
   추론 경로에 공통으로 연결할 수 있습니다.

주의할 점은 현재 `run_local_dogfight.py`와 `run_unreal_inference.py`가 바로 지원하는
RL backend는 RLlib lightweight bundle(`metadata.json`, `policy_weights.pkl.gz`)이라는
점입니다. ONNX 파일을 현재 CLI에 그대로 넣어 실행하는 기능은 아직 구현되어 있지
않습니다. 실제 ONNX 연결이 필요하면 `ActionProvider`를 구현하는 ONNX adapter와 CLI
옵션을 추가해야 합니다.

판단 근거:

- local/Unreal 추론은 모두 `ActionProvider.compute_action()`이 반환하는 4차원 action을
  사용합니다.
- `run_local_dogfight.py`는 ownship/target provider를 환경에 주입하고,
  `run_unreal_inference.py`는 provider를 `ProviderCommandPolicy`에 주입합니다.
- 이 연결 지점은 RLlib bundle, BT DLL, hybrid provider가 이미 공유하는 추상화이므로
  ONNX 추론도 같은 adapter 패턴으로 확장하는 것이 가장 작고 안전한 변경입니다.

## 12. 제출 실행

`student/my_submission.py`의 `BUNDLE_DIR`, `TEAM_NAME`, `SERVER_IP`를 설정한 뒤 실행합니다.
BT 또는 hybrid 제출은 `BT_DLL`과 `BT_RULE_XML`도 팀 파일명에 맞춥니다.
기본 Rule은 `Rule_forTraining.xml`이고, 팀별 Rule은 `Rule_team01.xml`처럼
별도 파일로 두는 것을 권장합니다. 학습 YAML의 `step_ratio: 6`과 맞추기 위해
Unreal 추론 기본 `ACTION_REPEAT`/`--action-repeat`은 6입니다.

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe student\my_submission.py
```

CLI로 직접 실행할 때:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe run_unreal_inference.py `
  --mode rl `
  --bundle-dir artifacts\models\team01\sac_mlp_v1 `
  --team-name team01 `
  --server-ip <IP> --server-port 9999 `
  --action-repeat 6

C:\Users\USER\anaconda3\envs\aip\python.exe run_unreal_inference.py `
  --mode hybrid `
  --bundle-dir artifacts\models\team01\sac_mlp_v1 `
  --bt-dll AIP_BASE.dll `
  --bt-rule-xml Rule_team01.xml `
  --team-name team01 `
  --server-ip <IP> --server-port 9999 `
  --action-repeat 6
```

## 13. 판단 근거 및 적용 범위

- 이번 단순화는 `Release/`에만 적용했습니다. `MyTrainEnv/`는 내부 작업/검증용 구현을
  유지합니다.
- 학생 템플릿은 그대로 실행 가능하지만, 완성형 전술 보상/커리큘럼을 답안처럼
  제공하지 않습니다.
- 연구/비교 YAML은 학생 배포본에서 제거했고,
  `LogDevelop/260513_1716_release_student_template_simplification.md`에 보존했습니다.
- Native checkpoint 재개와 lightweight bundle 재시작은 같은 CLI 옵션을
  `train_rllib.py`, `train_curriculum.py` 모두에 제공했습니다. 전자는 학습 상태 보존,
  후자는 weight-only 초기화가 목적입니다.
- 학생 custom observation은 `student/my_observation.py` 모듈 hook으로 주입합니다.
  기본 관측 모드는 유지하고, `--observation-module`을 명시한 실험에만 적용합니다.
- custom 관측은 학습, 로컬 검증, Unreal 제출 경로에서 같은 모듈을 사용해야 하므로
  README와 YAML에 module path를 명시합니다.
- SAC LSTM/RNNSAC 실험은 RLlib 내부 패치가 필요한 고급 경로이므로, 일반 학생
  템플릿과 분리해 `RLLibLstm/` 가이드를 기준으로 적용합니다. DLL, XML,
  aircraft/engine 자산은 여전히 이름 변경, 이동, 삭제 대상이 아닙니다.
- ONNX는 현재 즉시 실행 가능한 CLI 입력 형식이 아니라 확장 가능한 inference 구조로
  설명합니다. 실제 사용 시에는 `ActionProvider` 기반 adapter 구현이 필요합니다.
- Release Word 매뉴얼은 `LogDevelop/Release_SAC_LSTM_User_Manual.docx`로 별도
  작성해 학생 실습 절차와 고급 LSTM 연구 절차를 구분합니다.
