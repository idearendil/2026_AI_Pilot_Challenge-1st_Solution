# Release Experiment YAML

이 폴더는 학생 배포용 최소 YAML 템플릿만 둡니다. 각 파일은 주석으로 다음 수정 지점을
안내합니다.

1. 시나리오 선택
2. 학습 알고리즘 선택
3. 학습 하이퍼파라미터 조정
4. 개인 관측/보상 설정
5. 신경망 선택
6. 로그 남기는 설정

## 실행

```powershell
cd C:\Users\USER\workspace\DogFightEnv\DogFightEnv\Release
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml --dry-run
C:\Users\USER\anaconda3\envs\aip\python.exe scripts\run_experiment.py experiments\student_sac_mlp.yaml
```

## 제공 YAML

| 파일 | 용도 |
|---|---|
| `student_sac_mlp.yaml` | SAC + MLP baseline. 가장 먼저 돌려볼 기본 off-policy 템플릿 |
| `student_sac_lstm.yaml` | SAC + actor/Q LSTM. RLLibLstm 패치 환경에서 사용하는 고급 템플릿 |
| `student_ppo_mlp.yaml` | PPO + MLP baseline. replay buffer 없이 on-policy로 학습 |
| `student_ppo_lstm.yaml` | PPO + LSTM. 기본 RLlib recurrent 경로 사용 |
| `student_mixed_initial_sac_mlp.yaml` | reset마다 BT/loiter target을 섞는 initial scenario SAC MLP 예시 |
| `student_mixed_initial_sac_lstm.yaml` | reset마다 BT/loiter target을 섞는 initial scenario SAC LSTM 예시 |

## 주요 수정 포인트

| YAML 경로 | 의미 |
|---|---|
| `output.name` | 팀 이름 또는 실험 그룹 이름 |
| `output.tag` | 저장될 모델/로그 버전 이름 |
| `env.observation_mode` | `classic12`, `relative14`, `tactical16`, `custom` |
| `env.observation_module` | `student.my_observation` 같은 custom 관측 module path |
| `env.target_mode` | `fixed`, `loiter`, `autopilot`, `behavior_tree` |
| `env_config.initial_scenario` | reset 시 초기 배치와 target type 분포 설정 |
| `env.reward_module` | `student.my_reward` 같은 개인 보상 module path |
| `env_config.reward` | 기본 보상 scale 조정 |
| `algo.name` | `sac` 또는 `ppo` |
| `algo.lr`, `gamma`, `train_batch_size` | 주요 학습 하이퍼파라미터 |

`legacy37`와 `ref_old_1vs1`은 연구 비교용 계약이므로 학생 배포본에서는 기본
노출하지 않습니다. 해당 실험은 `MyTrainEnv/`의 ref_old 비교 YAML을 기준으로
관리합니다.
| `algo.mlp` | MLP hidden layer/activation 설정 |
| `algo.lstm` | LSTM 사용, hidden size, sequence length 설정 |
| `algo.network` | SAC LSTM sequence 구조를 layer 순서로 직접 지정 |
| `runtime.iterations` | 학습 iteration 수 |
| `runtime.save_lightweight_bundle` | 추론용 lightweight bundle 저장 여부 |
| `runtime.lightweight_bundle_frequency` | N iteration마다 bundle snapshot 저장, `0`은 최종본만 저장 |
| `runtime.save_native_checkpoint` | RLlib native checkpoint 저장 여부 |
| `runtime.native_checkpoint_frequency` | N iteration마다 native checkpoint 저장, `0`은 최종본만 저장 |
| `runtime.init_bundle` | lightweight bundle weight에서 새 학습 시작 |
| `runtime.restore_checkpoint` | RLlib native checkpoint에서 전체 학습 상태 복원 |
| `policy_probe` | 학습 중 고정 입력 actor 출력/state 로그 |
| `engagement_log` | 학습 중 짧은 평가 교전 Tacview CSV 저장 |

저장 위치:

- lightweight bundle 최종본: `artifacts/models/<output.name>/<output.tag>`
- lightweight bundle 주기 저장: `artifacts/models/<output.name>/<output.tag>/bundle_000010`
- native checkpoint 최종본: `artifacts/checkpoints/<output.name>/<output.tag>/checkpoint_final`
- native checkpoint 주기 저장: `artifacts/checkpoints/<output.name>/<output.tag>/checkpoint_000010`

기존 `runtime.checkpoint_frequency`는 호환 alias입니다. 새 YAML에서는
`runtime.native_checkpoint_frequency`를 사용합니다.

## 판단 근거

- Release는 학생이 바로 수정할 파일만 남겨 혼선을 줄입니다.
- ref_old 비교용, 내부 smoke, 연구용 curriculum YAML은 MyTrainEnv 쪽에서 관리합니다.
- `initial_scenario.mode: ref_old_random`을 사용하면 reset마다 ref old scenario index를
  뽑아 BT/loiter target을 섞을 수 있습니다. 이때 초기 `env.target_mode`는
  `behavior_tree`로 둡니다. index `0..4`는 BT target, `5..7`은 loiter target입니다.
- `run_experiment.py`는 YAML 원본을 `--experiment-yaml`로 넘기며, `train_rllib.py`가
  YAML의 `env_config`를 실제 환경 설정에 병합합니다.
- SAC LSTM은 Ray/RLlib 설치본 패치가 필요한 고급 기능이므로 `student_sac_lstm.yaml`에
  명시했습니다. 패치가 없는 환경에서는 `student_sac_mlp.yaml` 또는 PPO 템플릿을 먼저
  사용합니다.
- lightweight bundle은 제출/추론용 weight 산출물이고 native checkpoint는 학습 상태
  복구용 산출물이므로 저장 여부와 주기를 분리합니다.
