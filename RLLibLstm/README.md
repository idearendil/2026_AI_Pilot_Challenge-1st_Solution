# RLLibLstm

Ray 2.54.0 RLlib SAC New API에 DogFightEnv actor-LSTM 및 선택형 recurrent Q
경로를 적용하기 위한
패치 보관 디렉토리입니다.

## 포함 파일
- `ray_2_54_0_patched/ray/rllib/algorithms/sac/default_sac_rl_module.py`
- `ray_2_54_0_patched/ray/rllib/algorithms/sac/sac_catalog.py`
- `ray_2_54_0_patched/ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py`
- `ray_2_54_0_patched/ray/rllib/algorithms/sac/torch/sac_torch_learner.py`
- `ray_2_54_0_patched/ray/rllib/utils/replay_buffers/prioritized_episode_buffer.py`
- `ray_2_54_0_original/...`: 패치 전 원본 백업
- `tools/apply_rllib_sac_lstm_patch.py`: 현재 workspace에서 사용한 패치 스크립트
- `tools/smoke_sac_lstm_module_forward.py`: Ray cluster를 띄우지 않는 RLModule forward
  계약 확인 스크립트
- `tools/smoke_sac_lstm_actor_critic_module_forward.py`: actor_critic scope에서
  `qf/qf_twin/target_qf/target_qf_twin` recurrent Q 계약을 확인하는 no-Ray smoke
- `tools/smoke_rl_action_provider_lstm_bundle.py`: Ray cluster 없이 lightweight bundle
  RLModule state와 `RLActionProvider` recurrent state 유지 계약을 확인하는 스크립트
- `tools/smoke_build_algorithm_from_bundle_lstm.py`: 실제 `build_algorithm_from_bundle()`
  기반 Ray Algorithm 복원 smoke 스크립트
- `tools/analyze_lstm_smoke_log.py`: 학습 smoke 로그에서 `seq_lens`, sequence padding,
  probe feature 시간축 값을 자동 점검하는 분석 스크립트
- `tools/smoke_unreal_policy_lstm_reset.py`: Unreal command policy reset이 recurrent
  action provider reset으로 전달되는지 확인하는 no-Ray/no-UDP smoke 스크립트
- `tools/smoke_prioritized_sequence_replay.py`: patched prioritized replay가 실제 sequence
  slice와 `(B,T)` TD-error priority update를 처리하는지 확인하는 smoke 스크립트
- `patch_record.json`: 원본/패치본 SHA256 기록
- `SAC_LSTM_FULL_GUIDE.md`: 다른 PC 이식, smoke, prioritized replay 재도입 절차를
  포함한 전체 가이드

## 적용 대상
- Ray: `2.54.0`
- Python env 예시: `C:\Users\USER\anaconda3\envs\aip`
- 대상 site-packages:
  - `Lib/site-packages/ray/rllib/algorithms/sac/default_sac_rl_module.py`
  - `Lib/site-packages/ray/rllib/algorithms/sac/sac_catalog.py`
  - `Lib/site-packages/ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py`
  - `Lib/site-packages/ray/rllib/algorithms/sac/torch/sac_torch_learner.py`
  - `Lib/site-packages/ray/rllib/utils/replay_buffers/prioritized_episode_buffer.py`

## 적용 방법
다른 PC에서 Ray 2.54.0이 설치된 conda env 경로를 지정해 적용 스크립트를 실행합니다.
먼저 dry-run으로 대상 파일과 Ray 버전을 확인합니다.

```powershell
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py `
  C:\Users\USER\anaconda3\envs\aip --dry-run
```

확인 후 실제 적용합니다. 스크립트는 conda env root, `python.exe`, `site-packages`,
`ray`, `ray\rllib` 경로를 모두 받을 수 있습니다.

```powershell
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py `
  C:\Users\USER\anaconda3\envs\aip
```

인자를 생략하면 활성화된 conda 환경의 `%CONDA_PREFIX%`를 사용합니다.

```powershell
conda activate aip
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py --dry-run
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py
```

적용 기록은 `RLLibLstm/patch_record.json`에 남고, 원본 백업은 기본적으로
`RLLibLstm/tools/backups/<env-name>_<timestamp>/`에 저장됩니다.

## 패치 내용
- `DefaultSACRLModule.get_initial_state()`가 actor encoder의 initial state를 반환한다.
- `SACCatalog.build_qf_encoder()`가 `dogfight_lstm_scope=actor_critic`일 때
  `[obs, action] -> Q LSTM -> Q head` 구조의 recurrent Q encoder를 만든다.
- `SACCatalog`가 `dogfight_network_spec.type=sequence_v1`에서 파생된 actor/Q
  pre-LSTM MLP, LSTM layer 수, post-LSTM head MLP 설정을 반영한다.
- `DefaultSACTorchRLModule`의 inference/train path가 actor encoder에
  `Columns.STATE_IN`과 `Columns.NEXT_STATE_IN`을 전달한다.
- actor encoder가 반환한 `Columns.STATE_OUT`과 `Columns.NEXT_STATE_OUT`을 output에
  유지한다.
- `DefaultSACTorchRLModule`의 Q helper는 actor_critic scope에서 Q/twin/target Q
  state를 learner forward 내부에서 zero-init하고, `obs, action` 순서 concat을
  debug 출력으로 남긴다.
- `SACTorchLearner` continuous loss는 `Columns.LOSS_MASK`가 있으면 padded timestep을
  critic/actor/alpha loss와 metric에서 제외한다.
- `PrioritizedEpisodeReplayBuffer`가 `batch_length_T>1`일 때 1-step padded transition이
  아니라 실제 episode sequence slice를 반환한다.
- sequence-shaped TD error `(B,T)`는 sequence priority scalar `(B,)`로 축약한다.
- `DOGFIGHT_RNNSAC_DEBUG=1`이면 다음 marker로 shape와 sequence probe를 출력한다.

```text
[DogFightEnv][RLlibSAC][LSTM_IO]
```

## Actor-Critic LSTM 실행
기본값은 기존 호환을 위해 `actor_only`이다. Ray 1.9.2 RNNSAC 모사형 full recurrent
actor-critic Q를 켜려면 다음 옵션을 추가한다.

```powershell
python train_rllib.py --algorithm sac --use-lstm-sac --lstm-scope actor_critic `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io
```

YAML에서는 다음처럼 지정한다.

```yaml
algo:
  lstm:
    enabled: true
    scope: actor_critic
    cell_size: 64
    max_seq_len: 8
    debug_io: true
```

## sequence_v1 네트워크 구조 지정
`RNNSAC_model.py`의 `fc -> lstm -> fc -> fc` 구조처럼 MLP/LSTM 배치를 명시하려면
`algo.network.type: sequence_v1`을 사용한다. 이 DSL은 Ray 2.54 New API의 안정적인
Catalog 경로에 매핑되는 범위를 우선 지원한다.

```yaml
algo:
  lstm:
    enabled: true
    scope: actor_critic
    cell_size: 64
    max_seq_len: 8
    debug_io: true
  network:
    type: sequence_v1
    actor:
      encoder:
        - linear: {out: 128}
        - activation: relu
        - lstm: {hidden: 64, layers: 1}
        - linear: {out: 128}
        - activation: relu
        - linear: {out: 128}
        - activation: relu
      head:
        - linear: {out: action_dist_inputs}
    critic:
      input: obs_action
      zero_init_state: true
      encoder:
        - linear: {out: 128}
        - activation: relu
        - lstm: {hidden: 64, layers: 1}
        - linear: {out: 128}
        - activation: relu
        - linear: {out: 128}
        - activation: relu
      head:
        - linear: {out: q_value}
```

현재 지원 범위:
- `linear`, `activation`, `lstm` layer block
- actor/Q의 pre-LSTM MLP와 post-LSTM MLP 분리
- Q/twin/target Q 모두 같은 critic spec 사용
- Q 입력은 항상 `[obs, action]` concat

예시 파일:

```powershell
python scripts\run_experiment.py experiments\sac_lstm_sequence_actor_critic_smoke.yaml
python scripts\run_experiment.py experiments\sac_lstm_sequence_actor_critic_resume_smoke.yaml
```

Q 계약:
- 입력: `obs=(B,T,obs_dim)`, `actions=(B,T,action_dim)`
- concat: `(B,T,obs_dim+action_dim)` with `[obs, action]`
- Q state: helper 내부 zero-init, actor `STATE_IN` 재사용 금지
- 출력: `qf_preds`, `qf_twin_preds`, `q_curr`, `q_target_next` 모두 `(B,T)`

## Actor-Critic LSTM 학습 재개

Lightweight bundle weight-only 재시작:

```powershell
python train_rllib.py --algorithm sac `
  --init-bundle artifacts\models\f16_single_agent\lstm_actor_critic_100it `
  --iterations 100 --output-tag lstm_actor_critic_resume
```

`train_rllib.py`와 `train_curriculum.py`는 `--init-bundle`의 `metadata.json`을 읽어
`use_lstm_sac=True`, `lstm_scope=actor_critic`, `lstm_cell_size`, `max_seq_len`을 자동
복원한다. 자동 복원이 일어나면 다음 marker가 출력된다.

```text
[DogFightEnv][LSTM_RESUME] init_bundle=... use_lstm_sac=True lstm_scope=actor_critic lstm_cell_size=64 max_seq_len=8
```

Native checkpoint 재개는 optimizer/replay까지 이어받지만, restore 전 같은
architecture로 `build_algo()`해야 하므로 같은 LSTM 옵션을 명시한다.

```powershell
python train_rllib.py --algorithm sac `
  --use-lstm-sac --lstm-scope actor_critic --lstm-cell-size 64 --max-seq-len 8 `
  --restore-checkpoint artifacts\checkpoints\f16_single_agent\<tag>\<checkpoint_dir>
```

## Replay Buffer 주의
DogFightEnv 프로젝트 코드에서는 `--use-lstm-sac`일 때 RLlib 기본
`PrioritizedEpisodeReplayBuffer` 대신 `EpisodeReplayBuffer`를 사용하도록 설정했다.

판단 근거:
- Ray 2.54.0 `PrioritizedEpisodeReplayBuffer.sample(..., sample_episodes=True)`는
  코드 주석상 sequence length 1만 가능하다.
- 실제 smoke에서도 `obs_shape=(256, 8, 16)`으로 padding된 모양은 나왔지만
  `seq_lens`가 모두 1이었다.
- TD error가 sequence 배열로 나오면서 prioritized priority update가 scalar를
  기대해 `ValueError: The truth value of an array with more than one element is
  ambiguous`가 발생했다.

Prioritized replay를 다시 사용하려면 다음 중 하나가 추가로 필요하다.
- `PrioritizedEpisodeReplayBuffer`의 sequence sampling을 `EpisodeReplayBuffer`의
  sequence slice 방식과 동등하게 확장한다.
- sequence TD error를 `(B,)` priority scalar로 reduce한다. 예:
  valid mask 기준 `max(abs(td_error_t))` 또는 `mean(abs(td_error_t))`.
- `_last_sampled_indices`가 sequence당 하나의 priority 대상인지, timestep별 priority
  대상인지 계약을 명확히 맞춘다.

현재 권고는 actor-LSTM SAC 안정화 단계에서는 `EpisodeReplayBuffer`를 사용하고,
성능 비교 단계에서 prioritized sequence replay를 별도 패치로 확장하는 것이다.

현재 prioritized sequence replay는 opt-in이다.

```powershell
python train_rllib.py --algorithm sac --iterations 10 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --use-lstm-prioritized-replay `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io
```

## Smoke
Ray 잔존 node를 먼저 정리합니다.

```powershell
ray stop --force
```

프로젝트 학습 smoke:

```powershell
cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 10 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --lstm-cell-size 64 --max-seq-len 8 `
  --debug-io > lstm_smoke_train.txt
```

기대 출력:

```text
label=train_continuous
obs_shape=(..., 8, 16)
seq_lens=...
obs_probe_first_feature=[시간축으로 연속적인 값들]
```

로그 자동 점검:

```powershell
python RLLibLstm\tools\analyze_lstm_smoke_log.py `
  DogFightEnv\MyTrainEnv\lstm_smoke_train.txt --max-seq-len 8
```

Actor-critic recurrent Q no-Ray smoke:

```powershell
python RLLibLstm\tools\smoke_sac_lstm_actor_critic_module_forward.py
```

2026-05-18 확인 결과:
- `qf`, `qf_twin`, `target_qf`, `target_qf_twin` encoder 모두 recurrent 확인.
- `q_concat_shape=(2, 4, 5)`로 `obs_dim=3`, `action_dim=2` concat 확인.
- `q_action_probe_first_feature=[0.0, 0.2, 0.4, 0.6]`로 action이 LSTM 앞에
  `[obs, action]` 순서로 들어가는 것을 확인.
- `qf_preds`, `qf_twin_preds`, `q_curr`, `q_target_next` shape 모두 `(2, 4)`.

2026-05-19 확인 결과:
- actor_critic 학습 smoke analyzer가 `qf/qf_twin/target_qf/target_qf_twin` 여섯
  경로의 debug record를 모두 확인했다.
- 30 iteration 안정성 smoke에서 learner actor/critic/alpha metric은 numeric으로
  유지됐고, 예외 없이 lightweight bundle 저장까지 완료됐다.
- no-Ray bundle provider smoke에서 actor recurrent state가 step 사이에 유지되고
  reset 시 초기화되는 것을 확인했다.
- `sequence_v1` actor-critic LSTM 100 iteration smoke도 analyzer 기준 PASS.
  `seq_lens_counts=1:1, 4:1, 7:1, 8:578`로 대부분 full sequence였고,
  Q concat은 `(B,8,20) = obs16 + action4`, Q output은 `(B,8)`로 유지됐다.
  후반 learner metric은 numeric으로 유지됐다.

100 iteration 검증 로그:

```powershell
python RLLibLstm\tools\analyze_lstm_smoke_log.py `
  DogFightEnv\MyTrainEnv\lstm_sequence_actor_critic_train_debug_100it.txt `
  --max-seq-len 8 --expect-q-debug
```

판단 근거:
- 구현 완료 여부는 reward 평균만으로 판단하지 않고, sequence replay, recurrent state,
  Q concat 순서, target/twin Q 경로 debug record를 함께 확인한다.
- `Reward=[nan]` 행은 해당 iteration에 새 episode completion metric이 없는 경우가
  있으므로 learner loss/alpha가 numeric인지 같이 확인한다.

Unreal policy reset 전달 smoke:

```powershell
python RLLibLstm\tools\smoke_unreal_policy_lstm_reset.py
```

Prioritized sequence replay smoke:

```powershell
python RLLibLstm\tools\smoke_prioritized_sequence_replay.py --installed
```

Prioritized sequence replay integrity:

```powershell
python RLLibLstm\tools\verify_prioritized_sequence_replay_integrity.py `
  --target both --rounds 12 --batch-size-b 64 --batch-length-t 8
```

2026-05-18 확인 결과:
- installed site-packages와 `RLLibLstm` patched copy 모두 `result=PASS`.
- sampled sequence는 대부분 length 8이고, episode boundary 근처의 5~7 길이 sequence도
  chronological order와 priority update contract를 통과했다.

Prioritized LSTM SAC metric debug:

```powershell
cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 6 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --use-lstm-prioritized-replay `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io `
  > lstm_prioritized_metric_debug.txt
```

확인 기준:
- `[DogFightEnv][PrioritizedSeqReplay][SAMPLE] batch_length_T=8 actual_len=8`
- `[DogFightEnv][RLlibSAC][LSTM_IO] label=train_continuous obs_shape=(..., 8, ...)`
- iteration update 이후 `Actor`, `Critic`, `Alpha`가 numeric 값으로 출력

주의:
- `Alpha=0.6000`이 learner update 전 iteration에서만 보이면 SAC entropy alpha가 아니라
  prioritized replay config의 `alpha=0.6` 오탐일 수 있다. 현재 `train_rllib.py`는
  `config`/`replay_buffer_config` 경로를 nested metric 후보에서 제외한다.
- 공개 CLI 기준 LSTM/RLlib 입출력 debug는 `--debug-io` 하나로 켠다. 오래된
  `--debug-lstm-io`는 기존 실행 스크립트 호환용 숨은 alias이며 신규 문서와 명령
  예시에서는 사용하지 않는다.

Episode replay와 prioritized replay 비교:

```powershell
python RLLibLstm\tools\compare_lstm_replay_training_logs.py `
  DogFightEnv\MyTrainEnv\artifacts\logs\f16_single_agent\lstm_episode_cmp\training_log.csv `
  DogFightEnv\MyTrainEnv\artifacts\logs\f16_single_agent\lstm_prioritized_cmp\training_log.csv `
  --tail 5
```
