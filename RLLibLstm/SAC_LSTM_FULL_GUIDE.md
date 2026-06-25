# SAC LSTM + RLlib 2.54 적용 전체 가이드

## 목적
DogFightEnv에서 Ray RLlib 2.54.0 SAC New API에 actor-LSTM 및 Ray 1.9.2 RNNSAC
모사형 recurrent Q를 적용하고, 다른 PC에서 같은 작업을 이어갈 수 있도록 절차와
판단 근거를 정리한다.

현재 기본값은 기존 호환을 위해 `actor_only LSTM + EpisodeReplayBuffer`이다.
Ray 1.9.2 학습 성능 모사를 위한 실험 경로로 `actor_critic` scope를 추가했다.
`PrioritizedEpisodeReplayBuffer` sequence replay는 opt-in 실험 경로로 추가했다.

판단 근거:
- RLlib 2.54 기본 SAC RLModule은 `get_initial_state()`가 빈 dict를 반환해 SAC가
  stateful module로 인식되지 않는다.
- actor LSTM은 EnvRunner, replay, learner, inference provider가 모두
  `STATE_IN`/`STATE_OUT` 계약을 유지해야 동작한다.
- Ray 1.9.2 RNNSAC의 성능 좋았던 구조는 `zero_init_states=True`,
  `[obs, action] -> Q LSTM -> Q head`, target Q, twin Q가 함께 쓰인 형태로
  해석했고, Ray 2.54 New API의 기존 target/twin Q 뼈대는 유지한 채 Q encoder만
  recurrent로 교체하는 것이 가장 작은 이식 단위다.
- RLlib 2.54 `PrioritizedEpisodeReplayBuffer`는 `sample_episodes=True` 경로에서
  sequence length 1 중심으로 동작해 LSTM replay sequence를 만들지 못한다.

---

## 1. 디렉토리 구성

```text
RLLibLstm/
├── README.md
├── SAC_LSTM_FULL_GUIDE.md
├── manifest.json
├── patch_record.json
├── ray_2_54_0_original/
│   └── ray/rllib/algorithms/sac/...
├── ray_2_54_0_patched/
│   └── ray/rllib/algorithms/sac/...
└── tools/
    ├── apply_rllib_sac_lstm_patch.py
    ├── analyze_lstm_smoke_log.py
    ├── smoke_build_algorithm_from_bundle_lstm.py
    ├── smoke_prioritized_sequence_replay.py
    ├── smoke_rl_action_provider_lstm_bundle.py
    ├── smoke_sac_lstm_actor_critic_module_forward.py
    └── smoke_sac_lstm_module_forward.py
```

`ray_2_54_0_original`은 원본 백업, `ray_2_54_0_patched`는 현재 적용한 패치본이다.
`manifest.json`은 파일별 SHA256을 기록한다.

---

## 2. 다른 PC 적용 절차

### 2.1 Ray 버전 확인

```powershell
python -c "import ray; print(ray.__version__)"
```

기준 버전은 `2.54.0`이다. 다른 버전이면 파일 구조와 내부 API가 달라질 수 있으므로
그대로 덮어쓰지 말고 diff 기반으로 이식한다.

### 2.2 site-packages 위치 확인

```powershell
python -c "import ray, pathlib; print(pathlib.Path(ray.__file__).parent)"
```

예시:

```text
C:\Users\USER\anaconda3\envs\aip\Lib\site-packages\ray
```

### 2.3 원본 백업 및 패치 적용

권고 절차는 적용 스크립트를 사용하는 것이다. 다른 PC에서는 conda env root를 인자로
넘기면 스크립트가 자동으로 `Lib/site-packages/ray/rllib` 위치를 찾는다. 먼저 dry-run으로
대상 파일과 Ray 버전을 확인한다.

```powershell
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py `
  C:\Users\USER\anaconda3\envs\aip --dry-run
```

확인 후 실제 적용한다.

```powershell
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py `
  C:\Users\USER\anaconda3\envs\aip
```

스크립트 입력 경로는 다음 형태를 모두 지원한다.

- conda env root: `C:\Users\USER\anaconda3\envs\aip`
- Python 실행 파일: `C:\Users\USER\anaconda3\envs\aip\python.exe`
- site-packages: `...\Lib\site-packages`
- Ray package root: `...\Lib\site-packages\ray`
- RLlib root: `...\Lib\site-packages\ray\rllib`

인자를 생략하면 활성화된 conda 환경의 `%CONDA_PREFIX%`를 사용한다.

```powershell
conda activate aip
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py --dry-run
python RLLibLstm\tools\apply_rllib_sac_lstm_patch.py
```

스크립트는 기존 site-packages 파일을 기본적으로
`RLLibLstm/tools/backups/<env-name>_<timestamp>/`에 저장하고,
`RLLibLstm/patch_record.json`에 SHA256을 기록한다. Ray 버전이 `2.54.0`이 아니면
중단하며, 정말 강제로 적용해야 할 때만 `--force-version`을 사용한다.

적용 대상:
- `ray/rllib/algorithms/sac/default_sac_rl_module.py`
- `ray/rllib/algorithms/sac/sac_catalog.py`
- `ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py`
- `ray/rllib/algorithms/sac/torch/sac_torch_learner.py`
- `ray/rllib/utils/replay_buffers/prioritized_episode_buffer.py`

수동으로 적용해야 한다면 같은 상대 경로의 `ray_2_54_0_patched` 파일을
site-packages에 덮어쓴다.

```powershell
copy RLLibLstm\ray_2_54_0_patched\ray\rllib\algorithms\sac\default_sac_rl_module.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\default_sac_rl_module.py

copy RLLibLstm\ray_2_54_0_patched\ray\rllib\algorithms\sac\sac_catalog.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\sac_catalog.py

copy RLLibLstm\ray_2_54_0_patched\ray\rllib\algorithms\sac\torch\default_sac_torch_rl_module.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\torch\default_sac_torch_rl_module.py

copy RLLibLstm\ray_2_54_0_patched\ray\rllib\algorithms\sac\torch\sac_torch_learner.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\torch\sac_torch_learner.py

copy RLLibLstm\ray_2_54_0_patched\ray\rllib\utils\replay_buffers\prioritized_episode_buffer.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\utils\replay_buffers\prioritized_episode_buffer.py
```

### 2.4 Ray 잔존 node 정리

```powershell
ray stop --force
```

판단 근거:
- 이전 Ray local cluster가 남아 있으면 `build_algo()` 또는 learner smoke 결과가
  오염될 수 있다.
- 긴 검증 스크립트에는 `try/finally: ray.shutdown()`을 넣는다.

---

## 3. RLlib 패치 내용

### 3.1 `default_sac_rl_module.py`

변경 목적:
- SAC module이 actor encoder의 recurrent state를 알도록 한다.

핵심 계약:

```python
def get_initial_state(self) -> dict:
    if hasattr(self.pi_encoder, "get_initial_state"):
        return self.pi_encoder.get_initial_state()
    return {}
```

판단 근거:
- `get_initial_state()`가 빈 dict이면 EnvRunner와 connector가 stateful module로
  처리하지 않는다.
- actor encoder의 state만 반환하는 이유는 Q state를 replay에 저장하지 않는 v1
  구현이기 때문이다. `actor_critic` scope에서도 Q state는 learner forward helper
  내부에서 zero-init한다.

### 3.2 `sac_catalog.py`

변경 목적:
- `dogfight_lstm_scope=actor_critic`일 때 Q/twin/target Q encoder를 recurrent로 만든다.

핵심 계약:
- continuous Q encoder input dim은 `obs_dim + action_dim`.
- concat 순서는 항상 `[obs, action]`.
- Q head input dim은 `lstm_cell_size`.
- actor encoder는 기존 `DefaultModelConfig(use_lstm=True)` 경로를 유지한다.

판단 근거:
- Ray 2.54 SAC는 target/twin Q module 생성 경로를 이미 갖고 있으므로 새 Q module을
  만들 필요가 없다.
- Ray 1.9.2 RNNSAC 모사는 Q encoder만 recurrent로 바꾸는 방식이 New API와 가장 잘
  맞는다.

### 3.3 `torch/default_sac_torch_rl_module.py`

변경 목적:
- inference와 train continuous path에서 actor encoder에 state 입력을 전달하고,
  actor encoder가 반환하는 state 출력을 보존한다.
- `actor_critic` scope에서는 Q/twin/target Q helper가 Q LSTM zero state를 만들고,
  Q 입력 concat과 state shape를 debug record로 남긴다.

확인할 주요 입력:
- `Columns.OBS`
- `Columns.NEXT_OBS`
- `Columns.ACTIONS`
- `Columns.STATE_IN`
- `Columns.NEXT_STATE_IN`
- `Columns.SEQ_LENS`

확인할 주요 출력:
- `Columns.ACTION_DIST_INPUTS`
- `ACTION_DIST_INPUTS_NEXT`
- `Columns.STATE_OUT`
- `Columns.NEXT_STATE_OUT`
- `QF_PREDS`, `QF_TWIN_PREDS`, `q_curr`, `q_target_next` with `(B,T)`

debug marker:

```text
[DogFightEnv][RLlibSAC][LSTM_IO]
```

판단 근거:
- SAC learner는 current actor output과 next actor output을 모두 사용한다.
- 따라서 current state와 next state를 분리해 넣어야 replay/learner 계약을 육안으로
  확인할 수 있다.
- actor `STATE_IN`을 Q encoder에 재사용하면 policy memory와 critic memory가 섞이므로
  Q helper 내부 zero-init을 명시적으로 사용한다.

### 3.4 `torch/sac_torch_learner.py`

변경 목적:
- sequence padding이 critic/actor/alpha loss와 metric에 들어가지 않도록
  `Columns.LOSS_MASK`를 continuous SAC loss에 적용한다.

판단 근거:
- LSTM replay는 episode boundary에서 padded timestep을 만들 수 있다.
- padded timestep이 loss에 들어가면 짧은 sequence가 Q target과 alpha update를
  오염시킬 수 있다.

---

## 4. DogFightEnv 프로젝트 설정

### 4.1 CLI 옵션

`train_rllib.py`, `train_curriculum.py` 양쪽에 다음 옵션을 사용한다.

```powershell
--use-lstm-sac
--lstm-scope actor_only|actor_critic
--lstm-cell-size 64
--max-seq-len 8
--debug-io
```

`--lstm-scope` 기본값은 `actor_only`이다. Ray 1.9.2 RNNSAC 모사형 target/twin
recurrent Q를 켤 때만 `actor_critic`을 지정한다.
입출력 debug 공개 CLI는 `--debug-io`로 통일한다. 기존 `--debug-lstm-io`는 오래된
실행 기록 호환용 숨은 alias이며 신규 실행 예시에는 사용하지 않는다.

### 4.2 YAML 옵션

```yaml
algo:
  name: sac
  lstm:
    enabled: true
    scope: actor_critic
    cell_size: 64
    max_seq_len: 8
    debug_io: true
```

### 4.3 Replay buffer 기본값

`--use-lstm-sac`일 때 현재 권고 설정:

```python
replay_buffer_config["type"] = "EpisodeReplayBuffer"
replay_buffer_config["batch_length_T"] = max_seq_len
```

판단 근거:
- `EpisodeReplayBuffer`는 실제 episode slice를 만들어 LSTM sequence를 구성할 수 있다.
- `PrioritizedEpisodeReplayBuffer`는 Ray 2.54 기준 sequence sampling이 1-step으로
  제한되어 LSTM 학습 입력을 망가뜨린다.

### 4.4 Actor-Critic Q 입출력 계약

`actor_critic` scope에서 Q/twin/target Q의 계약은 다음과 같다.

```text
obs:     (B, T, obs_dim)
actions: (B, T, action_dim)
concat:  (B, T, obs_dim + action_dim), order=[obs, action]
q_state: {'h': (B, 1, H), 'c': (B, 1, H)} zero-init inside helper
output:  qf_preds/qf_twin_preds/q_curr/q_target_next = (B, T)
```

금지 사항:
- actor `STATE_IN`을 Q encoder에 재사용하지 않는다.
- action을 LSTM 뒤에 붙이지 않는다.
- concat 순서를 `[action, obs]`로 바꾸지 않는다.
- padded timestep을 loss와 metric에 포함하지 않는다.

---

## 5. Smoke 검증 절차

### 5.1 RLModule 단독 smoke

Ray cluster를 띄우지 않는 actor-only regression 확인:

```powershell
python RLLibLstm\tools\smoke_sac_lstm_module_forward.py
```

기대 출력:

```text
label=inference obs_shape=(1, 1, 3)
label=train_continuous obs_shape=(1, 4, 3)
state_out={'h': ..., 'c': ...}
next_state_out={'h': ..., 'c': ...}
```

Ray cluster를 띄우지 않는 actor-critic recurrent Q 확인:

```powershell
python RLLibLstm\tools\smoke_sac_lstm_actor_critic_module_forward.py
```

기대 출력:

```text
[DogFightEnv][actor_critic_smoke] train_output_shapes=
  {'qf_preds': (2, 4), 'qf_twin_preds': (2, 4), 'q_curr': (2, 4), 'q_target_next': (2, 4)}
[DogFightEnv][actor_critic_smoke] q_debug_first=
  {'q_concat_shape': (2, 4, 5), 'q_action_probe_first_feature': [0.0, 0.2, 0.4, 0.6]}
[DogFightEnv][actor_critic_smoke] PASS
```

2026-05-18 확인 결과:
- `qf_encoder`, `qf_twin_encoder`, `target_qf_encoder`, `target_qf_twin_encoder` 모두
  `get_initial_state()`를 가진 recurrent encoder로 확인했다.
- `q_concat_shape=(2,4,5)`로 `obs_dim=3`, `action_dim=2`의 `[obs, action]` concat을
  확인했다.
- rollout Q, actor loss Q, target Q, target twin Q가 모두 같은 helper를 통과했다.

### 5.2 DogFightEnv 학습 smoke

```powershell
ray stop --force

cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 10 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --lstm-scope actor_critic `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io > lstm_smoke_train.txt
```

기대 출력:

```text
label=train_continuous
obs_shape=(..., 8, 16)
next_obs_shape=(..., 8, 16)
actions_shape=(..., 8, 4)
seq_lens=...
state_in=...
next_state_in=...
state_out=...
next_state_out=...
obs_probe_first_feature=[시간축으로 연속적인 값들]
```

주의:
- `seq_lens`가 모두 1이고 `obs_probe_first_feature`가 `[값, 0, 0, ...]` 형태면
  실제 sequence가 아니라 padding된 transition batch다.
- 이 경우 replay buffer sampling 경로를 다시 확인한다.

로그 자동 점검:

```powershell
python RLLibLstm\tools\analyze_lstm_smoke_log.py `
  DogFightEnv\MyTrainEnv\lstm_smoke_train.txt --max-seq-len 8
```

`actor_critic` recurrent Q까지 자동 점검:

```powershell
python RLLibLstm\tools\analyze_lstm_smoke_log.py `
  DogFightEnv\MyTrainEnv\lstm_actor_critic_smoke_train.txt `
  --max-seq-len 8 --expect-q-debug
```

2026-05-19 actor_critic smoke 로그 확인 결과:

```text
[DogFightEnv][lstm_log_analyzer] inference_records= 20
[DogFightEnv][lstm_log_analyzer] train_records= 20
[DogFightEnv][lstm_log_analyzer] seq_lens_counts= 1:1, 3:2, 5:1, 8:577
[DogFightEnv][lstm_log_analyzer] q_debug_records= 120
[DogFightEnv][lstm_log_analyzer] q_debug_label_counts= q_curr_qf:20, q_curr_qf_twin:20, qf_rollout:20, qf_twin_rollout:20, target_qf:20, target_qf_twin:20
[DogFightEnv][lstm_log_analyzer] result=PASS
```

판단 근거:
- `q_debug_records=120`은 20개 train forward마다 6개 Q 경로가 모두 기록됐다는 의미다.
- `q_concat_shape=(29,8,20)`은 `obs_dim=16`, `action_dim=4`의 `[obs, action]` concat
  계약과 일치한다.
- `q_state_h=(29,1,64)`, `q_out_shape=(29,8)`로 Q LSTM zero-state 및 Q output shape도
  계약과 일치한다.

### 5.2.1 Actor-Critic 30 Iteration Stability Smoke

Debug 출력을 끄고 30 iteration 안정성 run을 수행했다.

```powershell
ray stop --force
cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 30 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --lstm-scope actor_critic `
  --lstm-cell-size 64 --max-seq-len 8 `
  --output-tag lstm_actor_critic_30it `
  > lstm_actor_critic_30it.txt
```

2026-05-19 확인 결과:
- iteration `0~29` 완료.
- `Traceback`, `ValueError`, `RuntimeError`, `Exception`, `ERROR` 모두 0회.
- `actor_loss`, `critic_loss`, `alpha_loss`, `alpha`는 update 시작 후 모두 finite.
- 마지막 learner metric:
  - `actor_loss=-9.2940`
  - `critic_loss=0.0547`
  - `alpha=0.9003`
- lightweight bundle과 training record 저장 완료.
- `Reward=[nan]`, `WinRate=[nan]`, `WEZ_ep=[nan]`은 iteration 13 이후 나타났지만,
  30 iteration 동안 완료 episode가 1개뿐이라 새 episode가 없는 result window의 env
  aggregate가 비어 생긴 값으로 판단한다. learner metric NaN/inf는 확인되지 않았다.

판단 근거:
- recurrent Q 구현이 수치적으로 터진 경우 learner loss/alpha 또는 backward에서 예외가
  먼저 나타날 가능성이 크다.
- 이번 로그에서는 학습 update가 지속되고 bundle 저장까지 완료됐으므로, 다음 평가는
  episode 표본 수를 늘리는 run에서 수행한다.

### 5.2.2 Actor-Critic 100 Iteration Stability Smoke

2026-05-19 확인 결과:
- iteration `0~99` 완료.
- sampled steps `11648`, completed episodes `4`.
- actor/critic/alpha learner metric은 update 시작 후 97개 row에서 numeric.
- 마지막 learner metric:
  - `actor_loss=-17.2414`
  - `critic_loss=0.0329`
  - `alpha_loss=-2.5497`
  - `alpha=0.6847`
- lightweight bundle 저장 완료.
- `RLLibLstm/tools/smoke_rl_action_provider_lstm_bundle.py --bundle-dir ...lstm_actor_critic_100it`
  실행 결과 actor recurrent state 유지와 reset 초기화 확인.

주의:
- 이 run은 `--debug-io` 없이 실행됐으므로 `analyze_lstm_smoke_log.py --expect-q-debug`의
  LSTM/Q debug marker 검증 대상은 아니다. Q 입출력 계약은 10 iteration debug smoke와
  no-Ray actor_critic module smoke에서 확인한다.
- 100 iteration 동안 완료 episode가 4개뿐이라 reward/win/WEZ 성능 판단에는 표본이
  부족하다. `Reward=[nan]`, `WinRate=[nan]`, `WEZ_ep=[nan]`가 대부분 iteration에
  나타나는 것은 새 완료 episode가 없는 result window의 env aggregate로 판단한다.

현재 2026-05-18 smoke 로그 기준 기대 요약:

```text
[DogFightEnv][lstm_log_analyzer] train_records= 20
[DogFightEnv][lstm_log_analyzer] seq_lens_counts= 2:1, 3:1, 8:579
[DogFightEnv][lstm_log_analyzer] seq_lens_max= 8
[DogFightEnv][lstm_log_analyzer] result=PASS
```

판단 근거:
- `seq_lens`가 모두 1이 아니고 대부분 8로 들어가면 replay가 실제 sequence slice를
  만들고 있다는 의미다.
- `obs_probe_first_feature`는 도메인 feature 의미에 따라 증가/감소 방향이 달라질 수
  있으므로, 자동 검증기는 padding 여부를 강하게 판정하고 방향성은 사람이 볼 수 있게
  `increasing/decreasing/mixed`로 출력한다.

### 5.3 RLActionProvider no-Ray bundle smoke

Ray cluster를 띄우지 않고 lightweight bundle의 RLModule state와 provider recurrent
state 유지 계약을 확인한다.

```powershell
python RLLibLstm\tools\smoke_rl_action_provider_lstm_bundle.py
```

기대 출력:

```text
[DogFightEnv][provider_smoke] use_lstm_sac= True
[DogFightEnv][RLActionProvider][LSTM_IO] obs_shape=(1, 1, 16) seq_lens=tensor([1])
[DogFightEnv][provider_smoke] step=0 action_shape=(4,) ... state_norm=...
[DogFightEnv][provider_smoke] after_reset_state=None
```

### 5.3.1 Actor-Critic LSTM 학습 재개

Lightweight bundle 기반 weight-only 재시작은 `--init-bundle`을 사용한다. 2026-05-19
이후 `train_rllib.py`와 `train_curriculum.py`는 bundle metadata를 읽어 SAC LSTM
architecture 인자를 자동 복원한다.

```powershell
python train_rllib.py --algorithm sac `
  --init-bundle artifacts\models\f16_single_agent\lstm_actor_critic_100it `
  --iterations 100 --output-tag lstm_actor_critic_resume
```

자동 복원이 일어나면 다음 marker가 출력된다.

```text
[DogFightEnv][LSTM_RESUME] init_bundle=... use_lstm_sac=True lstm_scope=actor_critic lstm_cell_size=64 max_seq_len=8
```

복원되는 값:

- `use_lstm_sac=True`
- `lstm_scope=actor_critic`
- `lstm_cell_size`
- `max_seq_len`

판단 근거:
- lightweight bundle은 policy/RLModule weight만 담고 optimizer와 replay buffer는 새로
  시작한다.
- actor_critic LSTM은 Q/twin/target Q encoder 구조도 달라지므로, 새 Algorithm build
  전에 저장 bundle과 같은 LSTM architecture로 config를 맞춰야 한다.
- `--init-bundle`은 architecture를 자동 복원하지만 관측 모드와 관측 모듈은 학습 당시와
  같은 값을 쓰는지 metadata와 함께 확인한다.

Native RLlib checkpoint 재개는 전체 Algorithm 상태를 보존한다. 단, 현재 학습 스크립트는
`algorithm.restore()` 전에 `config.build_algo()`를 수행하므로 checkpoint와 같은
architecture 옵션을 명시하는 것이 안전하다.

```powershell
python train_rllib.py --algorithm sac `
  --use-lstm-sac --lstm-scope actor_critic --lstm-cell-size 64 --max-seq-len 8 `
  --restore-checkpoint artifacts\checkpoints\f16_single_agent\<tag>\<checkpoint_dir> `
  --iterations 100 --output-tag lstm_actor_critic_native_resume
```

Native checkpoint를 만들려면 원 학습 때 `--save-native-checkpoint`를 함께 사용한다.

### 5.4 build_algorithm_from_bundle Ray smoke

실제 추론 경로에서 사용하는 `build_algorithm_from_bundle()`를 검증한다. 이 smoke는
Ray local cluster와 `config.build_algo()`를 사용하므로 실행 전 기존 Ray node를 정리한다.

```powershell
ray stop --force

python RLLibLstm\tools\smoke_build_algorithm_from_bundle_lstm.py
```

기대 출력:

```text
[DogFightEnv][RLlibConfig][LSTM_RESTORE] use_lstm_sac=True max_seq_len=8 lstm_cell_size=64
[DogFightEnv][ray_bundle_smoke] use_lstm_sac= True
[DogFightEnv][RLActionProvider][LSTM_IO] obs_shape=(1, 1, 16) seq_lens=tensor([1])
[DogFightEnv][ray_bundle_smoke] step=0 action_shape=(4,) ... state_norm=...
[DogFightEnv][ray_bundle_smoke] after_reset_state=None
```

판단 근거:
- no-Ray smoke는 provider와 RLModule state 계약만 본다.
- Ray smoke는 metadata 기반 Algorithm 복원, weight loading, provider 추론까지 실제
  production 복원 경로를 확인한다.

2026-05-18 확인 결과:
- 실제 사용자 실행에서 `use_lstm_sac=True`가 확인됐다.
- `RLlibSAC`와 `RLActionProvider` 모두 `obs_shape=(1, 1, 16)`,
  `seq_lens=tensor([1])`, `state_in/state_out=(1,1,64)`를 출력했다.
- step별 `state_norm`이 `5.344554 -> 8.083995 -> 9.558984`로 변해 recurrent state가
  다음 step으로 유지되는 것을 확인했다.
- `after_reset_state=None`으로 episode reset 계약도 확인했다.

주의:
- lightweight bundle의 `algorithm_config._model_config`는 JSON 저장 과정에서
  `DefaultModelConfig(...)` 문자열 또는 dict로 직렬화될 수 있다.
- Ray 2.54 `AlgorithmConfig.model_config`는 실제 dataclass instance를 기대하므로,
  `build_algorithm_from_bundle()`은 직렬화된 `_model_config`를 제거한 뒤 bundle
  metadata의 `use_lstm_sac`, `lstm_cell_size`, `max_seq_len`으로 다시 구성한다.
- 이 재구성이 동작하면 위의 `[DogFightEnv][RLlibConfig][LSTM_RESTORE]` marker가 출력된다.

### 5.5 Unreal policy reset smoke

Unreal UDP나 Ray를 띄우지 않고, Unreal command policy reset이 action provider reset으로
전달되는지 확인한다.

```powershell
python RLLibLstm\tools\smoke_unreal_policy_lstm_reset.py
```

기대 출력:

```text
[DogFightEnv][unreal_policy_reset_smoke] MyTrainEnv provider_reset_calls=1 lightweight_reset_calls=1
[DogFightEnv][unreal_policy_reset_smoke] Release provider_reset_calls=1 lightweight_reset_calls=1
[DogFightEnv][unreal_policy_reset_smoke] result=PASS
```

판단 근거:
- Unreal client는 `MT_Init` 수신 시 `command_policy.reset(...)`을 호출한다.
- LSTM 추론에서는 이 reset이 `RLActionProvider.reset()`까지 전달되어야 이전 episode의
  recurrent state가 새 episode로 섞이지 않는다.
- `DOGFIGHT_RNNSAC_DEBUG=1`이면 reset 시 다음 marker로 육안 확인할 수 있다.

```text
[DogFightEnv][RLActionProvider][LSTM_RESET] had_state=True frame_index=...
```

---

## 6. 추론 적용

`RLActionProvider`는 LSTM module을 감지하면 다음 입력으로 호출한다.

```text
obs: (1, 1, obs_dim)
STATE_IN: {'h': (1, 1, cell), 'c': (1, 1, cell)}
SEQ_LENS: [1]
```

반환된 `STATE_OUT`을 다음 step의 `STATE_IN`으로 저장한다.

episode boundary에서는 `RLActionProvider.reset()`을 호출해야 한다.

판단 근거:
- 학습이 recurrent state를 사용했는데 추론에서 state를 유지하지 않으면 학습/추론
  입력 계약이 달라진다.

---

## 7. PrioritizedEpisodeReplayBuffer를 다시 쓰는 방안

### 7.1 왜 prioritized replay를 쓰고 싶은가

장점:
- TD error가 큰 sample을 더 자주 학습해 sample efficiency가 좋아질 수 있다.
- sparse reward 또는 드문 교전 이벤트가 중요한 환경에서는 의미 있는 transition을
  더 자주 재사용할 가능성이 있다.

단점:
- priority bias를 importance sampling weight로 보정해야 한다.
- LSTM에서는 timestep priority와 sequence priority의 계약을 명확히 정해야 한다.
- 잘못 구현하면 좋은 sample을 더 뽑는 것이 아니라 짧은 transition/padding을 더
  많이 뽑는 문제가 된다.

### 7.2 현재 Ray 2.54에서 바로 쓰면 안 되는 이유

확인된 현상:

```text
obs_shape=(256, 8, 16)
seq_lens=tensor([1, 1, ...])
obs_probe_first_feature=[0.99945, 0.0, 0.0, ...]
ValueError: The truth value of an array with more than one element is ambiguous
```

판단 근거:
- `PrioritizedEpisodeReplayBuffer`는 `sample_episodes=True` 경로에서 sequence length 1
  중심으로 episode fragment를 반환한다.
- learner의 TD error는 sequence shape `(B,T)`가 될 수 있는데, priority update는 scalar
  priority를 기대한다.

### 7.3 권고 구현안 A: prioritized sequence buffer 패치

목표:
- `PrioritizedEpisodeReplayBuffer`가 `batch_length_T > 0`일 때 실제 sequence episode
  slice를 반환하도록 만든다.

구현 대상:
- `ray/rllib/utils/replay_buffers/prioritized_episode_buffer.py`

핵심 변경:
1. `sample(..., sample_episodes=True, batch_length_T=8)`에서 transition 1-step 생성
   대신 `episode.slice(slice(episode_ts, actual_length), len_lookback_buffer=lookback)`
   경로를 사용한다.
2. `actual_length`는 `episode_ts + batch_length_T + lookback` 기준으로 잡되 episode
   끝을 넘으면 resample하거나 짧은 valid length를 명시한다.
3. `weights`와 `n_step` extra output은 sequence 길이에 맞게 만든다.
4. `_last_sampled_indices`는 sequence당 하나만 저장한다.
5. debug print를 추가한다.

권고 debug marker:

```text
[DogFightEnv][PrioritizedSeqReplay][SAMPLE]
batch_length_T=8 actual_len=... lookback=... sampled_index=...
```

2026-05-18 구현 상태:
- `RLLibLstm/ray_2_54_0_patched/ray/rllib/utils/replay_buffers/prioritized_episode_buffer.py`
  에 sequence sampling patch를 추가했다.
- `batch_length_T>1`일 때 실제 `episode.slice(...)`를 반환한다.
- priority tree index는 sequence당 하나만 `_last_sampled_indices`에 저장한다.
- 설치본 Ray에도 동일 파일을 적용했고, `--installed` smoke를 통과했다.

판단 근거:
- sequence 하나에 priority 하나를 두는 방식이 가장 단순하고, prioritized replay의
  segment tree 구조와 충돌이 적다.

### 7.4 권고 구현안 B: TD error priority scalar reduction

목표:
- learner가 반환한 sequence TD error `(B,T)`를 replay buffer가 받을 수 있는 `(B,)`
  scalar priority로 줄인다.

구현 대상 후보:
- `ray/rllib/utils/replay_buffers/utils.py::update_priorities_in_episode_replay_buffer`
- 또는 `PrioritizedEpisodeReplayBuffer.update_priorities`

권고 방식:

```python
priority = np.abs(td_error)
if priority.ndim > 1:
    # valid mask가 있으면 padding을 제외한다.
    priority = priority.max(axis=1)
priority = np.asarray(priority, dtype=np.float64)
```

대안:
- `max(abs(td_error_t))`: sequence 안의 큰 오류를 보존한다.
- `mean(abs(td_error_t))`: sequence 전체 평균 난이도를 반영한다.

권고:
- 초기 구현은 `max`를 사용한다.

판단 근거:
- 공중전 이벤트는 특정 timestep에서 강하게 발생할 수 있어 평균이 희석할 가능성이
  있다.
- 다만 학습이 불안정하면 `mean` 또는 clipping을 비교한다.

2026-05-18 구현 상태:
- `PrioritizedEpisodeReplayBuffer.update_priorities()`에서 tensor/array TD-error를
  NumPy로 변환한다.
- `(B,T,...)` priority 입력은 `max(abs(td_error), axis=time_and_rest)`로 `(B,)`로
  축약한다.
- 길이가 sequence count와 맞지 않지만 reshape 가능한 경우 `(B,-1)`로 한 번 더 줄인다.

### 7.4.1 Prioritized sequence replay smoke

Ray Algorithm을 띄우지 않고 patched replay buffer만 검증한다.

```powershell
python RLLibLstm\tools\smoke_prioritized_sequence_replay.py --installed
```

기대 출력:

```text
[DogFightEnv][PrioritizedSeqReplay][SAMPLE] batch_length_T=8 actual_len=...
[DogFightEnv][prioritized_seq_smoke] lengths= [...]
[DogFightEnv][prioritized_seq_smoke] result=PASS
```

판단 근거:
- `lengths`에 8이 포함되어야 실제 sequence slice가 나오는 것이다.
- `probes`가 `[값, 0, 0, ...]` 형태면 transition padding 문제로 본다.
- `(B,T)` TD-error update가 예외 없이 끝나고 `_last_sampled_indices`가 clear되어야 한다.

### 7.5 권고 구현 순서

1. 현재 `EpisodeReplayBuffer` 기반 LSTM SAC가 10 iteration smoke를 통과하는지 확인한다.
2. `PrioritizedEpisodeReplayBuffer` sequence sampling patch를 만든다.
3. `update_priorities` sequence TD error reduction patch를 만든다.
4. `RLLibLstm/tools/smoke_sac_lstm_module_forward.py`와 별도 prioritized replay unit
   smoke를 추가한다.
5. DogFightEnv smoke에서 다음을 확인한다.

```text
label=train_continuous
obs_shape=(..., 8, 16)
seq_lens가 1로만 채워지지 않음
obs_probe_first_feature가 시간축 연속 값
priority_shape=(B,)
```

6. `EpisodeReplayBuffer`와 prioritized sequence replay를 같은 seed/짧은 iteration으로
   비교한다.

Opt-in 학습 smoke:

```powershell
ray stop --force

cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 10 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --use-lstm-prioritized-replay `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io > lstm_prioritized_smoke_train.txt
```

2026-05-18 확인 결과:
- `lstm_prioritized_smoke_train.txt` 분석 결과 `result=PASS`.
- `seq_lens_counts=1:1, 2:1, 3:1, 8:579`로 대부분 실제 length 8 sequence였다.
- `[DogFightEnv][PrioritizedSeqReplay][SAMPLE]` marker가 1260회 출력됐다.
- Traceback/ValueError/RuntimeError/Exception은 없었고 lightweight bundle 저장까지
  완료됐다.
- 단, iteration summary와 `training_log.csv`의 Actor/Critic/Alpha는 모두 `n/a`로 남아
  후속 단계에서 raw learner result key 또는 metric extractor를 확인해야 한다.

2026-05-18 후속 보강:
- `train_rllib.py`의 metric extractor가 RLlib result dict의 깊은 위치도 탐색하도록
  수정했다.
- `--debug-io` 상태에서 SAC LSTM loss metric을 찾지 못하면 다음 marker로 raw
  learner/loss/alpha key 후보를 1회 출력한다.

```text
[DogFightEnv][RLlibResult][LEARNER_KEYS] iteration=...
```

짧은 metric debug run:

```powershell
ray stop --force

cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 6 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --use-lstm-prioritized-replay `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io `
  > lstm_prioritized_metric_debug.txt
```

2026-05-18 metric debug 확인 결과:
- `lstm_prioritized_metric_debug.txt`에서 prioritized LSTM SAC가 예외 없이 bundle 저장까지
  완료됐다.
- 자동 analyzer 결과 `inference_records=20`, `train_records=19`,
  `seq_lens_counts=1:1, 2:1, 4:1, 7:3, 8:547`, `result=PASS`.
- 로그 marker 기준 `PrioritizedSeqReplay`는 380회, `label=train_continuous`는 19회,
  Traceback/ValueError/RuntimeError/Exception은 0회였다.
- learner batch는 `obs_shape=(29~30, 8, 16)`, `actions_shape=(29~30, 8, 4)`이고,
  `seq_lens` 대부분이 8이었다. episode boundary 주변에서 7/4/2 같은 짧은 sequence가
  일부 나타나는 것은 정상 범위다.
- `obs_probe_first_feature`가 시간축 방향으로 연속적인 값으로 찍혀, LSTM 입력이
  역순으로 쌓이거나 1-step padded sample만 들어가는 문제는 재현되지 않았다.
- iteration 4부터 Actor/Critic/Alpha가 각각 `-2.6182 / 2.0800 / 0.9999`,
  iteration 5에서 `-2.9952 / 1.7053 / 0.9970`으로 출력됐다.
- iteration 0~3의 `Alpha=0.6000`은 SAC temperature가 아니라 prioritized replay
  config의 `alpha=0.6`을 metric extractor가 잘못 주운 오탐으로 판단했다.
  이후 `config`/`replay_buffer_config` 경로는 nested metric 후보에서 제외하도록
  `train_rllib.py`를 보강했다.

판단 근거:
- prioritized replay debug marker가 `batch_length_T=8 actual_len=8` 중심으로 출력되고,
  `_forward_train_continuous()` debug가 `STATE_IN`, `STATE_OUT`, `NEXT_STATE_IN`,
  `NEXT_STATE_OUT` shape를 유지한다.
- SAC learner가 실제 update를 시작한 뒤에는 actor/critic/entropy alpha metric이
  RLlib result의 learner subtree에서 정상적으로 발견된다.

### 7.4.3 Prioritized sequence replay 무결성/강건성 검증

긴 학습 전에 replay buffer 자체를 독립 검증한다.

```powershell
python RLLibLstm\tools\verify_prioritized_sequence_replay_integrity.py `
  --target both --rounds 12 --batch-size-b 64 --batch-length-t 8
```

검증 항목:
- sampled sequence가 충분한 data에서 1-step으로만 떨어지지 않는지 확인한다.
- observation probe가 source episode 안에서 시간순 `+1` 연속성을 유지하는지 확인한다.
- sequence가 서로 다른 source episode offset bucket을 가로지르지 않는지 확인한다.
- `weights`와 `n_step` extra model output이 sequence length와 같은 길이로 유지되는지
  확인한다.
- priority tree index가 sampled sequence당 하나씩 기록되는지 확인한다.
- TD-error priority update가 scalar, vector, `(B,T)` matrix, flat `(B*T,)` 형태를
  모두 처리하는지 확인한다.

2026-05-18 확인 결과:
- installed site-packages와 `RLLibLstm` patched copy 모두 `result=PASS`.
- patched copy는 12라운드에서 sampled sequence 97개, length count
  `{5: 8, 6: 12, 7: 7, 8: 70}`.
- installed copy는 12라운드에서 sampled sequence 96개, length count
  `{5: 6, 6: 3, 7: 6, 8: 81}`.
- episode boundary 근처의 짧은 sequence도 chronological order, boundary, weight/n_step,
  priority update contract를 통과했다.

판단 근거:
- 이 검증은 Ray Algorithm build 없이 replay buffer class만 사용하므로, failure 원인이
  learner/connector가 아니라 replay buffer sequence sampling 자체인지 빠르게 분리할 수
  있다.
- installed와 patched copy를 동시에 검증해 다른 PC 이식 시 site-packages 적용 누락도
  조기에 잡을 수 있다.

### 7.4.4 EpisodeReplayBuffer 대 prioritized replay 비교

무결성 검증을 통과하면 다음 단계는 같은 조건에서 짧은 비교 run을 수행하는 것이다.

Episode baseline:

```powershell
ray stop --force

cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 20 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io `
  --output-tag lstm_episode_cmp > lstm_episode_cmp.txt
```

Prioritized sequence replay:

```powershell
ray stop --force

cd DogFightEnv\MyTrainEnv
python train_rllib.py --algorithm sac --iterations 20 --train-batch-size 64 `
  --rollout-fragment-length 8 --use-lstm-sac --use-lstm-prioritized-replay `
  --lstm-cell-size 64 --max-seq-len 8 --debug-io `
  --output-tag lstm_prioritized_cmp > lstm_prioritized_cmp.txt
```

비교:

```powershell
cd C:\Users\USER\workspace\DogFightEnv
python RLLibLstm\tools\compare_lstm_replay_training_logs.py `
  DogFightEnv\MyTrainEnv\artifacts\logs\f16_single_agent\lstm_episode_cmp\training_log.csv `
  DogFightEnv\MyTrainEnv\artifacts\logs\f16_single_agent\lstm_prioritized_cmp\training_log.csv `
  --tail 5
```

판단 기준:
- 짧은 20 iteration run에서는 reward/win/crash가 아직 `n/a`일 수 있으므로,
  우선 actor/critic/alpha numeric 안정성, replay memory, iteration time을 본다.
- episode가 충분히 종료되는 길이의 run에서는 reward, `ep_min_distance`,
  `ep_wez_steps`, crash rate, win rate를 함께 비교한다.
- prioritized가 loss 안정성 또는 접근 지표를 악화시키면 기본값은 계속
  `EpisodeReplayBuffer`로 유지한다.

비교 지표:
- actor loss, critic loss, alpha
- `ep_min_distance`
- `ep_wez_steps`
- crash rate
- win rate
- replay sample debug의 valid sequence length

---

## 7.4 sequence_v1 네트워크 구조 지정

Ray 1.9.2 시절 `RNNSAC_model.py`의 `fc -> lstm -> fc -> fc` 형태를 YAML로
명시하려면 `algo.network.type: sequence_v1`을 사용한다. 이 기능은 완전 임의
PyTorch graph가 아니라 Ray 2.54 New API `SACCatalog`에 안정적으로 매핑 가능한
layer sequence를 우선 지원한다.

예시:

```yaml
algo:
  name: sac
  train_batch_size: 64
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

현재 지원:
- layer block: `linear`, `activation`, `lstm`
- actor와 critic의 pre-LSTM MLP/post-LSTM MLP 분리
- Q/twin/target Q에 같은 critic spec 적용
- lightweight bundle metadata에 `network_spec` 저장
- `--init-bundle` 학습 재개 시 `network_spec` 자동 복원

제약:
- critic recurrent stack은 `lstm.scope: actor_critic`에서만 허용한다.
- critic 입력은 `input: obs_action`만 지원한다.
- Q concat 순서는 항상 `[obs, action]`이다.
- 완전 임의 PyTorch layer graph가 필요하면 다음 단계에서 프로젝트 내부 Python
  builder import 방식을 추가한다.

짧은 smoke:

```powershell
cd C:\Users\USER\workspace\DogFightEnv\DogFightEnv\MyTrainEnv
ray stop --force
python scripts\run_experiment.py experiments\sac_lstm_sequence_actor_critic_smoke.yaml
ray stop --force
python scripts\run_experiment.py experiments\sac_lstm_sequence_actor_critic_resume_smoke.yaml
```

resume smoke에서 다음 출력이 나오면 bundle metadata에서 LSTM/network 구조가 복원된
것이다.

```text
[DogFightEnv][LSTM_RESUME] ... network_type=sequence_v1
```

---

## 8. 구현 상태와 남은 일

완료:
- actor-LSTM RLlib SAC patch 보관
- project-side CLI/YAML 옵션
- `RLActionProvider` recurrent state 유지
- plain `EpisodeReplayBuffer` 기반 sequence replay 설정
- debug print로 입력/출력 계약 확인
- 2026-05-18 `EpisodeReplayBuffer` smoke에서 learner sequence batch 확인
  - `obs_shape=(29, 8, 16)`
  - `seq_lens` 대부분 8
  - `obs_probe_first_feature`가 시간축 연속 값
  - actor/critic/alpha loss 기록 및 lightweight bundle 저장
- 2026-05-18 lightweight bundle `RLActionProvider` smoke에서 recurrent inference 확인
  - provider 입력 `obs_shape=(1, 1, 16)`, `seq_lens=[1]`
  - `STATE_OUT`이 다음 step의 `STATE_IN`으로 유지됨
  - `reset()` 후 provider state가 `None`으로 초기화됨
- 2026-05-19 `sequence_v1` YAML network spec을 추가해 actor/Q의 pre-LSTM MLP,
  LSTM, post-LSTM MLP 구조를 bundle metadata까지 보존한다.

남은 일:
- prioritized와 non-prioritized 성능 비교
- 실제 `build_algorithm_from_bundle()` 기반 Ray Algorithm smoke와 Unreal 추론 검증은
  사용자 수행 항목으로 분리
- 완전 임의 PyTorch graph가 필요하면 `python_module` builder 방식을 별도 구현

---

## 9. 복구 방법

패치가 문제를 일으키면 `ray_2_54_0_original`의 파일을 site-packages에 되돌린다.

```powershell
copy RLLibLstm\ray_2_54_0_original\ray\rllib\algorithms\sac\default_sac_rl_module.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\default_sac_rl_module.py

copy RLLibLstm\ray_2_54_0_original\ray\rllib\algorithms\sac\torch\default_sac_torch_rl_module.py `
  %CONDA_PREFIX%\Lib\site-packages\ray\rllib\algorithms\sac\torch\default_sac_torch_rl_module.py
```

그 뒤 Ray를 정리한다.

```powershell
ray stop --force
```
