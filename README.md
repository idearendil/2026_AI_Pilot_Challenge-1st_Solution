# DogFightEnv — F-16 1v1 Dogfight RL

**한국어** · [English](README.en.md)

F-16 전투기 1대1 공중전(dogfight)을 위한 강화학습(RL) 환경과, 학습·평가·제출 도구 모음이다.
물리 시뮬레이션은 JSBSim 기반 F-16 FDM(flight dynamics model)을 사용하며, 대회 서버
(BattleServer)와 동일한 관측/행동/판정 규약을 따른다.

이 저장소는 다음 다섯 가지에 필요한 코드만 담고 있다.

1. **학습 pipeline 2종** — CPU 환경(`claude_code`)과 CUDA/GPU 환경(`cuda_fdm`)
2. **학습모델 → 제출 bundle 변환** — CPU/CUDA 각각
3. **성능 테스트 3종** — `power_test`, `final_power_test`, `league`
4. **전투 로그 저장 + 시각화(replay viewer)**
5. **제출 파일 빌드** — BattleServer_V1.2_VeryLow 에서 바로 구동

정책(policy)은 factorized categorical 을 쓰는 **MLP** actor 이며, 관측은 claude164r
(`claude_code/my_observation.py`, OBS_SIZE=214)로 두 pipeline 이 동일하다.

---

## 다운로드 (대용량 파일)

아래 두 파일은 용량이 커서 이 repo 에 포함하지 않고 GitHub Releases 로 제공한다.

- **로컬 대회 서버(visual simulator)** — [BattleServer_V1.2_VeryLow.zip](https://github.com/idearendil/AIP/releases/download/visual_simulator/BattleServer_V1.2_VeryLow.zip)
  압축을 풀어 **repo 루트에** `BattleServer_V1.2_VeryLow/` 폴더로 둔다. 제출 파일 구동·전투
  시각화에 쓰는 로컬 서버다(아래 5번 참고).
- **우리 팀 최종 제출 파일** — [final_submission.3-9.zip](https://github.com/idearendil/AIP/releases/download/submission_file/final_submission.3-9.zip)
  대회에 제출한 최종 모델의 실행 패키지(3-9 시나리오 학습본). 압축을 풀어 exe 를 실행하면
  `config.json` 의 서버로 접속한다(빌드 방법은 아래 5번 참고).

---

## 폴더 구조

```
Release/
├─ claude_code/           # CPU 학습 pipeline + 공용 tooling(테스트·제출·시각화)
│   ├─ train.py               # CPU PPO 학습 entrypoint
│   ├─ agents.py              # ownship/target agent 로딩·provider 생성 공용 유틸
│   ├─ power_test.py          # 두 agent N판 대결 → 승률·유의성
│   ├─ final_power_test.py    # 한 모델 vs 모든 baseline
│   ├─ league.py              # final_team_models 리그전 → 승률 heatmap
│   ├─ run_local_dogfight.py  # 로컬 대결 + tacview 로그 저장(시각화용)
│   ├─ build_submission.py    # 제출 exe/zip 빌더
│   ├─ submission_client.py   # 제출 실행 진입점(config.json 기반 UDP client)
│   └─ snapshot_to_bundle.py  # CPU 학습 snapshot → 제출 bundle
├─ cuda_fdm/              # CUDA/GPU 학습 pipeline (F-16 FDM 을 GPU 로 평탄화 포팅)
│   ├─ train_gpu.py           # GPU PPO 학습 entrypoint
│   ├─ rl_env.py, gpu_env.py, obs_reward.py, ppo_gpu.py, ic.py
│   └─ gpu_ckpt_to_bundle.py  # CUDA 체크포인트 → 제출 bundle
├─ baselines/            # BT·MPC 상대 8종(테스트용)
│   ├─ Release_MPC_team_share/  Stable_MPC_team_share/
│   ├─ Jeon_BT1  Jeon_BT2  Lee_BT1  Shin_BT_best  Shin_BT_def (.dll/.xml)
│   └─ unreal_bt_client.exe
├─ src/dogfight/         # env·provider 코어(state schema, action provider, unreal protocol)
├─ aircraft/  engine/    # JSBSim F-16 물리 asset (env 필수)
├─ tools/                # 전투 로그 replay viewer(dogfight_dashboard)
├─ final_team_models/    # league 입력(내 MLP + 팀원 모델 bundle)
├─ *.dll  *.py (루트)     # env runtime(JSBSimAIPLib.dll, DogFightEnvWrapper 등)
└─ BattleServer_V1.2_VeryLow/  # 로컬 대회 서버(Releases 에서 다운로드 — 위 참고)
```

> `runs/`, `artifacts/`, `wandb/`, `dist/`, `BattleServer_*`, `jsbsim/` 는 로컬 산출물/런타임이라
> `.gitignore` 로 제외된다(공개 repo 에는 올라가지 않음).

---

## 설치

```bash
pip install -r requirements.txt
```

- Python 3.11 권장. Windows(JSBSimAIPLib.dll 등 native DLL) 환경 기준.
- 모든 스크립트는 **저장소 루트에서** 실행한다(env 가 `aircraft/`·`engine/` 등 상대경로 asset 을
  로드하므로 CWD 가 루트여야 한다). BT rule XML 도 루트 기준 상대경로다.

---

## 1) 학습 pipeline

### CPU 환경 (`claude_code`)

```bash
python claude_code/train.py --iterations 50 --output-name team01 --output-tag ppo_v1 \
    --observation-module claude_code.my_observation
```

- 학습 결과는 `artifacts/models/<name>/<tag>/` 에 snapshot(.pt)으로, best iteration 은 2-파일
  bundle(`metadata.json` + `policy_weights.pkl.gz`)로 저장된다.

### CUDA/GPU 환경 (`cuda_fdm`)

수천 개 env 를 GPU 에서 병렬로 돌리는 PPO. 모델 구조·feature·action space·reward·exploiter
방식은 CPU 환경과 별개(각자 pipeline 그대로)이며, **초기 상태 분포**만 두 전투기가 옆에서
반대 방향을 보는 3-9 line(시나리오 A)과 정면 대치하는 head-on(시나리오 B)을 **4:1**로 섞는다
(`scenario_b_prob=0.2`, `cuda_fdm/rl_env.py`).

```bash
python -m cuda_fdm.train_gpu --save runs/gpu.pt
```

> wandb 로깅을 쓰려면 환경변수 `WANDB_API_KEY` 를 설정하거나 `wandb login` 을 먼저 실행한다
> (키는 소스에 하드코딩하지 않는다).

---

## 2) 학습모델 → 제출 bundle 변환

```bash
# CPU 학습 snapshot → bundle
python claude_code/snapshot_to_bundle.py --snapshot-dir claude_code/models/team01/ppo_v1 \
    --output-dir artifacts/cpu_ppo_final

# CUDA 학습 체크포인트 → bundle
python -m cuda_fdm.gpu_ckpt_to_bundle --ckpt runs/gpu.pt --output-dir artifacts/gpu_ppo_final
```

두 산출물 모두 동일한 2-파일 bundle 포맷이라 이후 도구(테스트·제출)에서 똑같이 쓴다.

---

## 3) 성능 테스트

세 도구 모두 **agent spec** 문자열로 슬롯을 지정한다:

| spec | 의미 |
|------|------|
| `bundle:<경로>` | CPU/CUDA 학습 bundle |
| `ckpt:<경로>` | CUDA 학습 체크포인트(runs/*.pt, 변환 없이 바로) |
| `bt:<이름>` | baselines/ BT DLL(Lee_BT1, Jeon_BT1, Jeon_BT2, Shin_BT_best, Shin_BT_def) |
| `release_mpc` / `stable_mpc` / `unreal_exe` | baselines/ 의 MPC·외부 BT exe |

### power_test — 두 agent N판 대결
슬롯마다 **10/60Hz**(`--*-hz`)와 신경망 **argmax/stochastic**(`--*-action`)을 따로 고른다.

```bash
python -m claude_code.power_test --ownship bundle:artifacts/gpu_ppo_final \
    --target bt:Lee_BT1 --games 100
python -m claude_code.power_test --ownship "ckpt:runs/gpu.pt" \
    --ownship-action argmax --target stable_mpc --games 100
```

### final_power_test — 한 모델 vs 모든 baseline(8종)
옵션 3개(`--hz`, `--action`, `--games`)를 한 번 지정하면 모든 baseline 에 동일 적용된다.

```bash
python -m claude_code.final_power_test --ownship bundle:artifacts/gpu_ppo_final --games 100
```

### league — final_team_models 리그전 → heatmap
`final_team_models/` 의 모델(내 MLP + 팀원)과 baseline MPC·cutoff exe 를 모든 쌍끼리 붙여
승률 행렬을 색칠한 heatmap PNG 를 만든다. 초기 분포는 학습과 동일한 **3-9 : head-on = 4:1**,
모든 신경망은 **argmax·10Hz**.

```bash
python -m claude_code.league --games 50 --num-workers 8 --out league_winrate.png
```

---

## 4) 전투 로그 저장 + 시각화

```bash
# 한 판 돌리고 tacview CSV + summary 로그 저장
python claude_code/run_local_dogfight.py --ownship bundle:artifacts/gpu_ppo_final \
    --target bt:Lee_BT1 --save-log

# 저장된 로그를 replay viewer 로 재생
python tools/web_log_viewer.py
```

`run_local_dogfight` 도 power_test 와 같은 agent spec/`--*-hz`/`--*-action` 옵션을 받는다.

---

## 5) 제출 파일 빌드 + BattleServer 구동

학습된 모델(CPU/CUDA bundle) 하나만 얼려 exe/zip 으로 만든다. 제어주기(10/60Hz)와
argmax/stochastic 은 `config.json`(`control_hz`, `deterministic`)에서 정한다.

```bash
python claude_code/build_submission.py --bundle-dir artifacts/gpu_ppo_final \
    --team-name team01 --server-ip 127.0.0.1 --server-port 9999
# → dist/submission/DogfightSubmission/  및  dist/submission.zip
```

BattleServer_V1.2_VeryLow([위 다운로드](#다운로드-대용량-파일))를 로컬에서 실행한 뒤, zip 을
풀고 `DogfightSubmission.exe` 를 실행하면 `config.json` 의 서버 주소로 접속해 전투가 진행된다.
우리 팀이 대회에 낸 최종 제출본은 위 다운로드의 `final_submission.3-9.zip` 로 받을 수 있다.

---

## 주의사항

- **CWD = 저장소 루트.** env 가 상대경로로 native DLL·aircraft·engine asset·BT rule XML 을
  로드한다.
- **BT rule XML.** BT DLL 은 `AIP_RULE_XML`(rule 경로)을 `JSBSimAIPLib.dll` 로드 시점에 한 번만
  읽는다. 각 테스트 스크립트가 import 전에 자동으로 세팅한다. 한 프로세스에서 서로 다른 rule 의
  BT 둘을 동시에 붙일 수는 없다.
- **관측 규약.** 두 학습 pipeline 모두 claude164r(`claude_code.my_observation`)로 학습하므로
  bundle·ckpt 를 슬롯 무관하게 서로 붙일 수 있다.
