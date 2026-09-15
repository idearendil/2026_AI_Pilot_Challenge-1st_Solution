# DogFightEnv — F-16 1v1 Dogfight RL

[![한국어](https://img.shields.io/badge/한국어-2563eb?style=for-the-badge)](README.md)
[![English](https://img.shields.io/badge/English-64748b?style=for-the-badge)](README.en.md)

F-16 전투기 1대1 공중전(dogfight)을 강화학습(RL)으로 다루기 위한 환경과, 학습부터 평가·제출까지
필요한 도구를 함께 모아 둔 저장소다. 물리 엔진은 JSBSim 기반 F-16 FDM(flight dynamics model)을
쓰고, 관측·행동·승패 판정은 모두 대회 서버(BattleServer)와 같은 규약을 따른다.

여기에는 다음 다섯 가지에 꼭 필요한 코드만 남겨 두었다.

1. **학습 pipeline 2종** — CPU 환경(`claude_code`)과 CUDA/GPU 환경(`cuda_fdm`)
2. **학습모델 → 제출 bundle 변환** — CPU·CUDA 각각
3. **성능 테스트 3종** — `power_test`, `final_power_test`, `league`
4. **전투 로그 저장 + 시각화(replay viewer)**
5. **제출 파일 빌드** — BattleServer_V1.2_VeryLow 에서 바로 구동

정책(policy)은 factorized categorical 을 쓰는 **MLP** actor 이고, 관측은 두 pipeline 이 똑같이
claude164r(`claude_code/my_observation.py`, OBS_SIZE=214)를 쓴다.

---

## 다운로드 (대용량 파일)

아래 두 파일은 덩치가 커서 저장소에 직접 넣지 않고 GitHub Releases 로 따로 받도록 해 두었다.

- **로컬 대회 서버 (visual simulator)** — [BattleServer_V1.2_VeryLow.zip](https://github.com/idearendil/AIP/releases/download/visual_simulator/BattleServer_V1.2_VeryLow.zip)
  받은 뒤 압축을 풀어 저장소 **루트에** `BattleServer_V1.2_VeryLow/` 폴더째로 두면 된다. 제출 파일을
  돌려 보거나 전투를 눈으로 확인할 때 쓰는 로컬 서버다(자세한 건 아래 5번).
- **우리 팀 최종 제출 파일** — [final_submission.3-9.zip](https://github.com/idearendil/AIP/releases/download/submission_file/final_submission.3-9.zip)
  실제 대회에 제출한 최종 모델의 실행 패키지(3-9 시나리오 학습본)다. 압축을 풀고 exe 를 실행하면
  `config.json` 에 적힌 서버로 접속한다(직접 빌드하는 방법은 아래 5번).

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

> `runs/`, `artifacts/`, `wandb/`, `dist/`, `BattleServer_*`, `jsbsim/` 는 로컬에서 생기는
> 산출물·런타임이라 `.gitignore` 로 빼 두었다(공개 repo 에는 올라가지 않는다).

---

## 설치

```bash
pip install -r requirements.txt
```

- Python 3.11 을 권장하며, JSBSimAIPLib.dll 같은 native DLL 을 쓰는 Windows 환경을 기준으로 한다.
- 모든 스크립트는 **저장소 루트에서** 실행한다. env 가 `aircraft/`·`engine/` 같은 asset 을 상대경로로
  읽기 때문에 작업 디렉터리(CWD)가 루트여야 하고, BT rule XML 도 루트 기준 상대경로다.

---

## 1) 학습 pipeline

### CPU 환경 (`claude_code`)

```bash
python claude_code/train.py --iterations 50 --output-name team01 --output-tag ppo_v1 \
    --observation-module claude_code.my_observation
```

학습이 진행되면 매 iteration snapshot(.pt)이 `artifacts/models/<name>/<tag>/` 에 쌓이고, 가장 좋은
iteration 은 2-파일 bundle(`metadata.json` + `policy_weights.pkl.gz`)로 저장된다.

### CUDA/GPU 환경 (`cuda_fdm`)

수천 개의 env 를 GPU 에서 한꺼번에 굴리는 PPO 다. 모델 구조·feature·action space·reward·exploiter
방식은 CPU 환경과 별개로 각자의 pipeline 을 그대로 쓰고, **초기 상태 분포**만 두 전투기가 옆에서
서로 반대 방향을 보는 3-9 line(시나리오 A)과 정면으로 마주 보는 head-on(시나리오 B)을 **4:1** 로
섞도록 해 두었다(`scenario_b_prob=0.2`, `cuda_fdm/rl_env.py`).

```bash
python -m cuda_fdm.train_gpu --save runs/gpu.pt
```

> wandb 로깅을 쓰고 싶으면 환경변수 `WANDB_API_KEY` 를 미리 넣거나 `wandb login` 을 한 번 실행해
> 두면 된다(키는 소스에 넣지 않는다).

---

## 2) 학습모델 → 제출 bundle 변환

```bash
# CPU 학습 snapshot → bundle
python claude_code/snapshot_to_bundle.py --snapshot-dir claude_code/models/team01/ppo_v1 \
    --output-dir artifacts/cpu_ppo_final

# CUDA 학습 체크포인트 → bundle
python -m cuda_fdm.gpu_ckpt_to_bundle --ckpt runs/gpu.pt --output-dir artifacts/gpu_ppo_final
```

둘 다 똑같은 2-파일 bundle 포맷으로 나오기 때문에, 이후의 테스트·제출 도구에서 구분 없이 쓸 수 있다.

---

## 3) 성능 테스트

세 도구 모두 슬롯에 넣을 상대를 **agent spec** 문자열로 지정한다.

| spec | 의미 |
|------|------|
| `bundle:<경로>` | CPU/CUDA 학습 bundle |
| `ckpt:<경로>` | CUDA 학습 체크포인트(runs/*.pt, 변환 없이 바로) |
| `bt:<이름>` | baselines/ BT DLL(Lee_BT1, Jeon_BT1, Jeon_BT2, Shin_BT_best, Shin_BT_def) |
| `release_mpc` / `stable_mpc` / `unreal_exe` | baselines/ 의 MPC·외부 BT exe |

### power_test — 두 agent N판 대결
슬롯마다 **10/60Hz**(`--*-hz`)와, 신경망이라면 **argmax/stochastic**(`--*-action`)을 따로 고를 수 있다.

```bash
python -m claude_code.power_test --ownship bundle:artifacts/gpu_ppo_final \
    --target bt:Lee_BT1 --games 100
python -m claude_code.power_test --ownship "ckpt:runs/gpu.pt" \
    --ownship-action argmax --target stable_mpc --games 100
```

### final_power_test — 한 모델 vs 모든 baseline(8종)
옵션 세 개(`--hz`, `--action`, `--games`)만 한 번 정하면 8종 baseline 전부에 똑같이 적용된다.

```bash
python -m claude_code.final_power_test --ownship bundle:artifacts/gpu_ppo_final --games 100
```

### league — final_team_models 리그전 → heatmap
`final_team_models/` 의 모델(내 MLP + 팀원)과 baseline MPC·cutoff exe 를 모든 쌍끼리 붙여, 승률
행렬을 색으로 칠한 heatmap PNG 를 만든다. 초기 분포는 학습과 똑같이 **3-9 : head-on = 4:1**,
신경망은 전부 **argmax·10Hz** 로 돈다.

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

`run_local_dogfight` 도 power_test 와 똑같은 agent spec/`--*-hz`/`--*-action` 옵션을 받는다.

---

## 5) 제출 파일 빌드 + BattleServer 구동

학습이 끝난 모델(CPU/CUDA bundle) 하나만 얼려서 exe/zip 으로 묶는다. 제어 주기(10/60Hz)와
argmax/stochastic 여부는 `config.json`(`control_hz`, `deterministic`)에서 정한다.

```bash
python claude_code/build_submission.py --bundle-dir artifacts/gpu_ppo_final \
    --team-name team01 --server-ip 127.0.0.1 --server-port 9999
# → dist/submission/DogfightSubmission/  및  dist/submission.zip
```

먼저 BattleServer_V1.2_VeryLow([위 다운로드](#다운로드-대용량-파일))를 로컬에서 띄운 다음, zip 을 풀고
`DogfightSubmission.exe` 를 실행하면 `config.json` 의 서버 주소로 접속해 전투가 시작된다. 우리 팀이
실제로 대회에 낸 최종 제출본은 위 다운로드의 `final_submission.3-9.zip` 로 바로 받아 볼 수 있다.

---

## 알아 둘 점

- **CWD 는 저장소 루트.** env 가 native DLL·aircraft·engine asset·BT rule XML 을 모두 상대경로로
  읽는다.
- **BT rule XML.** BT DLL 은 rule 경로(`AIP_RULE_XML`)를 `JSBSimAIPLib.dll` 이 로드되는 순간 딱 한 번만
  읽어 캐싱한다. 각 테스트 스크립트가 import 전에 알아서 세팅해 주며, 서로 rule 이 다른 BT 두 개를
  한 프로세스에서 동시에 붙일 수는 없다.
- **관측 규약.** 두 학습 pipeline 이 모두 claude164r(`claude_code.my_observation`)로 학습하므로,
  bundle 이든 ckpt 든 슬롯을 가리지 않고 서로 붙여 비교할 수 있다.
