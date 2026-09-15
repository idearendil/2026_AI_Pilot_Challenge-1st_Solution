# DogFightEnv — F-16 1v1 Dogfight RL

[한국어](README.md) · **English**

A reinforcement-learning (RL) environment for F-16 1v1 air combat (dogfighting), together with
tools for training, evaluation, and submission. The physics use a JSBSim-based F-16 flight
dynamics model (FDM) and follow the same observation/action/scoring conventions as the
competition server (BattleServer).

This repository contains only the code needed for five things:

1. **Two training pipelines** — a CPU environment (`claude_code`) and a CUDA/GPU environment (`cuda_fdm`)
2. **Trained model → submission bundle conversion** — for CPU and CUDA respectively
3. **Three performance tests** — `power_test`, `final_power_test`, `league`
4. **Battle-log saving + visualization (replay viewer)**
5. **Submission packaging** — runs directly on BattleServer_V1.2_VeryLow

The policy is an **MLP** actor with a factorized categorical head. Both pipelines share the same
observation, claude164r (`claude_code/my_observation.py`, OBS_SIZE=214).

---

## Downloads (large files)

The two files below are too large to keep in this repo, so they are provided via GitHub Releases.

- **Local competition server (visual simulator)** — [BattleServer_V1.2_VeryLow.zip](https://github.com/idearendil/AIP/releases/download/visual_simulator/BattleServer_V1.2_VeryLow.zip)
  Unzip it into the **repo root** as `BattleServer_V1.2_VeryLow/`. It is the local server used to run
  submissions and visualize battles (see section 5 below).
- **Our team's final submission** — [final_submission.3-9.zip](https://github.com/idearendil/AIP/releases/download/submission_file/final_submission.3-9.zip)
  The runnable package of the final model submitted to the competition (trained on the 3-9 scenario).
  Unzip and run the exe; it connects to the server in `config.json` (build steps in section 5).

---

## Layout

```
Release/
├─ claude_code/           # CPU training pipeline + shared tooling (tests, submission, viz)
│   ├─ train.py               # CPU PPO training entrypoint
│   ├─ agents.py              # shared ownship/target agent loading & provider factory
│   ├─ power_test.py          # two agents, N games → win rate & significance
│   ├─ final_power_test.py    # one model vs every baseline
│   ├─ league.py              # final_team_models round-robin → win-rate heatmap
│   ├─ run_local_dogfight.py  # local match + tacview log (for visualization)
│   ├─ build_submission.py    # submission exe/zip builder
│   ├─ submission_client.py   # submission entrypoint (config.json UDP client)
│   └─ snapshot_to_bundle.py  # CPU training snapshot → submission bundle
├─ cuda_fdm/              # CUDA/GPU training pipeline (F-16 FDM flattened onto the GPU)
│   ├─ train_gpu.py           # GPU PPO training entrypoint
│   ├─ rl_env.py, gpu_env.py, obs_reward.py, ppo_gpu.py, ic.py
│   └─ gpu_ckpt_to_bundle.py  # CUDA checkpoint → submission bundle
├─ baselines/            # 8 BT/MPC opponents (for testing)
│   ├─ Release_MPC_team_share/  Stable_MPC_team_share/
│   ├─ Jeon_BT1  Jeon_BT2  Lee_BT1  Shin_BT_best  Shin_BT_def (.dll/.xml)
│   └─ unreal_bt_client.exe
├─ src/dogfight/         # env/provider core (state schema, action provider, unreal protocol)
├─ aircraft/  engine/    # JSBSim F-16 physics assets (required by the env)
├─ tools/                # battle-log replay viewer (dogfight_dashboard)
├─ final_team_models/    # league inputs (my MLP + teammates' model bundles)
├─ *.dll  *.py (root)    # env runtime (JSBSimAIPLib.dll, DogFightEnvWrapper, ...)
└─ BattleServer_V1.2_VeryLow/  # local competition server (download from Releases — see above)
```

> `runs/`, `artifacts/`, `wandb/`, `dist/`, `BattleServer_*`, `jsbsim/` are local outputs/runtime
> and are excluded via `.gitignore` (not part of the public repo).

---

## Install

```bash
pip install -r requirements.txt
```

- Python 3.11 recommended; Windows (native DLLs such as JSBSimAIPLib.dll).
- Run every script **from the repository root** (the env loads `aircraft/`·`engine/` and other
  assets by relative path, so the CWD must be the root). BT rule XML paths are also root-relative.

---

## 1) Training pipelines

### CPU environment (`claude_code`)

```bash
python claude_code/train.py --iterations 50 --output-name team01 --output-tag ppo_v1 \
    --observation-module claude_code.my_observation
```

Snapshots (`.pt`) are written to `artifacts/models/<name>/<tag>/`; the best iteration is also saved
as a 2-file bundle (`metadata.json` + `policy_weights.pkl.gz`).

### CUDA/GPU environment (`cuda_fdm`)

PPO running thousands of envs in parallel on the GPU. Model structure, features, action space,
reward, and exploiter scheme are each pipeline's own (kept as-is). Only the **initial-state
distribution** mixes a 3-9 line (scenario A, the two jets abreast facing opposite ways) and a
head-on (scenario B, facing each other) at **4:1** (`scenario_b_prob=0.2`, `cuda_fdm/rl_env.py`).

```bash
python -m cuda_fdm.train_gpu --save runs/gpu.pt
```

> For wandb logging, set the `WANDB_API_KEY` environment variable or run `wandb login` first (no
> key is hardcoded in the source).

---

## 2) Trained model → submission bundle

```bash
# CPU training snapshot → bundle
python claude_code/snapshot_to_bundle.py --snapshot-dir claude_code/models/team01/ppo_v1 \
    --output-dir artifacts/cpu_ppo_final

# CUDA training checkpoint → bundle
python -m cuda_fdm.gpu_ckpt_to_bundle --ckpt runs/gpu.pt --output-dir artifacts/gpu_ppo_final
```

Both produce the same 2-file bundle format and are used identically by the tools below.

---

## 3) Performance tests

All three take an **agent spec** string per slot:

| spec | meaning |
|------|---------|
| `bundle:<path>` | CPU/CUDA training bundle |
| `ckpt:<path>` | CUDA training checkpoint (runs/*.pt, used directly) |
| `bt:<name>` | baselines/ BT DLL (Lee_BT1, Jeon_BT1, Jeon_BT2, Shin_BT_best, Shin_BT_def) |
| `release_mpc` / `stable_mpc` / `unreal_exe` | baselines/ MPC and external BT exe |

### power_test — two agents, N games
Choose **10/60Hz** (`--*-hz`) and neural-net **argmax/stochastic** (`--*-action`) per slot.

```bash
python -m claude_code.power_test --ownship bundle:artifacts/gpu_ppo_final \
    --target bt:Lee_BT1 --games 100
python -m claude_code.power_test --ownship "ckpt:runs/gpu.pt" \
    --ownship-action argmax --target stable_mpc --games 100
```

### final_power_test — one model vs all 8 baselines
Set the three options (`--hz`, `--action`, `--games`) once; they apply to every baseline.

```bash
python -m claude_code.final_power_test --ownship bundle:artifacts/gpu_ppo_final --games 100
```

### league — final_team_models round-robin → heatmap
Runs every pair among the `final_team_models/` models (my MLP + teammates) plus the baseline MPC
and cutoff exe, and renders a win-rate heatmap PNG. Initial distribution matches training
(**3-9 : head-on = 4:1**); all nets run **argmax·10Hz**.

```bash
python -m claude_code.league --games 50 --num-workers 8 --out league_winrate.png
```

---

## 4) Battle log + visualization

```bash
# Run one match and save tacview CSV + summary
python claude_code/run_local_dogfight.py --ownship bundle:artifacts/gpu_ppo_final \
    --target bt:Lee_BT1 --save-log

# Replay the saved log
python tools/web_log_viewer.py
```

`run_local_dogfight` takes the same agent spec / `--*-hz` / `--*-action` options as power_test.

---

## 5) Submission packaging + running on BattleServer

Freezes a single trained model (CPU/CUDA bundle) into an exe/zip. Control rate (10/60Hz) and
argmax/stochastic are set in `config.json` (`control_hz`, `deterministic`).

```bash
python claude_code/build_submission.py --bundle-dir artifacts/gpu_ppo_final \
    --team-name team01 --server-ip 127.0.0.1 --server-port 9999
# → dist/submission/DogfightSubmission/  and  dist/submission.zip
```

Start BattleServer_V1.2_VeryLow ([download above](#downloads-large-files)) locally, then unzip and
run `DogfightSubmission.exe`; it connects to the server address in `config.json` and the match runs.
Our team's actual competition submission is available as `final_submission.3-9.zip` in the downloads
above.

---

## Notes

- **CWD = repo root.** The env loads native DLLs, aircraft/engine assets, and BT rule XML by
  relative path.
- **BT rule XML.** A BT DLL reads `AIP_RULE_XML` (the rule path) only once, when `JSBSimAIPLib.dll`
  loads. Each test script sets it before import automatically. Two BTs with different rules cannot
  run in the same process at once.
- **Observation convention.** Both training pipelines use claude164r
  (`claude_code.my_observation`), so bundles and checkpoints can be matched against each other in
  either slot.
