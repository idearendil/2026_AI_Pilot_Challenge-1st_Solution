# -*- coding: utf-8 -*-
"""로컬 대결 + 리플레이(tacview) 로그 생성 — 시각화(dogfight_dashboard Replay 탭)용.

학습과 동일한 `make_env`(TierGatedDogFightEnv, 시간게이팅 damage)로 한 판을 돌리고, 끝나면
tacview CSV(ownship/target) + summary.json 로그를 저장한다. 저장된 로그는 replay viewer
(`python tools/web_log_viewer.py`)로 시각화해 전투를 다시 볼 수 있다.

ownship·target 슬롯에 넣을 수 있는 agent 는 power_test 와 동일(claude_code.agents):
  bundle:<경로> / ckpt:<경로> / bt:<이름> / release_mpc / stable_mpc / unreal_exe
슬롯마다 10Hz/60Hz(--*-hz)와 신경망 argmax/stochastic(--*-action)을 고른다.

예시:
  python claude_code/run_local_dogfight.py --ownship bundle:artifacts/gpu_ppo_final \
      --target bt:Lee_BT1 --save-log
  python -m claude_code.run_local_dogfight --ownship "ckpt:runs/mlp_768(only_headon).pt" \
      --ownship-action argmax --target stable_mpc --save-log
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── BT rule XML: claude_code.env_utils import 보다 먼저 세팅(DLL 이 import 시점에 캐싱) ──
from claude_code.agents import ENV_KEY, parse_agent_spec, bt_rule_for_specs  # noqa: E402


def _preparse_specs():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ownship", default="")
    pre.add_argument("--target", default="bt:Lee_BT1")
    known, _ = pre.parse_known_args()
    own = parse_agent_spec(known.ownship) if known.ownship else None
    tgt = parse_agent_spec(known.target) if known.target else None
    return own, tgt


def _apply_bt_rule_env():
    own, tgt = _preparse_specs()
    specs = [s for s in (own, tgt) if s]
    rule = bt_rule_for_specs(*specs) if specs else ""
    if rule:
        os.environ[ENV_KEY] = rule
        print(f"[local] BT rule XML = {rule} ({ENV_KEY})")


_apply_bt_rule_env()   # ← 반드시 아래 claude_code import 들보다 먼저!

import numpy as np  # noqa: E402

from claude_code import agents  # noqa: E402
from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="로컬 대결 + 리플레이(tacview) 로그 생성")
    p.add_argument("--ownship", required=True,
                   help="ownship agent spec. bundle:<경로> / ckpt:<경로> / bt:<이름> / "
                        "release_mpc / stable_mpc / unreal_exe")
    p.add_argument("--target", default="bt:Lee_BT1",
                   help="target agent spec(형식 동일, 기본 bt:Lee_BT1)")
    for side in ("ownship", "target"):
        p.add_argument(f"--{side}-hz", type=int, choices=[10, 60], default=10,
                       help=f"{side} 신경망 제어 주기(baseline 무시, 기본 10)")
        p.add_argument(f"--{side}-action", choices=["stochastic", "argmax"],
                       default="stochastic",
                       help=f"{side} 신경망 action 선택(baseline 무시, 기본 stochastic)")
    p.add_argument("--max-engage-time", type=float, default=120.0)
    p.add_argument("--episode-step-limit", type=int, default=7200)
    p.add_argument("--min-altitude", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-log", action="store_true", help="tacview CSV + summary 로그 저장")
    return p.parse_args()


def _save_replay_log(env) -> str:
    """올바른 RL-step 타임스탬프로 tacview CSV + summary 를 저장한다.

    프레임워크 env.make_tacviewLog() 는 time = _delta_t(=1/sim_hz) * step 으로 쓰는데, 로그
    엔트리는 RL-step 마다 1개(=step_ratio sub-step)라 시간이 step_ratio 배 압축된다. 여기선
    RL-step 당 실제 경과 시간 = step_ratio/sim_hz 로 시간 축을 바로잡아 같은 포맷으로 쓴다.
    """
    import datetime
    import json

    art_dir = env.config.get("artifacts_dir", "artifacts/logs")
    os.makedirs(art_dir, exist_ok=True)
    ts = datetime.datetime.today()
    stamp = f"{ts.year}_{ts.month}_{ts.day}_{ts.hour}_{ts.minute}_{ts.second}"
    dt_rl = float(env._delta_t) * float(env._step_ratio)   # RL-step 당 경과 시간(s)

    def _write(path, entries):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Time,Longitude,Latitude,Altitude,Roll (deg),Pitch (deg),"
                    "Yaw (deg),Health\n")
            for step, item in enumerate(entries):
                t = np.floor(dt_rl * step * 10000) / 10000
                health = item[6] if len(item) > 6 else ""
                f.write(f"{t},{item[1]},{item[0]},{item[2]},"
                        f"{item[3]},{item[4]},{item[5]},{health}\n")

    _write(os.path.join(art_dir, f"{stamp}_ownship_(F-16)[Blue].csv"), env.ownship_log)
    _write(os.path.join(art_dir, f"{stamp}_target_(F-16)[Red].csv"), env.target_log)
    with open(os.path.join(art_dir, f"{stamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"end_condition": env.info.get("end_condition", ""),
                   "outcome": env.info.get("outcome", ""),
                   "ownship_health": env.info.get("ownship_health"),
                   "target_health": env.info.get("target_health")},
                  f, indent=2, ensure_ascii=False)
    return art_dir


def main():
    args = parse_args()
    step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

    own_spec = parse_agent_spec(args.ownship)
    tgt_spec = parse_agent_spec(args.target)

    bt_rule = bt_rule_for_specs(own_spec, tgt_spec) or ""
    if bt_rule != os.environ.get(ENV_KEY, ""):
        raise RuntimeError(
            f"BT rule XML 불일치: 최종={bt_rule!r} / import 시점={os.environ.get(ENV_KEY)!r}.")

    import torch
    torch.manual_seed(args.seed)   # stochastic 샘플링 재현용

    any_nn = agents.is_nn(own_spec) or agents.is_nn(tgt_spec)
    obs_module = agents.OBS_MODULE if any_nn else ""

    overrides = {
        "target_mode": "fixed",            # 양 슬롯 모두 provider 가 조종
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "min_altitude": args.min_altitude,
        "randomize_start_side": False,     # 재현성
    }
    env = make_env(overrides=overrides, observation_module=obs_module, runner_index="local")

    own_provider = agents.make_side_provider(
        env, own_spec, step_ratio, hz=args.ownship_hz,
        deterministic=(args.ownship_action == "argmax"), wid=0, root=ROOT,
        base_port=agents.UNREAL_BASE_PORT + 1000)
    env._ownship_action_provider = own_provider

    tgt_provider = agents.make_side_provider(
        env, tgt_spec, step_ratio, hz=args.target_hz,
        deterministic=(args.target_action == "argmax"), wid=0, root=ROOT,
        base_port=agents.UNREAL_BASE_PORT)
    env._target_action_provider = tgt_provider

    def _d(spec, hz, action):
        if agents.is_nn(spec):
            return f"{spec['kind']}:{spec['name']}({hz}Hz,{action})"
        return spec["name"]

    print(f"[local] ownship={_d(own_spec, args.ownship_hz, args.ownship_action)} "
          f"target={_d(tgt_spec, args.target_hz, args.target_action)} "
          f"obs_module={obs_module or '(env default)'} "
          f"max_engage={args.max_engage_time}s step_limit={args.episode_step_limit}")

    try:
        obs, info = env.reset(seed=args.seed)
        own_provider.reset()
        tgt_provider.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total_reward += reward
            steps += 1

        sim_seconds = steps * float(env._delta_t) * float(env._step_ratio)
        print("simulation finished")
        print(f"end_condition: {info.get('end_condition', '')}")
        print(f"outcome: {info.get('outcome', '')}  steps: {steps}  "
              f"sim_time: {sim_seconds:.1f}s")
        print(f"total_reward: {total_reward:.4f}")
        print(f"ownship_health: {info.get('ownship_health', 'n/a')}")
        print(f"target_health: {info.get('target_health', 'n/a')}")

        if args.save_log:
            art_dir = _save_replay_log(env)
            print(f"tacview log saved → {art_dir}  "
                  f"(replay: python tools/web_log_viewer.py)")
    finally:
        for p in (tgt_provider, own_provider):
            try:
                p.close()
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    main()
