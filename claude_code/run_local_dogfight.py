"""claude_code 번들용 로컬 대결 + 리플레이(tacview) 로그 생성.

원본 `run_local_dogfight.py` 의 `rl` 백엔드는 RLlib 번들 전용(`RLActionProvider` +
`build_algorithm_from_bundle`)이라 claude_code PPO 번들(metadata.json +
policy_weights.pkl.gz)을 못 받는다. 이 스크립트는 원본과 거의 동일한 플래그를 받되
`rl` 백엔드를 claude `MLPActionProvider` 로 연결하고, 끝나면 env.make_tacviewLog() 로
ownship/target CSV + summary.json 리플레이 로그를 저장한다.

환경은 학습과 동일한 `make_env`(TierGatedDogFightEnv, 시간게이팅 damage)를 쓰고, 관측
모듈은 번들 메타에 기록된 값을 그대로 주입한다(학습/제출과 동일 관측 재구성). RL action
은 학습/제출과 동일하게 step_ratio(=6) 동안 유지한다.

예시 (학습한 번들 vs AIP_BASE_target.dll, 리플레이 저장):
  python claude_code/run_local_dogfight.py \
    --ownship-backend rl \
    --ownship-bundle-dir artifacts/models/team01/ppo_mlp_v1_final \
    --target-backend bt \
    --target-bt-dll AIP_BASE_target.dll \
    --max-engage-time 120 \
    --episode-step-limit 7200 \
    --save-log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from claude_code.action_provider import MLPActionProvider
from claude_code.evaluate import _ActionRepeatProvider
from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="claude_code 번들 로컬 대결 + 리플레이 로그")
    p.add_argument("--ownship-backend", choices=["rl", "bt"], default="rl")
    p.add_argument("--target-backend", choices=["rl", "bt", "loiter", "fixed", "autopilot"],
                   default="bt", help="bt = behavior_tree DLL(AIP_BASE_target.dll 등)")
    p.add_argument("--ownship-bundle-dir", help="ownship rl 일 때 claude 번들 경로")
    p.add_argument("--target-bundle-dir", help="target rl 일 때 claude 번들 경로")
    p.add_argument("--ownship-bt-dll", default="AIP_DCS_ownship.dll")
    p.add_argument("--target-bt-dll", default="AIP_BASE_target.dll")
    # 원본 CLI 호환용(번들 메타의 observation_module 이 우선한다).
    p.add_argument("--observation-mode", default="tactical16")
    p.add_argument("--max-engage-time", type=float, default=120.0)
    p.add_argument("--episode-step-limit", type=int, default=7200)
    p.add_argument("--min-altitude", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-log", action="store_true", help="tacview CSV + summary 로그 저장")
    return p.parse_args()


def _bt_to_behavior_tree(mode: str) -> str:
    return "behavior_tree" if mode == "bt" else mode


def _save_replay_log(env) -> str:
    """올바른 RL-step 타임스탬프로 tacview CSV + summary 를 저장한다.

    프레임워크 env.make_tacviewLog() 는 _write_log 에서 time = _delta_t(=1/sim_hz) * step
    으로 쓰는데, 로그 엔트리는 RL-step 마다 1개(=step_ratio sub-step)라 시간이 step_ratio
    배 압축된다(예: step_ratio=6 이면 120초가 20초로 찍힘). 여기선 RL-step 당 실제 경과
    시간 = step_ratio/sim_hz 로 시간 축을 바로잡아 같은 포맷으로 직접 쓴다.
    """
    import datetime
    import json
    import os

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

    # ownship 번들에서 관측 모듈을 읽어 env 에 동일 관측을 주입(학습/제출과 동일 재구성).
    obs_module = ""
    ownship_inner = None
    if args.ownship_backend == "rl":
        if not args.ownship_bundle_dir:
            raise ValueError("--ownship-backend rl 이면 --ownship-bundle-dir 이 필요합니다.")
        ownship_inner = MLPActionProvider(bundle_dir=args.ownship_bundle_dir)
        obs_module = ownship_inner.metadata.get("observation_module", "") or ""

    overrides = {
        "target_mode": _bt_to_behavior_tree(args.target_backend),
        "target_behavior_dll": args.target_bt_dll,
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "min_altitude": args.min_altitude,
        # 리플레이는 좌우 랜덤 시작 끔(재현성).
        "randomize_start_side": False,
    }
    if args.ownship_backend == "bt":
        overrides["ownship_control_mode"] = "behavior_tree"
        overrides["ownship_behavior_dll"] = args.ownship_bt_dll

    env = make_env(overrides=overrides, observation_module=obs_module, runner_index="local")

    # ownship rl 정책 주입 (학습/제출과 동일한 action_repeat=step_ratio).
    ownship_provider = None
    if ownship_inner is not None:
        ownship_provider = _ActionRepeatProvider(ownship_inner, step_ratio)
        env._ownship_action_provider = ownship_provider

    # target rl (선택) — 별도 번들로 조종. target 은 MLPActionProvider 가 아니라
    # SelfPlayProvider 로 만든다: MLPActionProvider 는 전역 reconstructor 싱글톤(_RECON)을
    # 쓰는데 ownship 도 같은 _RECON 을 쓰므로 둘이 충돌한다(상대 관점 HP 가 깨짐).
    # SelfPlayProvider 는 자체 reconstructor + 상대 관점 관측 재구성 + 자체 action_repeat 라
    # ownship 과 독립적으로 정확히 동작한다(학습/평가의 self-play 와 동일 경로).
    target_provider = None
    if args.target_backend == "rl":
        if not args.target_bundle_dir:
            raise ValueError("--target-backend rl 이면 --target-bundle-dir 이 필요합니다.")
        from claude_code.model import load_bundle
        from claude_code.normalizers import RunningMeanStd
        from claude_code.self_play import SelfPlayProvider
        tgt_model, tgt_meta = load_bundle(args.target_bundle_dir, device="cpu")
        tgt_rms = (RunningMeanStd.from_state_dict(tgt_meta["obs_normalization"])
                   if tgt_meta.get("obs_normalization") else None)
        target_provider = SelfPlayProvider(
            tgt_model, tgt_rms, env._observation_fn, env._observation_mode,
            step_ratio, "cpu", explore=False)   # 결정론적 리플레이
        env._target_action_provider = target_provider

    tgt_desc = (args.target_bt_dll if args.target_backend == "bt"
                else (args.target_bundle_dir or "") if args.target_backend == "rl"
                else "")
    print(f"[claude_code/local] ownship={args.ownship_backend} "
          f"target={args.target_backend}({tgt_desc}) "
          f"obs_module={obs_module or '(env default)'} "
          f"max_engage={args.max_engage_time}s step_limit={args.episode_step_limit}")

    try:
        obs, info = env.reset(seed=args.seed)
        if ownship_provider is not None:
            ownship_provider.reset()
        if target_provider is not None:
            target_provider.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            # action 인자는 provider 사용 시 무시되지만 step 시그니처상 필요.
            obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total_reward += reward
            steps += 1

        sim_seconds = steps * float(env._delta_t) * float(env._step_ratio)
        print("simulation finished")
        print(f"end_condition: {info.get('end_condition', '')}")
        print(f"outcome: {info.get('outcome', '')}  steps: {steps}  "
              f"sim_time: {sim_seconds:.1f}s")
        print(f"terminated: {terminated} truncated: {truncated}")
        print(f"total_reward: {total_reward:.4f}")
        print(f"ownship_health: {info.get('ownship_health', 'n/a')}")
        print(f"target_health: {info.get('target_health', 'n/a')}")

        if args.save_log:
            art_dir = _save_replay_log(env)
            print(f"tacview log saved → {art_dir}  (시간축 = RL-step×step_ratio/sim_hz)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
