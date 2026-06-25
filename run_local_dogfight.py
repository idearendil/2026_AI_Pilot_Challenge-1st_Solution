from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook


def parse_args():
    parser = argparse.ArgumentParser(description="Run local dogfight simulation between two inference backends.")
    parser.add_argument("--ownship-backend", choices=["rl", "bt", "hybrid", "fixed"], required=True)
    parser.add_argument("--target-backend", choices=["rl", "bt", "hybrid", "fixed"], required=True)
    parser.add_argument("--ownship-bundle-dir")
    parser.add_argument("--target-bundle-dir")
    parser.add_argument("--ownship-bt-dll", default="AIP_DCS_ownship.dll")
    parser.add_argument("--target-bt-dll", default="AIP_BASE_target.dll")
    parser.add_argument("--bt-rule-xml", help="Optional Rule.xml source to activate while the simulation runs.")
    parser.add_argument("--ownship-policy-id", default="default_policy")
    parser.add_argument("--target-policy-id", default="default_policy")
    parser.add_argument("--observation-mode", default="tactical16", choices=["classic12", "relative14", "tactical16", "custom"])
    parser.add_argument("--observation-module", default="", help="Optional custom observation module.")
    parser.add_argument("--hybrid-mode", choices=["residual", "blend", "switch"], default="residual")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--max-engage-time", type=float, default=300.0)
    parser.add_argument("--episode-step-limit", type=int, default=18000)
    parser.add_argument("--min-altitude", type=float, default=300.0)
    parser.add_argument("--save-log", action="store_true", help="Save tacview CSV log after the episode.")
    return parser.parse_args()


def build_provider(side: str, backend: str, bundle_dir: str | None, bt_dll: str, policy_id: str, hybrid_mode: str, alpha: float, residual_scale: float):
    if backend == "fixed":
        return None
    if backend == "bt":
        return BTActionProvider(dll_name=bt_dll)
    if backend == "rl":
        if not bundle_dir:
            raise ValueError(f"--{side}-bundle-dir is required when {side}-backend=rl")
        return RLActionProvider(bundle_dir=bundle_dir, algorithm_factory=build_algorithm_from_bundle, policy_id=policy_id)
    if backend == "hybrid":
        if not bundle_dir:
            raise ValueError(f"--{side}-bundle-dir is required when {side}-backend=hybrid")
        rl_provider = RLActionProvider(bundle_dir=bundle_dir, algorithm_factory=build_algorithm_from_bundle, policy_id=policy_id)
        bt_provider = BTActionProvider(dll_name=bt_dll)
        return HybridActionProvider(
            primary_provider=rl_provider,
            secondary_provider=bt_provider,
            mode=hybrid_mode,
            alpha=alpha,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def backend_to_env_mode(backend: str) -> str:
    if backend == "fixed":
        return "fixed"
    return "rl"


def main():
    args = parse_args()
    observation_hook = load_observation_hook(args.observation_module) if args.observation_module else None

    ownship_provider = build_provider(
        side="ownship",
        backend=args.ownship_backend,
        bundle_dir=args.ownship_bundle_dir,
        bt_dll=args.ownship_bt_dll,
        policy_id=args.ownship_policy_id,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        residual_scale=args.residual_scale,
    )
    target_provider = build_provider(
        side="target",
        backend=args.target_backend,
        bundle_dir=args.target_bundle_dir,
        bt_dll=args.target_bt_dll,
        policy_id=args.target_policy_id,
        hybrid_mode=args.hybrid_mode,
        alpha=args.alpha,
        residual_scale=args.residual_scale,
    )

    with activate_rule_xml(args.bt_rule_xml, ROOT):
        env = DogFightWrapper(
            env_config={
                "observation_mode": observation_hook["mode"] if observation_hook else args.observation_mode,
                "observation_module": args.observation_module,
                "ownship_control_mode": backend_to_env_mode(args.ownship_backend),
                "target_mode": backend_to_env_mode(args.target_backend),
                "max_engage_time": args.max_engage_time,
                "episode_step_limit": args.episode_step_limit,
                "min_altitude": args.min_altitude,
            },
            observation_fn=observation_hook["build_observation"] if observation_hook else None,
            observation_size=observation_hook["size"] if observation_hook else None,
            observation_low=observation_hook["low"] if observation_hook else None,
            observation_high=observation_hook["high"] if observation_hook else None,
            ownship_action_provider=ownship_provider,
            target_action_provider=target_provider,
        )

        try:
            observation, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                total_reward += reward

            print("simulation finished")
            print(f"end_condition: {info.get('end_condition', '')}")
            print(f"terminated: {terminated} truncated: {truncated}")
            print(f"total_reward: {total_reward:.4f}")
            print(f"ownship_health: {info.get('ownship_health', 'n/a')}")
            print(f"target_health: {info.get('target_health', 'n/a')}")

            if args.save_log:
                env.make_tacviewLog()
                print("tacview log saved")
        finally:
            env.close()


if __name__ == "__main__":
    main()
