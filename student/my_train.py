# -*- coding: utf-8 -*-
"""Minimal student training launcher.

This file intentionally keeps no RLlib training loop. Edit the config dicts
below; the shared trainer in train_rllib.py handles logging, dashboards,
records, and policy bundle export.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TRAINING_CONFIG = {
    "team_name": "team01",
    "output_tag": "v1",
    "algorithm": "sac",
    "iterations": 50,
    "reward_module": "student.my_reward",
    "observation_module": "",
}


ENV_CONFIG = {
    "observation_mode": "tactical16",
    "target_mode": "behavior_tree",
    "target_behavior_dll": "AIP_BASE_target.dll",
    "max_engage_time": 300.0,
    "episode_step_limit": 18000,
}


RL_CONFIG = {
    "framework": "torch",
    "num_env_runners": 1,
    "num_envs_per_env_runner": 1,
    "rollout_fragment_length": "auto",
    "batch_mode": "truncate_episodes",
    "lr": 3e-4,
    "gamma": 0.99,
    "train_batch_size": 4096,
    "minibatch_size": 256,
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run the student reward template.")
    parser.add_argument("--team-name", default=TRAINING_CONFIG["team_name"])
    parser.add_argument("--output-tag", default=TRAINING_CONFIG["output_tag"])
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default=TRAINING_CONFIG["algorithm"],
    )
    parser.add_argument("--iterations", type=int, default=TRAINING_CONFIG["iterations"])
    parser.add_argument("--reward-module", default=TRAINING_CONFIG["reward_module"])
    parser.add_argument("--observation-module", default=TRAINING_CONFIG["observation_module"])
    parser.add_argument("--experiment-yaml", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train_rllib.py"),
        "--algorithm",
        args.algorithm,
        "--iterations",
        str(args.iterations),
        "--output-name",
        args.team_name,
        "--output-tag",
        args.output_tag,
        "--reward-module",
        args.reward_module,
        "--observation-mode",
        "custom" if args.observation_module else str(ENV_CONFIG["observation_mode"]),
        *(
            ["--observation-module", args.observation_module]
            if args.observation_module
            else []
        ),
        "--target-mode",
        str(ENV_CONFIG["target_mode"]),
        "--target-behavior-dll",
        str(ENV_CONFIG["target_behavior_dll"]),
        "--max-engage-time",
        str(ENV_CONFIG["max_engage_time"]),
        "--episode-step-limit",
        str(ENV_CONFIG["episode_step_limit"]),
        "--framework",
        str(RL_CONFIG["framework"]),
        "--num-env-runners",
        str(RL_CONFIG["num_env_runners"]),
        "--num-envs-per-env-runner",
        str(RL_CONFIG["num_envs_per_env_runner"]),
        "--rollout-fragment-length",
        str(RL_CONFIG["rollout_fragment_length"]),
        "--batch-mode",
        str(RL_CONFIG["batch_mode"]),
        "--lr",
        str(RL_CONFIG["lr"]),
        "--gamma",
        str(RL_CONFIG["gamma"]),
        "--train-batch-size",
        str(RL_CONFIG["train_batch_size"]),
        "--minibatch-size",
        str(RL_CONFIG["minibatch_size"]),
        "--notes",
        "student minimal reward template",
        "--experiment-yaml",
        args.experiment_yaml,
    ]


def main() -> int:
    args, passthrough = parse_args()
    command = build_command(args) + passthrough
    print("[student/my_train] " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
