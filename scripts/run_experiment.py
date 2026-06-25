from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_TARGETS = {
    "train_rllib": ROOT / "train_rllib.py",
    "train_curriculum": ROOT / "train_curriculum.py",
    "student/my_train": ROOT / "student" / "my_train.py",
}


class ExperimentError(ValueError):
    """Raised when an experiment YAML is invalid."""


def load_experiment(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentError(f"experiment YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ExperimentError("experiment YAML root must be a mapping")
    return data


def build_argv(exp: dict[str, Any], exp_path: Path) -> tuple[Path, list[str]]:
    script_name = str(exp.get("script", "train_rllib")).strip()
    if script_name not in SCRIPT_TARGETS:
        allowed = ", ".join(sorted(SCRIPT_TARGETS))
        raise ExperimentError(f"unsupported script: {script_name!r}. Allowed: {allowed}")

    output = _section(exp, "output")
    env = _section(exp, "env")
    algo = _section(exp, "algo")
    runtime = _section(exp, "runtime")
    dashboard = _section(exp, "dashboard")
    curriculum = _section(exp, "curriculum")

    output_name = _required(output, "name", "output.name")
    output_tag = _required(output, "tag", "output.tag")

    argv: list[str] = []
    argv += ["--algorithm", str(algo.get("name", "ppo"))]
    if script_name != "train_curriculum":
        argv += ["--iterations", str(runtime.get("iterations", 5))]

    if script_name == "student/my_train":
        argv += ["--team-name", str(output_name)]
    else:
        argv += ["--output-name", str(output_name)]
    argv += ["--output-tag", str(output_tag)]

    _add_optional(argv, "--framework", algo, "framework")
    _add_optional(argv, "--lr", algo, "lr")
    _add_optional(argv, "--gamma", algo, "gamma")
    _add_optional(argv, "--train-batch-size", algo, "train_batch_size")
    _add_optional(argv, "--minibatch-size", algo, "minibatch_size")
    _add_optional(argv, "--gae-lambda", algo, "gae_lambda")
    _add_optional(argv, "--clip-param", algo, "clip_param")
    _add_optional(argv, "--tau", algo, "tau")
    _add_optional(argv, "--target-entropy", algo, "target_entropy")
    _add_replay_buffer_options(argv, algo)
    _add_mlp_model_options(argv, algo)
    _add_network_options(argv, algo)
    _add_lstm_sac_options(argv, algo)

    _add_optional(argv, "--observation-mode", env, "observation_mode")
    _add_optional(argv, "--observation-module", env, "observation_module")
    _add_optional(argv, "--target-behavior-dll", env, "target_behavior_dll")
    _add_optional(argv, "--reward-module", env, "reward_module")
    if script_name != "train_curriculum":
        _add_optional(argv, "--target-mode", env, "target_mode")
        _add_optional(argv, "--max-engage-time", env, "max_engage_time")
        _add_optional(argv, "--episode-step-limit", env, "episode_step_limit")

    _add_optional(argv, "--num-env-runners", runtime, "num_env_runners")
    if script_name != "train_curriculum":
        _add_optional(
            argv,
            "--num-envs-per-env-runner",
            runtime,
            "num_envs_per_env_runner",
        )
        _add_optional(argv, "--rollout-fragment-length", runtime, "rollout_fragment_length")
        _add_optional(argv, "--batch-mode", runtime, "batch_mode")
        _add_optional(argv, "--checkpoint-frequency", runtime, "checkpoint_frequency")
        _add_optional(
            argv,
            "--lightweight-bundle-frequency",
            runtime,
            "lightweight_bundle_frequency",
        )
        _add_optional(
            argv,
            "--native-checkpoint-frequency",
            runtime,
            "native_checkpoint_frequency",
        )
        _add_optional(argv, "--print-every", runtime, "print_every")
    else:
        _add_optional(argv, "--start-stage", runtime, "start_stage")
        _add_optional(argv, "--stages-module", curriculum, "stages_module")

    if runtime.get("use_tune", False):
        argv.append("--use-tune")
    if runtime.get("save_lightweight_bundle", True) is False:
        argv.append("--no-save-lightweight-bundle")
    save_native = runtime.get("save_native_checkpoint", False)
    if save_native and script_name != "train_curriculum":
        argv.append("--save-native-checkpoint")
    if runtime.get("resume", False):
        if script_name != "train_curriculum":
            raise ExperimentError(
                "runtime.resume is only supported by train_curriculum; "
                "use runtime.restore_checkpoint for train_rllib."
            )
        argv.append("--resume")
    _add_optional(argv, "--restore-checkpoint", runtime, "restore_checkpoint")
    init_bundle = runtime.get("init_bundle")
    if init_bundle is None:
        init_bundle = runtime.get("restart_from_bundle")
    if init_bundle is not None:
        argv += ["--init-bundle", str(init_bundle)]

    if script_name == "train_curriculum":
        pass
    elif dashboard.get("enabled", True):
        _add_optional(argv, "--dashboard-logdir", dashboard, "logdir")
    else:
        argv.append("--disable-dashboard-log")
    _add_policy_probe_options(argv, exp)
    _add_engagement_log_options(argv, exp)

    if exp.get("notes"):
        argv += ["--notes", str(exp["notes"])]
    argv += ["--experiment-yaml", str(exp_path.resolve())]
    return SCRIPT_TARGETS[script_name], argv


def _section(exp: dict[str, Any], key: str) -> dict[str, Any]:
    value = exp.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentError(f"{key} must be a mapping")
    return value


def _required(section: dict[str, Any], key: str, label: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise ExperimentError(f"{label} is required")
    return value


def _add_optional(
    argv: list[str],
    flag: str,
    section: dict[str, Any],
    key: str,
) -> None:
    value = section.get(key)
    if value is not None:
        argv += [flag, str(value)]


def _add_replay_buffer_options(argv: list[str], algo: dict[str, Any]) -> None:
    """Map YAML replay-buffer aliases to trainer CLI flags."""
    capacity = algo.get("replay_buffer_capacity", algo.get("replay_buffer_size"))
    replay_cfg = algo.get("replay_buffer_config")
    if capacity is None and isinstance(replay_cfg, dict):
        capacity = replay_cfg.get("capacity")
    if capacity is not None:
        argv += ["--replay-buffer-capacity", str(capacity)]


def _add_mlp_model_options(argv: list[str], algo: dict[str, Any]) -> None:
    """Map algo.mlp YAML options to RLlib DefaultModelConfig CLI flags."""
    mlp = algo.get("mlp", {})
    if not isinstance(mlp, dict) or not mlp.get("enabled", False):
        return
    _add_model_optional(argv, "--model-fcnet-hiddens", mlp, "fcnet_hiddens")
    _add_model_optional(argv, "--model-fcnet-activation", mlp, "fcnet_activation")
    _add_model_optional(
        argv,
        "--model-head-fcnet-hiddens",
        mlp,
        "head_fcnet_hiddens",
    )
    _add_model_optional(
        argv,
        "--model-head-fcnet-activation",
        mlp,
        "head_fcnet_activation",
    )
    _add_model_optional(argv, "--model-vf-share-layers", mlp, "vf_share_layers")


def _add_network_options(argv: list[str], algo: dict[str, Any]) -> None:
    """Pass the optional SAC/PPO layer-sequence network spec to training scripts."""
    network = algo.get("network")
    if network is None:
        return
    if not isinstance(network, dict):
        raise ExperimentError("algo.network must be a mapping")
    argv += [
        "--network-spec-json",
        json.dumps(network, ensure_ascii=False, separators=(",", ":")),
    ]


def _add_model_optional(
    argv: list[str],
    flag: str,
    section: dict[str, Any],
    key: str,
) -> None:
    value = section.get(key)
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    elif isinstance(value, bool):
        value = str(value).lower()
    argv += [flag, str(value)]


def _add_lstm_sac_options(argv: list[str], algo: dict[str, Any]) -> None:
    """Map algo.lstm YAML options to recurrent CLI flags."""
    lstm = algo.get("lstm", {})
    if not isinstance(lstm, dict) or not lstm.get("enabled", False):
        return
    algorithm = str(algo.get("name", "ppo")).strip().lower()
    if algorithm == "sac":
        argv.append("--use-lstm-sac")
    elif algorithm == "ppo":
        argv.append("--use-lstm")
    else:
        raise ExperimentError(f"algo.lstm is not supported for algorithm {algorithm!r}")
    _add_optional(argv, "--lstm-scope", lstm, "scope")
    _add_optional(argv, "--lstm-cell-size", lstm, "cell_size")
    _add_optional(argv, "--max-seq-len", lstm, "max_seq_len")
    if lstm.get("prioritized_replay", False):
        argv.append("--use-lstm-prioritized-replay")
    if lstm.get("debug_io", False):
        argv.append("--debug-io")


def _add_policy_probe_options(argv: list[str], exp: dict[str, Any]) -> None:
    probe = exp.get("policy_probe", {})
    if probe is None:
        return
    if not isinstance(probe, dict):
        raise ExperimentError("policy_probe must be a mapping")
    if not probe.get("enabled", False):
        return
    argv += ["--policy-probe-interval", str(probe.get("interval", 1))]
    _add_optional(argv, "--policy-probe-steps", probe, "steps")
    if probe.get("print", True) is False:
        argv.append("--no-policy-probe-print")


def _add_engagement_log_options(argv: list[str], exp: dict[str, Any]) -> None:
    replay = exp.get("engagement_log", {})
    if replay is None:
        return
    if not isinstance(replay, dict):
        raise ExperimentError("engagement_log must be a mapping")
    if not replay.get("enabled", False):
        return
    argv += ["--engagement-log-interval", str(replay.get("interval", 1))]
    _add_optional(argv, "--engagement-log-steps", replay, "steps")
    _add_optional(argv, "--engagement-log-episodes", replay, "episodes")
    if replay.get("print", True) is False:
        argv.append("--no-engagement-log-print")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DogFight YAML experiment.")
    parser.add_argument("experiment_yaml", help="Path to experiment YAML file.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_path = Path(args.experiment_yaml)
    if not exp_path.is_absolute():
        exp_path = (Path.cwd() / exp_path).resolve()

    try:
        exp = load_experiment(exp_path)
        script_path, script_args = build_argv(exp, exp_path)
    except ExperimentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    command = [sys.executable, str(script_path), *script_args]
    print(f"[experiment] {exp.get('name', exp_path.stem)}")
    print(f"[script]     {script_path.relative_to(ROOT)}")
    print(f"[argv]       {' '.join(script_args)}")
    print(f"[run]        {' '.join(command)}")

    if args.dry_run:
        print("[dry-run] no training started.")
        return 0

    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
