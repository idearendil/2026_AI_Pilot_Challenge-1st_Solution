"""Curriculum training for dogfight RL — Sequential Phase with strong fault recovery.

Recovery guarantees
───────────────────
1. Atomic state file  — every iteration updates curriculum_state.json via
   write-to-temp + atomic rename; never leaves a partial write.
2. Native checkpoint  — saved every `checkpoint_interval` iterations
   (full algorithm state: weights + optimizer + replay buffer).
3. Lightweight bundle — saved at stage end AND on emergency exit.
4. Emergency save     — on ANY unhandled exception: bundle + native checkpoint
   + traceback written to stage_N/emergency/{timestamp}/.
5. Multi-path restore — on stage start, tries in order:
     (a) within-stage native checkpoint  (most recent progress)
     (b) previous-stage final bundle     (inter-stage weight transfer)
     (c) fresh start                     (no prior weights)
   Each path has its own try/except; next path tried on failure.
6. Resume mode        — `--resume` reads curriculum_state.json and
   continues from the exact iteration that was interrupted.

Usage
─────
  # Fresh start
  python train_curriculum.py --algorithm sac --output-name f16_v1

  # Resume after crash
  python train_curriculum.py --algorithm sac --output-name f16_v1 --resume

  # Resume from a specific native RLlib checkpoint
  python train_curriculum.py --algorithm sac --restore-checkpoint artifacts/.../checkpoint

  # Restart from a lightweight policy bundle (weights only)
  python train_curriculum.py --algorithm sac --init-bundle artifacts/.../final_bundle

  # Force a specific starting stage (skips prior stages, no weight transfer)
  python train_curriculum.py --algorithm sac --output-name f16_v1 --start-stage 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
_pythonpath_entries = [str(ROOT), str(SRC)]
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_entries)

from ray.tune.registry import register_env

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.checkpoint_io import (
    apply_lightweight_policy_bundle,
    save_lightweight_policy_bundle,
)
from dogfight.ai.engagement_replay_logger import EngagementReplayLogger
from dogfight.ai.policy_probe_logger import PolicyProbeLogger
from dogfight.ai.curriculum import (
    CurriculumStage,
    build_stage_env_config,
    check_advancement,
    get_stages,
)
from dogfight.ai.rllib_utils import (
    build_algorithm_config,
    normalize_algorithm_name,
)
from dogfight.ai.student_hooks import (
    load_curriculum_stages,
    load_observation_hook,
    load_reward_hook,
)
from dogfight.ai.training_record import save_training_record


def _ensure_ray_runtime_env() -> None:
    """Restart Ray with local project paths available to worker actors."""
    import ray

    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
    )


def env_creator(env_config):
    cfg = dict(env_config)
    cfg["_runner_index"] = getattr(
        env_config,
        "worker_index",
        cfg.get("_runner_index", "local"),
    )
    cfg["_env_index"] = getattr(
        env_config,
        "vector_index",
        cfg.get("_env_index", 0),
    )
    reward_fn = None
    observation_hook = None
    reward_module = str(cfg.get("reward_module", "")).strip()
    if reward_module:
        reward_fn, reward_config = load_reward_hook(reward_module)
        cfg.setdefault("reward", reward_config)
    observation_module = str(cfg.get("observation_module", "")).strip()
    if observation_module:
        observation_hook = load_observation_hook(observation_module)
        cfg["observation_mode"] = observation_hook["mode"]
        cfg["observation_module"] = observation_module
        cfg["observation_summary"] = observation_hook["description"]
    env = DogFightWrapper(
        cfg,
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
    )
    if reward_module and "reward" in cfg:
        env.config["reward"] = dict(cfg["reward"])
    return env


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Curriculum training for dogfight RL.")
    p.add_argument("--algorithm",         choices=["ppo", "sac"], default="sac")
    p.add_argument("--framework",         default="torch", choices=["torch"])
    p.add_argument("--num-env-runners",   type=int, default=1)
    p.add_argument("--observation-mode",  default="tactical16",
                   choices=["classic12", "relative14", "tactical16", "custom"])
    p.add_argument("--observation-module", default="",
                   help="Optional module with custom observation size and build_observation(...).")
    p.add_argument("--target-behavior-dll", default="AIP_BASE_target.dll")
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--train-batch-size",  type=int,   default=4096)
    p.add_argument("--minibatch-size",    type=int,   default=256)
    p.add_argument("--gae-lambda",        type=float, default=0.95)
    p.add_argument("--clip-param",        type=float, default=0.2)
    p.add_argument("--tau",               type=float, default=0.005)
    p.add_argument("--target-entropy",    default="auto")
    p.add_argument("--replay-buffer-capacity", type=int, default=None,
                   help="SAC replay buffer capacity. Ignored by PPO.")
    p.add_argument("--model-fcnet-hiddens", default=None,
                   help="Comma-separated RLlib model hidden sizes, e.g. 512,256,128.")
    p.add_argument("--model-fcnet-activation", default=None,
                   help="RLlib model encoder activation, e.g. relu or tanh.")
    p.add_argument("--model-head-fcnet-hiddens", default=None,
                   help="Comma-separated RLlib model head hidden sizes, or empty for none.")
    p.add_argument("--model-head-fcnet-activation", default=None,
                   help="RLlib model head activation, e.g. relu or tanh.")
    p.add_argument("--model-vf-share-layers", default=None,
                   help="Whether PPO value function shares layers: true or false.")
    p.add_argument("--network-spec-json", default="",
                   help=("JSON object for DogFight sequence_v1 network layout. "
                         "Usually supplied by scripts/run_experiment.py from algo.network."))
    p.add_argument("--use-lstm", action="store_true",
                   help="Enable RLlib DefaultModelConfig LSTM for non-SAC algorithms such as PPO.")
    p.add_argument("--use-lstm-sac", action="store_true",
                   help="Enable the patched Ray 2.54 SAC actor-LSTM path.")
    p.add_argument("--lstm-scope", choices=["actor_only", "actor_critic"],
                   default="actor_only",
                   help="SAC LSTM scope: actor_only or actor_critic recurrent Q.")
    p.add_argument("--lstm-cell-size", type=int, default=64,
                   help="LSTM hidden state size for --use-lstm-sac.")
    p.add_argument("--max-seq-len", type=int, default=8,
                   help="Replay/train sequence length for --use-lstm-sac.")
    p.add_argument("--debug-io", dest="debug_io", action="store_true",
                   help="Print recurrent SAC/RLlib debug I/O shape checks.")
    p.add_argument("--debug-lstm-io", dest="debug_io", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--use-lstm-prioritized-replay", action="store_true",
                   help=("Use patched PrioritizedEpisodeReplayBuffer sequence "
                         "sampling for --use-lstm-sac."))
    p.add_argument("--policy-probe-interval", type=int, default=0,
                   help=("Log fixed policy probe actions every N total iterations. "
                         "0 disables policy_probe.csv/jsonl."))
    p.add_argument("--policy-probe-steps", type=int, default=4,
                   help="Number of recurrent inference steps per policy probe.")
    p.add_argument("--no-policy-probe-print", action="store_true",
                   help="Write policy probe files without console summaries.")
    p.add_argument("--engagement-log-interval", type=int, default=0,
                   help=("Run a short policy-vs-target replay every N total "
                         "iterations and save Tacview CSV logs. 0 disables it."))
    p.add_argument("--engagement-log-steps", type=int, default=600,
                   help="Maximum environment steps per engagement replay episode.")
    p.add_argument("--engagement-log-episodes", type=int, default=1,
                   help="Number of replay episodes to save at each interval.")
    p.add_argument("--no-engagement-log-print", action="store_true",
                   help="Write engagement replay files without console summaries.")
    p.add_argument("--output-name",       default="f16_curriculum")
    p.add_argument("--output-tag",        default="v1")
    p.add_argument("--resume",            action="store_true",
                   help="Resume from curriculum_state.json of a previous run.")
    p.add_argument("--start-stage",       type=int, default=None,
                   help="Force-start at this stage index (ignores state file).")
    p.add_argument("--notes",             default="")
    p.add_argument("--reward-module",     default="",
                   help="Optional module with MY_REWARD_CONFIG and compute_reward(...).")
    p.add_argument("--stages-module",     default="",
                   help="Optional module with get_stages() returning CurriculumStage list.")
    p.add_argument("--restore-checkpoint", default="",
                   help="Restore a full RLlib native checkpoint for the first active stage.")
    p.add_argument("--init-bundle", "--restart-from-bundle", dest="init_bundle",
                   default="",
                   help="Load lightweight policy bundle weights for the first active stage.")
    p.add_argument("--experiment-yaml",   default="",
                   help="Optional source experiment YAML path for records.")
    args = p.parse_args()
    if args.restore_checkpoint and args.init_bundle:
        p.error("--restore-checkpoint and --init-bundle are mutually exclusive.")
    return args


def _build_model_config_args(args: argparse.Namespace) -> dict[str, Any]:
    model_config = {
        "fcnet_hiddens": args.model_fcnet_hiddens,
        "fcnet_activation": args.model_fcnet_activation,
        "head_fcnet_hiddens": args.model_head_fcnet_hiddens,
        "head_fcnet_activation": args.model_head_fcnet_activation,
        "vf_share_layers": args.model_vf_share_layers,
    }
    if args.network_spec_json:
        model_config["network_spec"] = json.loads(args.network_spec_json)
    model_config["enabled"] = any(value is not None for value in model_config.values())
    return model_config


def _sync_lstm_args_from_init_bundle(args) -> None:
    """Align SAC LSTM architecture args with a lightweight bundle before build."""

    if not args.init_bundle:
        return

    bundle_path = Path(args.init_bundle)
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle_meta = payload.get("metadata", {})
    saved_model_config = bundle_meta.get("model_config") or {}
    bundle_uses_lstm = bool(
        bundle_meta.get("use_lstm_sac")
        or saved_model_config.get("use_lstm")
    )
    if not bundle_uses_lstm:
        return

    algorithm_name = normalize_algorithm_name(args.algorithm)
    lstm_scope = (
        saved_model_config.get("dogfight_lstm_scope")
        or bundle_meta.get("lstm_scope")
        or "actor_only"
    )
    lstm_cell_size = int(
        saved_model_config.get("lstm_cell_size")
        or bundle_meta.get("lstm_cell_size")
        or args.lstm_cell_size
    )
    max_seq_len = int(
        saved_model_config.get("max_seq_len")
        or bundle_meta.get("max_seq_len")
        or args.max_seq_len
    )
    network_spec = (
        saved_model_config.get("dogfight_network_spec")
        or bundle_meta.get("network_spec")
    )

    if algorithm_name != "sac":
        changed = (
            not args.use_lstm
            or args.lstm_cell_size != lstm_cell_size
            or args.max_seq_len != max_seq_len
            or (
                network_spec is not None
                and args.network_spec_json
                != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
            )
        )
        args.use_lstm = True
        args.lstm_cell_size = lstm_cell_size
        args.max_seq_len = max_seq_len
        if network_spec is not None:
            args.network_spec_json = json.dumps(
                network_spec, ensure_ascii=False, separators=(",", ":")
            )
        if changed:
            print(
                "[DogFightEnv][LSTM_RESUME] "
                f"init_bundle={bundle_path} use_lstm=True "
                f"lstm_cell_size={lstm_cell_size} max_seq_len={max_seq_len} "
                f"network_type={(network_spec or {}).get('type')}"
            )
        return

    changed = (
        not args.use_lstm_sac
        or args.lstm_scope != lstm_scope
        or args.lstm_cell_size != lstm_cell_size
        or args.max_seq_len != max_seq_len
        or (
            network_spec is not None
            and args.network_spec_json
            != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
        )
    )
    args.use_lstm_sac = True
    args.lstm_scope = lstm_scope
    args.lstm_cell_size = lstm_cell_size
    args.max_seq_len = max_seq_len
    if network_spec is not None:
        args.network_spec_json = json.dumps(
            network_spec, ensure_ascii=False, separators=(",", ":")
        )
    if changed:
        print(
            "[DogFightEnv][LSTM_RESUME] "
            f"init_bundle={bundle_path} use_lstm_sac=True "
            f"lstm_scope={lstm_scope} lstm_cell_size={lstm_cell_size} "
            f"max_seq_len={max_seq_len} "
            f"network_type={(network_spec or {}).get('type')}"
        )


# ── Metric helpers ────────────────────────────────────────────────────────────

def _extract_learner_stats(result: dict) -> dict:
    keys = (
        "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
        "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
        "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
        "learner_steps_per_sec", "iteration_time_s",
    )
    stats = {k: "n/a" for k in keys}
    learners = result.get("learners", {})
    if learners:
        ps = next(iter(learners.values()), {})
        stats.update({
            "policy_loss":   ps.get("policy_loss",   "n/a"),
            "vf_loss":       ps.get("vf_loss",        "n/a"),
            "entropy":       ps.get("entropy",         "n/a"),
            "kl":            ps.get("mean_kl_loss",   ps.get("kl", "n/a")),
            "clip_frac":     ps.get("clip_frac",       "n/a"),
            "explained_var": ps.get("vf_explained_var","n/a"),
        })
        _fill_optional_learner_stats(stats, ps, result)
        return stats
    ps = next(iter(result.get("info", {}).get("learner", {}).values()), {})
    if ps:
        stats.update({
            "policy_loss":   ps.get("policy_loss",    "n/a"),
            "vf_loss":       ps.get("vf_loss",         "n/a"),
            "entropy":       ps.get("entropy",          "n/a"),
            "kl":            ps.get("kl",               "n/a"),
            "clip_frac":     ps.get("clip_frac",        "n/a"),
            "explained_var": ps.get("vf_explained_var", "n/a"),
        })
        _fill_optional_learner_stats(stats, ps, result)
    return stats


def _fill_optional_learner_stats(stats: dict, policy_stats: dict, result: dict) -> None:
    def first_present(*names: str):
        for source in (policy_stats, result):
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]
        return "n/a"

    stats["actor_loss"] = first_present("actor_loss", "policy_loss")
    stats["critic_loss"] = first_present("critic_loss", "qf_loss", "q_loss")
    stats["alpha_loss"] = first_present("alpha_loss")
    stats["alpha"] = first_present("alpha", "alpha_value")
    if stats["alpha"] == "n/a":
        log_alpha = first_present("log_alpha_value", "curr_log_alpha")
        if isinstance(log_alpha, (int, float)):
            stats["alpha"] = math.exp(float(log_alpha))
    stats["target_entropy"] = first_present("target_entropy")
    stats["replay_buffer_size"] = first_present(
        "replay_buffer_size", "num_steps_trained_this_iter"
    )
    stats["env_steps_per_sec"] = first_present(
        "env_steps_per_sec", "num_env_steps_sampled_throughput_per_sec"
    )
    stats["learner_steps_per_sec"] = first_present(
        "learner_steps_per_sec", "num_env_steps_trained_throughput_per_sec"
    )
    stats["iteration_time_s"] = first_present("time_this_iter_s")


def _fill_algorithm_runtime_stats(stats: dict, algorithm: Any) -> None:
    """Fill direct-loop stats that RLlib does not always place in result."""
    replay_buffer = getattr(algorithm, "local_replay_buffer", None)
    if replay_buffer is None:
        return

    if stats.get("replay_buffer_memory_mb") == "n/a":
        stats["replay_buffer_memory_mb"] = _estimate_object_memory_mb(replay_buffer)

    if stats.get("replay_buffer_size") != "n/a":
        return

    for getter_name in ("get_num_timesteps", "get_num_episodes"):
        getter = getattr(replay_buffer, getter_name, None)
        if getter is None:
            continue
        try:
            stats["replay_buffer_size"] = getter()
            return
        except Exception:
            pass

    try:
        stats["replay_buffer_size"] = len(replay_buffer)
    except Exception:
        pass


def _estimate_object_memory_mb(obj: Any) -> Any:
    """Estimate Python object memory recursively, returned in MiB."""
    if obj is None:
        return "n/a"

    seen: set[int] = set()

    def sizeof(value: Any, depth: int = 0) -> int:
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(value, 0)
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            size += nbytes
            return size
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            try:
                size += int(value.numel()) * int(value.element_size())
                return size
            except Exception:
                return size
        if depth >= 8:
            return size

        if isinstance(value, dict):
            for key, item in value.items():
                size += sizeof(key, depth + 1)
                size += sizeof(item, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += sizeof(item, depth + 1)
        elif hasattr(value, "__dict__"):
            size += sizeof(vars(value), depth + 1)
        return size

    try:
        return sizeof(obj) / (1024.0 * 1024.0)
    except Exception:
        return "n/a"


def _extract_custom_metrics(result: dict) -> dict:
    cm = result.get("env_runners", {}).get("custom_metrics", {})
    return {
        "win_rate":          cm.get("win_mean",                    "n/a"),
        "loss_rate":         cm.get("loss_mean",                   "n/a"),
        "timeout_rate":      cm.get("timeout_mean",                "n/a"),
        "crash_rate":        cm.get("crash_mean",                  "n/a"),
        "ep_wez_steps":      cm.get("ep_wez_steps_mean",           "n/a"),
        "ep_mean_distance":  cm.get("ep_mean_distance_mean",       "n/a"),
        "ep_min_distance":   cm.get("ep_min_distance_mean",        "n/a"),
        "ep_reward_survival":cm.get("ep_reward_survival_mean",     "n/a"),
        "ep_reward_pursuit": cm.get("ep_reward_pursuit_mean",      "n/a"),
        "ep_reward_damage":  cm.get("ep_reward_damage_mean",       "n/a"),
        "ep_altitude_penalty_steps": cm.get(
            "ep_altitude_penalty_steps_mean", "n/a"
        ),
        "action_sat_rate":   cm.get("action_saturation_rate_mean", "n/a"),
        "action_roll_mean":  cm.get("action_roll_mean_mean",       "n/a"),
        "action_pitch_mean": cm.get("action_pitch_mean_mean",      "n/a"),
        "action_rudder_mean":cm.get("action_rudder_mean_mean",     "n/a"),
        "action_throttle_mean": cm.get("action_throttle_mean_mean", "n/a"),
        "action_roll_std":   cm.get("action_roll_std_mean",        "n/a"),
        "action_pitch_std":  cm.get("action_pitch_std_mean",       "n/a"),
        "action_rudder_std": cm.get("action_rudder_std_mean",      "n/a"),
        "action_throttle_std": cm.get("action_throttle_std_mean",  "n/a"),
        "initial_alpha_deg": cm.get("initial_alpha_deg_mean",      "n/a"),
        "initial_ata_deg":   cm.get("initial_ata_deg_mean",        "n/a"),
        "initial_aa_deg":    cm.get("initial_aa_deg_mean",         "n/a"),
        "initial_distance_m":cm.get("initial_distance_m_mean",     "n/a"),
        "final_ata_deg":     cm.get("final_ata_deg_mean",          "n/a"),
        "final_aa_deg":      cm.get("final_aa_deg_mean",           "n/a"),
        "headon_guard_fail": cm.get("headon_guard_fail_mean",      "n/a"),
    }


def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)


def _console_header(algorithm_name: str) -> str:
    base = (
        f"{'St':>2} {'Iter':>5} | {'Reward':>10} | "
        f"{'Win%':>6} | {'Crash%':>7} | {'WEZ':>6} | "
    )
    if algorithm_name == "sac":
        return base + f"{'Actor':>9} | {'Critic':>9} | {'Alpha':>8} |"
    return base + f"{'Entropy':>8} | {'VF_loss':>8} |"


_CSV_FIELDS = [
    "stage", "iter_in_stage", "total_iter",
    "reward_mean", "ep_len_mean",
    "win_rate", "loss_rate", "timeout_rate", "crash_rate",
    "ep_wez_steps", "ep_mean_distance", "ep_min_distance",
    "ep_reward_survival", "ep_reward_pursuit", "ep_reward_damage",
    "ep_altitude_penalty_steps",
    "action_sat_rate",
    "action_roll_mean", "action_pitch_mean", "action_rudder_mean",
    "action_throttle_mean", "action_roll_std", "action_pitch_std",
    "action_rudder_std", "action_throttle_std",
    "initial_alpha_deg", "initial_ata_deg", "initial_aa_deg",
    "initial_distance_m", "final_ata_deg", "final_aa_deg",
    "headon_guard_fail",
    "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
    "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
    "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
    "learner_steps_per_sec", "iteration_time_s",
]

# ── CurriculumTrainer ─────────────────────────────────────────────────────────

class CurriculumTrainer:

    STAGES: list[CurriculumStage] = get_stages()

    def __init__(self, args):
        self.args = args
        self.algorithm_name = normalize_algorithm_name(args.algorithm)
        self.stages = (
            load_curriculum_stages(args.stages_module)
            if args.stages_module
            else self.STAGES
        )
        self.curriculum_dir = (
            ROOT / "artifacts" / "curriculum" / args.output_name / args.output_tag
        )
        self.curriculum_dir.mkdir(parents=True, exist_ok=True)
        self.state_path   = self.curriculum_dir / "curriculum_state.json"
        self.csv_path     = self.curriculum_dir / "training_log.csv"
        self._total_iter  = 0
        self._explicit_checkpoint_restored = False
        self._explicit_bundle_loaded = False

        env_preview_config = {
            "observation_mode":    args.observation_mode,
            "target_mode":         "fixed",
            "target_behavior_dll": args.target_behavior_dll,
            "ownship_control_mode":"rl",
        }
        if args.reward_module:
            env_preview_config["reward_module"] = args.reward_module
        if args.observation_module:
            env_preview_config["observation_module"] = args.observation_module
        env_preview = env_creator(env_preview_config)
        self.base_env_config = {
            **env_preview_config,
            "observation_mode": env_preview.config["observation_mode"],
            "reward": dict(env_preview.config["reward"]),
            "wez":    dict(env_preview.config["wez"]),
        }
        if args.observation_module:
            self.base_env_config["observation_module"] = args.observation_module
            self.base_env_config["observation_summary"] = dict(
                env_preview.config["observation_summary"]
            )
        obs_shape = getattr(env_preview.observation_space, "shape", (0,))
        action_shape = getattr(env_preview.action_space, "shape", (4,))
        self._probe_obs_dim = int(obs_shape[0]) if obs_shape else 0
        self._probe_action_dim = int(action_shape[0]) if action_shape else 4
        env_preview.close()
        self.policy_probe_logger = PolicyProbeLogger(
            self.curriculum_dir,
            obs_dim=self._probe_obs_dim,
            action_dim=self._probe_action_dim,
            interval=args.policy_probe_interval,
            sequence_steps=args.policy_probe_steps,
            print_to_console=not args.no_policy_probe_print,
            append=args.resume,
        )
        self.engagement_replay_logger = EngagementReplayLogger(
            self.curriculum_dir,
            env_factory=env_creator,
            env_config=self.base_env_config,
            interval=args.engagement_log_interval,
            max_steps=args.engagement_log_steps,
            episodes=args.engagement_log_episodes,
            print_to_console=not args.no_engagement_log_print,
            append=args.resume,
        )

    def run(self):
        _ensure_ray_runtime_env()
        state = self._init_or_load_state()

        if state["status"] == "completed":
            print("Curriculum already completed. Use a different --output-tag to restart.")
            return

        csv_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
            if not csv_exists:
                writer.writeheader()
            self._csv_writer = writer
            self._csv_file   = csv_file
            self.policy_probe_logger.__enter__()
            self.engagement_replay_logger.__enter__()

            try:
                for stage in self.stages:
                    stage_state = state["stages"][str(stage.index)]
                    if stage_state["status"] == "completed":
                        self._total_iter += stage_state.get("iterations_trained", 0)
                        print(f"[Stage {stage.index}] Already completed — skipping.")
                        continue
                    if self.args.start_stage is not None and stage.index < self.args.start_stage:
                        self._total_iter += stage_state.get("iterations_trained", 0)
                        print(f"[Stage {stage.index}] Skipped by --start-stage.")
                        continue

                    self._run_stage(stage, state)

                    if state.get("status") == "failed":
                        print(f"\n[Stage {stage.index}] Failed. Run with --resume to continue.")
                        break
                else:
                    state["status"] = "completed"
                    self._save_state(state)
                    print("=== Curriculum training completed ===")
            finally:
                self.policy_probe_logger.__exit__(None, None, None)
                self.engagement_replay_logger.__exit__(None, None, None)
                if self.policy_probe_logger.enabled:
                    print(f"policy probe CSV saved to {self.policy_probe_logger.csv_path}")
                    print(f"policy probe JSONL saved to {self.policy_probe_logger.jsonl_path}")
                if self.engagement_replay_logger.enabled:
                    print(
                        "engagement replay index saved to "
                        f"{self.engagement_replay_logger.csv_path}"
                    )
                    print(
                        "engagement replay JSONL saved to "
                        f"{self.engagement_replay_logger.jsonl_path}"
                    )

    def _run_stage(self, stage: CurriculumStage, state: dict):
        stage_state = state["stages"][str(stage.index)]
        start_iter  = stage_state.get("iterations_trained", 0)
        metric_window: list[dict] = list(stage_state.get("metric_history", []))

        print(f"\n{'='*60}")
        print(f"  Stage {stage.index}: {stage.name}")
        print(f"  {stage.description}")
        print(f"  target_mode={stage.target_mode}  "
              f"max_iter={stage.max_iterations}  resume_from={start_iter}")
        print(f"{'='*60}")

        stage_env_config = build_stage_env_config(self.base_env_config, stage)
        env_name = f"dogfight-curriculum-stage{stage.index}"
        register_env(env_name, env_creator)

        algorithm = self._build_algorithm(stage, stage_env_config, env_name)
        state["current_stage"] = stage.index
        stage_state["status"]  = "in_progress"
        self._save_state(state)

        self._restore_weights(algorithm, stage, stage_state, state)

        interrupted_iter = start_iter
        try:
            for it in range(start_iter, stage.max_iterations):
                interrupted_iter = it
                try:
                    result = algorithm.train()
                except Exception as train_exc:
                    print(f"\n[Stage {stage.index}] algorithm.train() failed at iter {it}: {train_exc}")
                    self._emergency_save(algorithm, stage, it, state, train_exc)
                    raise

                metrics = self._collect_metrics(result, stage.index, it, algorithm)
                metric_window.append(metrics)

                self._csv_writer.writerow({
                    "stage": stage.index, "iter_in_stage": it,
                    "total_iter": self._total_iter, **metrics,
                })
                self._csv_file.flush()
                self.policy_probe_logger.maybe_log(
                    algorithm,
                    iteration=self._total_iter,
                    sampled_steps=self._total_iter,
                    stage=stage.index,
                )
                self.engagement_replay_logger.env_config = stage_env_config
                self.engagement_replay_logger.maybe_log(
                    algorithm,
                    iteration=self._total_iter,
                    sampled_steps=self._total_iter,
                    stage=stage.index,
                )

                stage_state["iterations_trained"]    = it + 1
                stage_state["metric_history"]        = metric_window[-stage.advance_window * 2:]
                state["total_iterations_elapsed"]    = self._total_iter
                state["updated_at"]                  = _now()
                self._save_state(state)
                self._total_iter += 1

                if (it + 1) % stage.checkpoint_interval == 0:
                    self._save_checkpoint(algorithm, stage, it + 1, stage_state, state)

                self._print_row(stage.index, it, metrics)

                window = metric_window[-stage.advance_window:]
                if len(window) >= stage.advance_window:
                    ok, reason = check_advancement(stage, window)
                    if ok:
                        print(f"\n  ✓ Stage {stage.index} advancement: {reason}")
                        break

            stage_state["status"] = "completed"
            stage_state["advance_reason"] = reason if (
                'ok' in dir() and ok) else "max_iterations_reached"

        except KeyboardInterrupt:
            print(f"\n[Stage {stage.index}] Interrupted at iter {interrupted_iter}.")
            self._emergency_save(algorithm, stage, interrupted_iter, state,
                                 RuntimeError("KeyboardInterrupt"))
            stage_state["status"] = "interrupted"
            state["status"]       = "failed"
            self._save_state(state)
            algorithm.stop()
            raise

        except Exception as exc:
            stage_state["status"] = "failed"
            state["status"]       = "failed"
            state["last_error"]   = str(exc)
            self._save_state(state)
            algorithm.stop()
            raise

        finally:
            try:
                bundle_dir = self._stage_dir(stage) / "final_bundle"
                save_lightweight_policy_bundle(
                    algorithm, bundle_dir,
                    metadata={
                        "stage_index": stage.index,
                        "stage_name":  stage.name,
                        "algorithm":   self.algorithm_name,
                        "obs_mode":    stage_env_config.get(
                            "observation_mode",
                            self.args.observation_mode,
                        ),
                        "observation_module": stage_env_config.get("observation_module", ""),
                        "use_lstm": self.args.use_lstm,
                        "use_lstm_sac": self.args.use_lstm_sac,
                        "use_lstm_prioritized_replay": (
                            self.args.use_lstm_prioritized_replay
                            if self.args.use_lstm_sac else None
                        ),
                        "lstm_cell_size": (
                            self.args.lstm_cell_size
                            if self.args.use_lstm or self.args.use_lstm_sac
                            else None
                        ),
                        "max_seq_len": (
                            self.args.max_seq_len
                            if self.args.use_lstm or self.args.use_lstm_sac
                            else None
                        ),
                        "lstm_scope": self.args.lstm_scope if self.args.use_lstm_sac else None,
                        "network_spec": (
                            json.loads(self.args.network_spec_json)
                            if self.args.network_spec_json
                            else None
                        ),
                    },
                )
                stage_state["final_bundle_dir"] = str(bundle_dir)
                self._save_state(state)
                print(f"  → Final bundle saved: {bundle_dir}")
            except Exception as bexc:
                print(f"  [WARNING] Final bundle save failed: {bexc}")

            try:
                algorithm.stop()
            except Exception:
                pass

        try:
            record_dir = (ROOT / "artifacts" / "records" /
                          self.args.output_name / self.args.output_tag /
                          f"stage_{stage.index}")
            save_training_record(
                output_dir=record_dir,
                algorithm_name=self.algorithm_name,
                cli_args=vars(self.args),
                env_config=stage_env_config,
                algorithm_config={},
                result_history=metric_window,
                workspace_root=ROOT,
            )
        except Exception as rexc:
            print(f"  [WARNING] Training record save failed: {rexc}")

    def _build_algorithm(self, stage: CurriculumStage, env_config: dict, env_name: str):
        args = self.args
        config = build_algorithm_config(
            algorithm_name=self.algorithm_name,
            env_name=env_name,
            env_config=env_config,
            args={
                "framework":        args.framework,
                "num_env_runners":  args.num_env_runners,
                "lr":               args.lr,
                "gamma":            args.gamma,
                "train_batch_size": args.train_batch_size,
                "minibatch_size":   args.minibatch_size,
                "gae_lambda":       args.gae_lambda,
                "clip_param":       args.clip_param,
                "tau":              args.tau,
                "target_entropy":   args.target_entropy,
                "replay_buffer_capacity": args.replay_buffer_capacity,
                "model_config": _build_model_config_args(args),
                "network_spec": args.network_spec_json,
                "use_lstm": args.use_lstm,
                "use_lstm_sac": args.use_lstm_sac,
                "use_lstm_prioritized_replay": args.use_lstm_prioritized_replay,
                "lstm_scope": args.lstm_scope,
                "lstm_cell_size": args.lstm_cell_size,
                "max_seq_len": args.max_seq_len,
                "debug_io": args.debug_io,
            },
        )
        return config.build_algo()

    def _restore_weights(self, algorithm, stage: CurriculumStage,
                         stage_state: dict, state: dict):
        if self.args.restore_checkpoint and not self._explicit_checkpoint_restored:
            ckpt_path = Path(self.args.restore_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"restore checkpoint not found: {ckpt_path}")
            print(f"  [Restore] Explicit native checkpoint → {ckpt_path}")
            algorithm.restore(str(ckpt_path))
            self._explicit_checkpoint_restored = True
            stage_state["last_checkpoint_dir"] = str(ckpt_path)
            self._save_state(state)
            print("  [Restore] ✓ Explicit checkpoint restored successfully.")
            return

        if self.args.init_bundle and not self._explicit_bundle_loaded:
            bundle_path = Path(self.args.init_bundle)
            if not bundle_path.exists():
                raise FileNotFoundError(f"lightweight bundle not found: {bundle_path}")
            print(f"  [Restore] Explicit lightweight bundle → {bundle_path}")
            self._apply_bundle_weights(algorithm, str(bundle_path))
            self._explicit_bundle_loaded = True
            stage_state["restart_bundle_dir"] = str(bundle_path)
            self._save_state(state)
            print("  [Restore] ✓ Explicit bundle weights loaded successfully.")
            return

        ckpt = stage_state.get("last_checkpoint_dir")
        if ckpt and Path(ckpt).exists():
            print(f"  [Restore] Path 1: native checkpoint → {ckpt}")
            try:
                algorithm.restore(ckpt)
                print(f"  [Restore] ✓ Checkpoint restored successfully.")
                return
            except Exception as exc:
                print(f"  [Restore] ✗ Checkpoint restore failed: {exc}")

        if stage.index > 0:
            prev_bundle = state["stages"].get(str(stage.index - 1), {}).get("final_bundle_dir")
            if prev_bundle and Path(prev_bundle).exists():
                print(f"  [Restore] Path 2: previous-stage bundle → {prev_bundle}")
                try:
                    self._apply_bundle_weights(algorithm, prev_bundle)
                    print(f"  [Restore] ✓ Weight transfer from stage {stage.index - 1} succeeded.")
                    return
                except Exception as exc:
                    print(f"  [Restore] ✗ Weight transfer failed: {exc}")

            for prev_idx in range(stage.index - 2, -1, -1):
                fallback_bundle = state["stages"].get(str(prev_idx), {}).get("final_bundle_dir")
                if fallback_bundle and Path(fallback_bundle).exists():
                    print(f"  [Restore] Path 2b: fallback bundle stage {prev_idx} → {fallback_bundle}")
                    try:
                        self._apply_bundle_weights(algorithm, fallback_bundle)
                        print(f"  [Restore] ✓ Fallback transfer from stage {prev_idx} succeeded.")
                        return
                    except Exception as exc:
                        print(f"  [Restore] ✗ Fallback failed: {exc}")

            for prev_idx in range(stage.index - 1, -1, -1):
                emg_dirs = state["stages"].get(str(prev_idx), {}).get("emergency_dirs", [])
                for emg in reversed(emg_dirs):
                    emg_bundle = Path(emg) / "bundle"
                    if emg_bundle.exists():
                        print(f"  [Restore] Path 2c: emergency bundle → {emg_bundle}")
                        try:
                            self._apply_bundle_weights(algorithm, str(emg_bundle))
                            print(f"  [Restore] ✓ Emergency bundle transfer succeeded.")
                            return
                        except Exception as exc:
                            print(f"  [Restore] ✗ Emergency bundle failed: {exc}")

        print(f"  [Restore] Path 3: fresh start (no prior weights loaded).")

    def _apply_bundle_weights(self, algorithm, bundle_dir: str):
        apply_lightweight_policy_bundle(algorithm, bundle_dir)

    def _save_checkpoint(self, algorithm, stage: CurriculumStage, iteration: int,
                         stage_state: dict, state: dict):
        ckpt_dir = self._stage_dir(stage) / "checkpoints" / f"iter_{iteration:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            algorithm.save(str(ckpt_dir))
            stage_state["last_checkpoint_dir"] = str(ckpt_dir)
            self._save_state(state)
            print(f"  [Checkpoint] Saved iter {iteration} → {ckpt_dir}")
        except Exception as exc:
            print(f"  [WARNING] Checkpoint save failed at iter {iteration}: {exc}")

    def _emergency_save(self, algorithm, stage: CurriculumStage,
                        iteration: int, state: dict, exc: Exception):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emg_root = self._stage_dir(stage) / "emergency" / timestamp
        emg_root.mkdir(parents=True, exist_ok=True)
        print(f"\n  [EMERGENCY] Saving to {emg_root}")
        try:
            tb = traceback.format_exc()
            (emg_root / "exception.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{tb}", encoding="utf-8"
            )
        except Exception:
            pass
        bundle_ok = False
        try:
            save_lightweight_policy_bundle(algorithm, emg_root / "bundle")
            bundle_ok = True
            print(f"  [EMERGENCY] ✓ Bundle saved.")
        except Exception as e2:
            print(f"  [EMERGENCY] ✗ Bundle failed: {e2}")
        ckpt_ok = False
        try:
            algorithm.save(str(emg_root / "checkpoint"))
            ckpt_ok = True
            print(f"  [EMERGENCY] ✓ Native checkpoint saved.")
        except Exception as e2:
            print(f"  [EMERGENCY] ✗ Native checkpoint failed: {e2}")
        try:
            ss = state["stages"][str(stage.index)]
            emg_entry = {
                "timestamp": timestamp, "iteration": iteration,
                "bundle_ok": bundle_ok, "ckpt_ok": ckpt_ok,
                "path": str(emg_root),
            }
            ss.setdefault("emergency_dirs", []).append(str(emg_root))
            ss["last_emergency"] = emg_entry
            if ckpt_ok:
                ss["last_checkpoint_dir"] = str(emg_root / "checkpoint")
            if bundle_ok:
                ss["last_bundle_dir"] = str(emg_root / "bundle")
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            self._save_state(state)
        except Exception as e3:
            print(f"  [EMERGENCY] State update failed: {e3}")

    def _init_or_load_state(self) -> dict:
        if self.args.resume and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded state from {self.state_path}")
            print(f"  Current stage: {state['current_stage']}  "
                  f"Total iters: {state['total_iterations_elapsed']}")
            self._total_iter = state.get("total_iterations_elapsed", 0)
            return state

        if self.state_path.exists() and not self.args.resume:
            raise RuntimeError(
                f"State file exists at {self.state_path}.\n"
                "  Use --resume to continue, or choose a different --output-tag."
            )

        state = {
            "output_name":               self.args.output_name,
            "output_tag":                self.args.output_tag,
            "algorithm":                 self.algorithm_name,
            "obs_mode":                  self.base_env_config.get(
                "observation_mode",
                self.args.observation_mode,
            ),
            "status":                    "in_progress",
            "current_stage":             self.stages[0].index,
            "current_iteration":         0,
            "total_iterations_elapsed":  0,
            "created_at":                _now(),
            "updated_at":                _now(),
            "notes":                     self.args.notes,
            "stages": {
                str(s.index): {
                    "status":              "pending",
                    "iterations_trained":  0,
                    "advance_reason":      None,
                    "final_bundle_dir":    None,
                    "last_bundle_dir":      None,
                    "last_checkpoint_dir": None,
                    "emergency_dirs":      [],
                    "metric_history":      [],
                }
                for s in self.stages
            },
        }
        if self.args.start_stage is not None:
            state["current_stage"] = self.args.start_stage
        self._save_state(state)
        return state

    def _save_state(self, state: dict):
        state["updated_at"] = _now()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _stage_dir(self, stage: CurriculumStage) -> Path:
        d = self.curriculum_dir / f"stage_{stage.index}_{stage.name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _collect_metrics(
        self,
        result: dict,
        stage_idx: int,
        it: int,
        algorithm: Any,
    ) -> dict:
        env_m   = result.get("env_runners", {})
        custom  = _extract_custom_metrics(result)
        learner = _extract_learner_stats(result)
        _fill_algorithm_runtime_stats(learner, algorithm)
        return {
            "stage":             stage_idx,
            "iter_in_stage":     it,
            "total_iter":        self._total_iter,
            "reward_mean":       env_m.get("episode_return_mean", "n/a"),
            "ep_len_mean":       env_m.get("episode_len_mean",    "n/a"),
            **custom,
            **learner,
        }

    def _print_row(self, stage_idx: int, it: int, m: dict):
        base = (
            f"stage=[{stage_idx}] | iter=[{it}] | "
            f"total_iter=[{m['total_iter']}] | "
            f"Reward=[{_fmt(m['reward_mean'], '.4f')}] | "
            f"WinRate=[{_fmt(m['win_rate'], '.3f')}] | "
            f"Crash=[{_fmt(m['crash_rate'], '.3f')}] | "
            f"WEZ=[{_fmt(m['ep_wez_steps'], '.1f')}] | "
        )
        if self.algorithm_name == "sac":
            print(
                base
                + f"Actor=[{_fmt(m['actor_loss'], '.4f')}] | "
                f"Critic=[{_fmt(m['critic_loss'], '.4f')}] | "
                f"Alpha=[{_fmt(m['alpha'], '.4f')}] | "
                f"ReplayMem=[{_fmt(m['replay_buffer_memory_mb'], '.1f')}MB]"
            )
            return
        print(
            base
            + f"Entropy=[{_fmt(m['entropy'], '.4f')}] | "
            f"VF_loss=[{_fmt(m['vf_loss'], '.4f')}]"
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _sync_lstm_args_from_init_bundle(args)
    trainer = CurriculumTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
