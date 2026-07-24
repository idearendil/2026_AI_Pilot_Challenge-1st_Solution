# -*- coding: utf-8 -*-
"""claude_code REDQ 학습 entrypoint (discrete SAC + REDQ off-policy).

기존 PPO 트랙(train.py)과 **완전히 독립**된 병행 트랙이다. env/관측/보상/번들 export 등
공용 인프라만 재사용하고 학습 알고리즘은 claude_code.redq 로 새로 얹는다.

Phase 0: 단일 프로세스 + scripted 상대(loiter). SAC 코어 검증용.
  python claude_code/train_redq.py --no-self-play --target-mode loiter \
      --total-env-steps 200000 --ensemble-size 1 --subset-size 1 --utd 1 \
      --output-name redq01 --output-tag phase0_sanity

배포 호환: 최종 정책(MLPDiscreteActor)은 기존 metadata.json + policy_weights.pkl.gz 번들로
export 되며, 기존 MLPActionProvider / 평가 하네스가 그대로 로드·구동한다.

* GPU 환경(aip_gpu, torch cu13x)에서 실행 전제. --device cuda.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from pathlib import Path

# CUDA + Ray 를 한 프로세스에서 같이 쓰면 Windows 에서 driver 가 간헐적 access violation 을
# 낸다(수 사이클 후 크래시). PyTorch 기본 caching allocator 대신 CUDA 드라이버의 async
# allocator(cudaMallocAsync)를 쓰면 Ray 백그라운드 스레드와의 충돌을 피할 여지가 있다.
# torch 가 CUDA 를 초기화하기 전에 설정해야 하므로 import 보다 먼저 둔다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# 콘솔 인코딩(cp949 등)에 없는 문자(예: em-dash)를 print 해도 UnicodeEncodeError 로
# 학습이 죽지 않게 한다. 인코딩은 유지하고 불가 문자만 안전 대체.
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np
import torch

# PPO 트랙과 동일한 wandb 키(환경변수 우선). 이 파일도 공개 repo 에 push 금지.
_WANDB_API_KEY = "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"

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# baseline BT rule XML (self-play 일 때만 의미. 다른 claude_code import 보다 먼저).
# self-play 는 기본 켜짐이므로 --no-self-play 가 없으면 rule 을 적용한다(train.py 와 동일).
from claude_code.bt_rule import apply_rule_env, DEFAULT_BT_DLL  # noqa: E402

if "--no-self-play" not in sys.argv:
    apply_rule_env()

from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.model import save_bundle, MLPDiscreteActor  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402
from claude_code.ppo import (_outcome_counts, _count_altitude_terms,  # noqa: E402
                             _outcome_counts_by_opp)
from claude_code.redq.config import RedqConfig  # noqa: E402
from claude_code.redq.trainer import RedqTrainer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="claude_code REDQ (discrete SAC) trainer for DogFight 1v1")
    # 학습 규모
    p.add_argument("--total-env-steps", type=int, default=5_000_000)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--collect-steps", type=int, default=1000, help="사이클당 수집 env-step")
    p.add_argument("--gamma", type=float, default=0.99)
    # replay
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-buffer", type=int, default=5_000)
    # 네트워크
    p.add_argument("--actor-hidden", default="512,512,512")
    p.add_argument("--actor-activation", default="relu", choices=["tanh", "relu", "elu"])
    p.add_argument("--critic-hidden", default="512,512,512")
    p.add_argument("--critic-activation", default="relu", choices=["tanh", "relu", "elu"])
    p.add_argument("--no-critic-layernorm", action="store_true",
                   help="critic LayerNorm 끄기(기본 켜짐). 높은 UTD Q 발산 방지 장치.")
    p.add_argument("--action-bins", type=int, default=21)
    # 최적화
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    # SAC entropy
    p.add_argument("--target-entropy-ratio", type=float, default=0.5)
    p.add_argument("--init-alpha", type=float, default=0.1)
    p.add_argument("--no-autotune-alpha", action="store_true")
    p.add_argument("--alpha-min", type=float, default=1e-4)
    p.add_argument("--alpha-max", type=float, default=2.0,
                   help="log_alpha clamp 상한(alpha runaway 방지). 높은 UTD 에서 특히 중요.")
    # REDQ
    p.add_argument("--ensemble-size", type=int, default=10, help="N: Q 앙상블 크기")
    p.add_argument("--subset-size", type=int, default=2, help="M: target min subset")
    p.add_argument("--utd", type=int, default=10, help="UTD ratio: env-step 당 update 횟수")
    p.add_argument("--tau", type=float, default=0.005)
    # DroQ (Plan B)
    p.add_argument("--droq", action="store_true", help="앙상블 대신 dropout+LN Q 2개")
    p.add_argument("--droq-dropout", type=float, default=0.01)
    p.add_argument("--droq-ensemble", type=int, default=2)
    # self-play / 상대
    p.add_argument("--self-play", dest="self_play", action="store_true", default=True)
    p.add_argument("--no-self-play", dest="self_play", action="store_false")
    p.add_argument("--target-mode", default="loiter",
                   help="--no-self-play 일 때 scripted 상대(loiter/fixed/behavior_tree/autopilot)")
    # opponent pool (self-play). PPO train.py 와 동일 규약: slot0=BT 고정, 그 뒤 snapshot.
    p.add_argument("--pool-size", type=int, default=6, help="pool 최대 슬롯(BT 포함)")
    p.add_argument("--selfplay-gate-threshold", type=float, default=0.6,
                   help="snapshot 후보들의 min-EMA 가 이 값 이상이면 현재 정책을 pool 에 추가")
    p.add_argument("--selfplay-ema-alpha", type=float, default=0.1)
    p.add_argument("--pool-sample-temp", type=float, default=0.3,
                   help="opponent 샘플링 softmax(-ema/τ) 온도")
    p.add_argument("--no-bt-opponent", action="store_true",
                   help="self-play pool 에 baseline BT 를 넣지 않음")
    # 관측/보상 모듈 (PPO 와 동일 기본값)
    p.add_argument("--reward-module", default="claude_code.my_reward")
    p.add_argument("--observation-module", default="claude_code.my_observation")
    # 인프라
    p.add_argument("--num-workers", type=int, default=physical_cpu_count())
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    # 저장/로깅
    p.add_argument("--output-name", default="redq01")
    p.add_argument("--output-tag", default="sac_redq_v1")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--save-every-cycles", type=int, default=50, help="N 사이클마다 번들 저장")
    p.add_argument("--log-every-cycles", type=int, default=1)
    # crash 자동 재시작(Windows CUDA+Ray 네이티브 크래시 우회). self-play(Phase 1) 전용.
    p.add_argument("--supervise", action="store_true",
                   help="바깥 supervisor 로 학습을 자식 프로세스로 띄우고, 크래시하면 마지막 "
                        "체크포인트에서 자동 재시작(총 step 도달까지). CUDA+Ray 크래시 우회용.")
    p.add_argument("--auto-resume", action="store_true",
                   help="시작 시 체크포인트가 있으면 이어서 학습(supervisor 가 자식에 자동 부여).")
    p.add_argument("--checkpoint-every-cycles", type=int, default=1,
                   help="N 사이클마다 학습상태 체크포인트 저장(replay 제외, 네트워크/pool/카운터).")
    p.add_argument("--max-restarts", type=int, default=1000,
                   help="supervisor 최대 재시작 횟수(무한루프 방지).")
    p.add_argument("--restart-timeout", type=float, default=90.0,
                   help="supervisor watchdog: heartbeat 가 이 초 동안 안 갱신되면 죽음/hang 으로 "
                        "보고 자식 트리를 죽여 재시작. heartbeat 는 5초마다 찍히므로(살아있으면 "
                        "느린 update 중에도 계속 tick) 이 값은 짧아도 된다.")
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="AIP contest")
    p.add_argument("--wandb-run-name", default="")
    return p.parse_args()


def _tuple(s):
    return tuple(int(x) for x in str(s).split(",") if x != "")


def build_config(args) -> RedqConfig:
    return RedqConfig(
        total_env_steps=args.total_env_steps, warmup_steps=args.warmup_steps,
        collect_steps_per_cycle=args.collect_steps, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        min_buffer_for_update=args.min_buffer,
        actor_hidden=_tuple(args.actor_hidden), actor_activation=args.actor_activation,
        critic_hidden=_tuple(args.critic_hidden), critic_activation=args.critic_activation,
        critic_layernorm=not args.no_critic_layernorm,
        num_bins=args.action_bins, act_dim=4,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        max_grad_norm=args.max_grad_norm,
        target_entropy_ratio=args.target_entropy_ratio, init_alpha=args.init_alpha,
        autotune_alpha=not args.no_autotune_alpha,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        ensemble_size=args.ensemble_size, subset_size=args.subset_size,
        utd_ratio=args.utd, tau=args.tau,
        use_droq=args.droq, droq_dropout=args.droq_dropout, droq_ensemble_size=args.droq_ensemble,
        self_play=bool(args.self_play),
        normalize_obs=True, obs_clip=10.0,
        reconstruct_state=(args.observation_module == "claude_code.my_observation"),
        num_workers=args.num_workers, device=args.device, seed=args.seed)


def _setup_logging(cfg, args, resume=False):
    """CSV writer + wandb 초기화. (writer, log_file, wb) 반환. resume 면 CSV append + wandb resume."""
    log_dir = Path(args.artifacts_dir) / "logs" / args.output_name / args.output_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "redq_training_log.csv"
    append = resume and csv_path.exists()
    log_file = csv_path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not append:
        writer.writerow([
            "cycle", "global_step", "grad_steps", "achieved_utd",
            "mean_return", "mean_length", "completed_episodes",
            "ep_damage", "ep_shaping", "ep_terminal",
            "win", "loss", "draw", "raw_win_rate", "altitude_term",
            "bt_win_rate", "bt_games", "pool_size", "opp_added",
            "q_loss", "actor_loss", "alpha", "alpha_loss", "policy_entropy",
            "q_mean", "q_ensemble_std", "q_pessimism", "td_target_mean", "explained_variance",
            "buffer_size", "buffer_age_p50", "buffer_age_p90", "buffer_bt_frac",
            "elapsed_sec",
        ])
    wb = None
    if args.wandb:
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
            run_name = args.wandb_run_name or f"{args.output_name}/{args.output_tag}"
            # 재시작 간 같은 run 에 이어 붙이도록 run id 를 output-name/tag 로 고정.
            run_id = f"redq-{args.output_name}-{args.output_tag}".replace("/", "-")
            wandb.init(project=args.wandb_project, name=run_name, id=run_id,
                       resume="allow",
                       config={**vars(args), "target_entropy": cfg.target_entropy()})
            wb = wandb
            print(f"[REDQ] wandb 활성: {args.wandb_project} / {run_name}", flush=True)
        except Exception as e:
            print(f"[REDQ] wandb 초기화 실패({e}) → 로깅 없이 진행", flush=True)
    return writer, log_file, wb


def _log_cycle(writer, wb, cfg, cycle, global_step, grad_steps, cstats, umetrics,
               age, elapsed, ema_win, bt_wr=float("nan"), bt_games=0,
               pool_size=0, opp_added=0):
    """한 사이클 지표를 CSV + wandb 에 기록. (oc, derived) 반환."""
    ep_ret = cstats["ep_returns"]
    mean_ret = float(np.mean(ep_ret)) if ep_ret else float("nan")
    mean_len = float(np.mean(cstats["ep_lengths"])) if cstats["ep_lengths"] else float("nan")
    comps = cstats["ep_components"]
    def _cmean(key):
        return float(np.mean([c.get(key, 0.0) for c in comps])) if comps else float("nan")
    ep_damage, ep_shaping, ep_terminal = _cmean("damage"), _cmean("shaping"), _cmean("terminal")
    oc = _outcome_counts(cstats["ep_outcomes"])
    alt_term = _count_altitude_terms(cstats["ep_ends"])
    achieved_utd = grad_steps / max(global_step, 1)

    g = umetrics.get
    q_loss, actor_loss = g("q_loss", float("nan")), g("actor_loss", float("nan"))
    alpha, alpha_loss = g("alpha", float("nan")), g("alpha_loss", float("nan"))
    entropy, q_mean = g("entropy", float("nan")), g("q_mean", float("nan"))
    q_std, td_mean = g("q_ensemble_std", float("nan")), g("td_target_mean", float("nan"))
    q_pess = g("q_pessimism", float("nan"))
    ev = g("explained_variance", float("nan"))

    writer.writerow([
        cycle, global_step, grad_steps, f"{achieved_utd:.3f}",
        f"{mean_ret:.4f}", f"{mean_len:.1f}", len(ep_ret),
        f"{ep_damage:.4f}", f"{ep_shaping:.4f}", f"{ep_terminal:.4f}",
        oc["win"], oc["loss"], oc["draw"], f"{oc['raw_win_rate']:.4f}", alt_term,
        f"{bt_wr:.4f}", bt_games, pool_size, opp_added,
        f"{q_loss:.5f}", f"{actor_loss:.5f}", f"{alpha:.5f}", f"{alpha_loss:.6f}",
        f"{entropy:.4f}", f"{q_mean:.4f}", f"{q_std:.5f}", f"{q_pess:.5f}", f"{td_mean:.4f}", f"{ev:.4f}",
        age["count"], f"{age['age_p50']:.0f}", f"{age['age_p90']:.0f}", f"{age['bt_frac']:.3f}",
        f"{elapsed:.2f}",
    ])
    if wb is not None:
        try:
            d = {
                "global_step": global_step, "cycle": cycle,
                "train/grad_steps": grad_steps, "train/achieved_utd": achieved_utd,
                "train/mean_return": mean_ret, "train/mean_length": mean_len,
                "train/completed_episodes": len(ep_ret),
                "reward/damage": ep_damage, "reward/shaping": ep_shaping,
                "reward/terminal": ep_terminal,
                "selfplay/raw_win_rate": oc["raw_win_rate"], "selfplay/ema_win_rate": ema_win,
                "selfplay/wins": oc["win"], "selfplay/losses": oc["loss"], "selfplay/draws": oc["draw"],
                "selfplay/pool_size": pool_size, "selfplay/opp_added": opp_added,
                "train/altitude_termination": alt_term,
                "loss/q_loss": q_loss, "loss/actor_loss": actor_loss, "loss/alpha_loss": alpha_loss,
                "sac/alpha": alpha, "policy/entropy": entropy,
                "policy/entropy_frac": (entropy / cfg.target_entropy() if cfg.target_entropy() else float("nan")),
                "critic/q_mean": q_mean, "critic/q_ensemble_std": q_std,
                "critic/q_pessimism": q_pess,
                "critic/td_target_mean": td_mean, "critic/explained_variance": ev,
                "buffer/size": age["count"], "buffer/age_p50": age["age_p50"],
                "buffer/age_p90": age["age_p90"], "buffer/bt_frac": age["bt_frac"],
                "buffer/n_opponents": age["n_opponents"],
            }
            if bt_games > 0:
                d["selfplay/bt_win_rate"] = bt_wr
                d["selfplay/bt_games"] = bt_games
            wb.log(d, step=global_step)
        except Exception as e:
            print(f"[REDQ] wandb.log 실패({e})", flush=True)

    return {"mean_ret": mean_ret, "mean_len": mean_len, "ep_damage": ep_damage,
            "oc": oc, "achieved_utd": achieved_utd, "q_loss": q_loss,
            "actor_loss": actor_loss, "alpha": alpha, "entropy": entropy,
            "q_std": q_std, "ev": ev}


def _base_metadata(cfg, args, obs_mode):
    return {
        "framework": "claude_code_redq",
        "algorithm": "discrete_sac_redq",
        "observation_module": args.observation_module,
        "observation_mode": obs_mode,
        "reward_module": args.reward_module,
        "redq": {"ensemble_size": cfg.ensemble_size, "subset_size": cfg.subset_size,
                 "utd_ratio": cfg.utd_ratio, "tau": cfg.tau,
                 "target_entropy_ratio": cfg.target_entropy_ratio},
    }


def _supervise_loop(args):
    """바깥 supervisor: 학습을 자식 프로세스로 반복 실행하고 크래시하면 마지막
    체크포인트에서 자동 재시작(총 step 도달까지). 공용 로직은 claude_code.supervisor."""
    from claude_code.supervisor import supervise_loop, build_child_cmd
    cmd = build_child_cmd(__file__)
    hb = (Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag
          / "redq_ckpt" / "heartbeat")
    supervise_loop(cmd, hb, timeout_s=float(args.restart_timeout),
                   max_restarts=int(args.max_restarts), label="REDQ")


def main():
    args = parse_args()
    if args.supervise:
        if not args.self_play:
            print("[REDQ] --supervise 는 self-play(Phase 1) 전용입니다.", flush=True)
            raise SystemExit(2)
        _supervise_loop(args)
        return
    cfg = build_config(args)
    if args.self_play:
        run_selfplay(args, cfg)
    else:
        run_phase0(args, cfg)


def run_phase0(args, cfg):
    """Phase 0: 단일 프로세스 + scripted 상대. SAC 코어 검증용(self-play 없음)."""
    import time
    env_kwargs = dict(overrides={"target_mode": args.target_mode},
                      reward_module=args.reward_module,
                      observation_module=args.observation_module)
    trainer = RedqTrainer(cfg, env_kwargs=env_kwargs)
    obs_mode = trainer.env.config.get("observation_mode", "tactical16")
    print(f"[REDQ/Phase0] obs={trainer.obs_dim}D act={trainer.act_dim} "
          f"device={trainer.learner.device} N={cfg.ensemble_size} M={cfg.subset_size} "
          f"UTD={cfg.utd_ratio} target_entropy={cfg.target_entropy():.2f}", flush=True)
    base_metadata = _base_metadata(cfg, args, obs_mode)
    bundle_dir = Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag
    writer, log_file, wb = _setup_logging(cfg, args)

    cycle, ema_win = 0, float("nan")
    try:
        while trainer.global_step < cfg.total_env_steps:
            t0 = time.time()
            cstats = trainer.collect(cfg.collect_steps_per_cycle)
            umetrics = trainer.update(cfg.collect_steps_per_cycle)
            cycle += 1
            age = trainer.replay.age_stats(trainer.global_step)
            oc0 = _outcome_counts(cstats["ep_outcomes"])
            if not np.isnan(oc0["raw_win_rate"]):
                ema_win = oc0["raw_win_rate"] if np.isnan(ema_win) else 0.9 * ema_win + 0.1 * oc0["raw_win_rate"]
            d = _log_cycle(writer, wb, cfg, cycle, trainer.global_step, trainer.grad_steps,
                           cstats, umetrics, age, time.time() - t0, ema_win)
            log_file.flush()
            if cycle % args.log_every_cycles == 0:
                _print_cycle(cycle, trainer.global_step, d, cfg, age)
            if cycle % args.save_every_cycles == 0:
                _export_bundle(trainer, bundle_dir, base_metadata)
    finally:
        _export_bundle(trainer, bundle_dir, base_metadata)
        log_file.close()
        trainer.close()
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass
    print(f"[REDQ] 완료: {bundle_dir}", flush=True)


def run_selfplay(args, cfg):
    """Phase 1: Ray 병렬 + opponent pool self-play (BT slot0 고정)."""
    import time
    from claude_code.redq.parallel import RedqParallelTrainer

    env_kwargs = dict(overrides={"target_mode": "loiter"},   # self-play 면 target provider 로 대체됨
                      reward_module=args.reward_module,
                      observation_module=args.observation_module)
    bt_dll = "" if args.no_bt_opponent else DEFAULT_BT_DLL
    # obs/act 차원은 관측 모듈에서 직접 얻는다(드라이버 프로세스에 JSBSim 을 로드하지 않기
    # 위해). JSBSim 을 드라이버에 올린 뒤 close() 하고 CUDA 작업을 하면 네이티브 access
    # violation 이 날 수 있어, 드라이버는 env 를 절대 만들지 않는다(수집은 worker 전담).
    if args.observation_module == "claude_code.my_observation":
        from claude_code.my_observation import OBSERVATION_SIZE, OBSERVATION_MODE
        obs_dim, obs_mode = int(OBSERVATION_SIZE), OBSERVATION_MODE
    else:
        obs_dim, obs_mode = 16, "tactical16"   # 기본 tactical16 fallback
    act_dim = 4   # DogFight 행동 공간 = [roll, pitch, rudder, throttle]

    # supervisor watchdog 용 heartbeat 를 **시작 즉시** 띄운다(Ray init/무거운 update 중에도
    # 데몬 스레드가 계속 tick → '느림'을 '죽음'으로 오판하지 않음). 프로세스가 죽으면 멈춘다.
    from claude_code.supervisor import start_heartbeat
    hb_dir = Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag / "redq_ckpt"
    hb_stop = start_heartbeat(hb_dir / "heartbeat")

    trainer = RedqParallelTrainer(cfg, env_kwargs, obs_dim, act_dim, bt_dll=bt_dll, bt_rule="")
    print(f"[REDQ/Phase1] obs={obs_dim}D act={act_dim} device={trainer.device_str} "
          f"workers={cfg.num_workers} N={cfg.ensemble_size} M={cfg.subset_size} UTD={cfg.utd_ratio} "
          f"target_entropy={cfg.target_entropy():.2f} BT={'off' if args.no_bt_opponent else DEFAULT_BT_DLL}",
          flush=True)

    bt_slots = 1 if bt_dll else 0
    pool_max = max(1 + bt_slots, int(args.pool_size))
    trainer.install_pool(pool_max, seed=cfg.seed)

    def _pool_weights():
        e = np.asarray([x["ema"] for x in trainer.pool], dtype=np.float64)
        if e.size <= 1:
            return [1.0] * max(e.size, 1)
        logits = -e / max(float(args.pool_sample_temp), 1e-6)
        logits -= logits.max()
        w = np.exp(logits)
        return (w / w.sum()).tolist()

    # crash 자동 재시작: 체크포인트가 있으면 이어서 학습(learner+pool+obs_rms+카운터 복원).
    bundle_dir = Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag
    ckpt_dir = bundle_dir / "redq_ckpt"
    start_cycle = 0
    resumed = False
    if args.auto_resume and (ckpt_dir / "driver.pt").exists():
        start_cycle = trainer.resume_from(ckpt_dir)
        resumed = True
        print(f"[REDQ/Phase1] 체크포인트 재시작: cycle {start_cycle}, "
              f"global_step {trainer.global_step}, pool {len(trainer.pool)} "
              f"(replay 는 비어서 시작 → 다시 채움)", flush=True)

    trainer.set_pool_weights(_pool_weights())
    base_metadata = _base_metadata(cfg, args, obs_mode)
    base_metadata["selfplay_gate_threshold"] = args.selfplay_gate_threshold
    base_metadata["pool_size"] = pool_max
    base_metadata["bt_opponent"] = bt_dll or None
    writer, log_file, wb = _setup_logging(cfg, args, resume=resumed)

    cycle, ema_win = start_cycle, float("nan")
    try:
        while trainer.global_step < cfg.total_env_steps:
            t0 = time.time()
            agg = trainer.collect_and_store(cfg.collect_steps_per_cycle)
            umetrics = trainer.update(cfg.collect_steps_per_cycle)
            cycle += 1

            # opponent 별 이번 사이클 성적으로 EMA 갱신 (slot index 기준).
            per_opp = _outcome_counts_by_opp(agg["ep_opp_indices"], agg["ep_outcomes"])
            alpha_ema = args.selfplay_ema_alpha
            for i, entry in enumerate(trainer.pool):
                dd = per_opp.get(i)
                if dd and dd.get("decided", 0) > 0:
                    entry["ema"] = (1 - alpha_ema) * entry["ema"] + alpha_ema * dd["raw_win_rate"]

            # baseline BT(slot0) 성적.
            bt_stat = per_opp.get(0) if bt_slots else None
            bt_games = int(bt_stat.get("decided", 0)) if bt_stat else 0
            bt_wr = float(bt_stat["raw_win_rate"]) if bt_games > 0 else float("nan")

            age = trainer.buffer_age_stats()

            # gate: snapshot 후보들의 min-EMA 가 임계값 이상이면 현재 정책 추가.
            opp_added = 0
            net_emas = [e["ema"] for e in trainer.pool if e["kind"] == "net"]
            can_update = (trainer.global_step >= cfg.warmup_steps
                          and age["count"] >= cfg.min_buffer_for_update)
            if can_update and net_emas and min(net_emas) >= args.selfplay_gate_threshold:
                new_gen, ev_gen, purged = trainer.add_current_to_pool()
                opp_added = 1
                print(f"[REDQ] *POOL+ gen{new_gen} 추가"
                      + (f" (evict gen{ev_gen}, replay {purged} purge)" if ev_gen is not None else ""),
                      flush=True)
            trainer.set_pool_weights(_pool_weights())
            cstats = {"ep_returns": agg["ep_returns"], "ep_lengths": agg["ep_lengths"],
                      "ep_components": agg["ep_components"], "ep_outcomes": agg["ep_outcomes"],
                      "ep_ends": agg["ep_end_conditions"]}
            oc0 = _outcome_counts(agg["ep_outcomes"])
            if not np.isnan(oc0["raw_win_rate"]):
                ema_win = oc0["raw_win_rate"] if np.isnan(ema_win) else 0.9 * ema_win + 0.1 * oc0["raw_win_rate"]
            d = _log_cycle(writer, wb, cfg, cycle, trainer.global_step, trainer.grad_steps,
                           cstats, umetrics, age, time.time() - t0, ema_win,
                           bt_wr=bt_wr, bt_games=bt_games, pool_size=len(trainer.pool),
                           opp_added=opp_added)
            log_file.flush()
            if cycle % args.log_every_cycles == 0:
                _print_cycle(cycle, trainer.global_step, d, cfg, age,
                             extra=f" | pool{len(trainer.pool)} bt_wr {bt_wr:.2f}")
            # crash 자동 재시작용 체크포인트(replay 제외). 크래시가 잦으므로 매 사이클 기본.
            if cycle % max(1, args.checkpoint_every_cycles) == 0:
                trainer.save_checkpoint(ckpt_dir, cycle)
            if cycle % args.save_every_cycles == 0:
                _export_bundle(trainer, bundle_dir, base_metadata)
    finally:
        hb_stop.set()
        _export_bundle(trainer, bundle_dir, base_metadata)
        log_file.close()
        trainer.close()
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass
    print(f"[REDQ] 완료: {bundle_dir}", flush=True)


def _print_cycle(cycle, global_step, d, cfg, age, extra=""):
    oc = d["oc"]
    print(f"cyc {cycle:4d} | step {global_step:8d} | ret {d['mean_ret']:8.3f} | "
          f"len {d['mean_len']:6.1f} | dmg {d['ep_damage']:6.3f} | wr {oc['raw_win_rate']:.2f} | "
          f"qL {d['q_loss']:7.3f} | aL {d['actor_loss']:8.3f} | alpha {d['alpha']:.3f} | "
          f"ent {d['entropy']:6.3f}/{cfg.target_entropy():.1f} | qstd {d['q_std']:.3f} | "
          f"ev {d['ev']:6.3f} | UTD {d['achieved_utd']:.1f} | buf {age['count']}{extra}",
          flush=True)


def _export_bundle(trainer, bundle_dir, base_metadata):
    """현재 actor 를 기존 2-파일 번들로 export (MLPActionProvider 호환).

    Phase 0(RedqTrainer)는 로컬 learner.actor 를, Phase 1(RedqParallelTrainer)은 GPU 액터
    에서 받은 state 로 CPU actor 를 복원해 저장한다.
    """
    meta = {**base_metadata, "global_step": trainer.global_step}
    if hasattr(trainer, "actor_state_for_bundle"):
        # 병렬(Phase 1): 액터 state → CPU MLPDiscreteActor 복원 후 저장.
        import torch as _torch
        state, obs_norm = trainer.actor_state_for_bundle()
        actor = MLPDiscreteActor(obs_dim=trainer.obs_dim, act_dim=trainer.act_dim,
                                 num_bins=trainer.cfg.num_bins,
                                 hidden=tuple(trainer.cfg.actor_hidden),
                                 activation=trainer.cfg.actor_activation)
        actor.load_state_dict({k: _torch.as_tensor(v) for k, v in state.items()})
        save_bundle(actor, bundle_dir, obs_norm=obs_norm, extra_metadata=meta)
    else:
        # 단일 프로세스(Phase 0).
        obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
        save_bundle(trainer.learner.actor, bundle_dir, obs_norm=obs_norm, extra_metadata=meta)


if __name__ == "__main__":
    main()
