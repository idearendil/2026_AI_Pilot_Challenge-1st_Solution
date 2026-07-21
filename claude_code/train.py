"""claude_code PPO 학습 entrypoint.

예시:
  python claude_code/train.py --iterations 50 --output-name team01 --output-tag ppo_mlp_v1

학습이 끝나면 원본과 동일한 2-파일 번들을 저장한다:
  artifacts/models/<output-name>/<output-tag>/
  ├── metadata.json
  └── policy_weights.pkl.gz

이 번들은 claude_code/submission.py 로 대결 서버에 바로 연결할 수 있다.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

# wandb API key. 환경변수 WANDB_API_KEY 가 있으면 그것을 우선 사용한다.
# 주의: 이 키가 소스에 하드코딩돼 있으므로 이 파일을 외부(공개 repo 등)에 commit/push
# 하지 않도록 유의할 것. 팀 공유 시엔 각자 환경변수로 넣는 방식을 권장.
_WANDB_API_KEY = "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"

# Release 루트/ src import 경로 등록 (단독 실행 대비).
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
from claude_code.model import save_bundle
from claude_code.parallel import logical_cpu_count
from claude_code.ppo import PPOConfig, PPOTrainer, IterationStats

TRAIN_CKPT_FORMAT = "claude_code_ppo_train_ckpt"
TRAIN_CKPT_VERSION = 1


def save_train_state(path, *, trainer, pool, pool_state, pool_max, iteration,
                     global_step, best, model_kwargs) -> None:
    """전체 학습 상태를 하나의 .pt 로 원자적 저장(중단돼도 이어서 학습 가능).

    저장: model(actor+critic) state_dict, actor/critic optimizer state, obs_rms,
    opponent pool(각 후보 actor net state + EMA + gen), pool 메타(next_gen 등),
    global_step, iteration(=직전 완료 iteration), best 추적. iteration+1 부터 재개한다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_rms = None
    if trainer.obs_rms is not None:
        obs_rms = {
            "mean": np.asarray(trainer.obs_rms.mean, dtype=np.float64),
            "var": np.asarray(trainer.obs_rms.var, dtype=np.float64),
            "count": float(trainer.obs_rms.count),
        }
    ckpt = {
        "format": TRAIN_CKPT_FORMAT,
        "version": TRAIN_CKPT_VERSION,
        "iteration": int(iteration),
        "global_step": int(global_step),
        "model_kwargs": dict(model_kwargs),
        "model_state": {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
        "actor_opt": trainer.actor_opt.state_dict(),
        "critic_opt": trainer.critic_opt.state_dict(),
        "obs_rms": obs_rms,
        "pool_max": int(pool_max),
        "next_gen": int(pool_state["next_gen"]),
        "n_added": int(pool_state["n_added"]),
        # 각 후보: EMA + actor net(+critic; state_dict 전체) + obs_rms 스냅샷.
        "pool": [{"gen": int(e["gen"]), "ema": float(e["ema"]),
                  "state": e["state"], "rms": e.get("rms")} for e in pool],
        "best": dict(best),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, str(tmp))
    os.replace(str(tmp), str(path))   # 원자적 교체 → 저장 중 중단돼도 기존 ckpt 보존


def load_train_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("format") != TRAIN_CKPT_FORMAT:
        raise ValueError(f"train-state checkpoint 형식이 아님: {path}")
    return ckpt


def parse_args():
    p = argparse.ArgumentParser(description="claude_code standalone PPO trainer for DogFight 1v1")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--rollout-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.00005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--hidden", default="512,512,512", help="actor hidden 크기, 예: 256,256")
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--log-std-init", type=float, default=-1.0, help="(이산 정책에서는 미사용)")
    p.add_argument("--action-bins", type=int, default=21,
                   help="각 행동 채널(roll/pitch/yaw/throttle)의 이산 카테고리 수 (균등 분할). "
                        "홀수여야 가운데 index=(n-1)/2 가 정확히 중립(0.0)이 된다.")
    # critic 을 actor 와 완전히 분리된 네트워크로 (구조/학습률 독립). 비우면 actor 와 동일.
    p.add_argument("--critic-hidden", default="", help="critic 전용 hidden (비우면 actor 와 동일)")
    p.add_argument("--critic-activation", default="", choices=["", "tanh", "relu", "elu"],
                   help="critic 전용 활성화 (비우면 actor 와 동일)")
    p.add_argument("--critic-lr", type=float, default=None,
                   help="critic 전용 학습률 (비우면 actor lr 공유)")
    p.add_argument("--no-normalize-obs", action="store_true", help="관측 정규화 끄기")
    # 기본값으로 claude_code 의 my_reward / my_observation 을 사용한다.
    # 프레임워크 기본 보상/관측을 쓰려면 빈 문자열을 넘긴다: --reward-module "" --observation-module ""
    p.add_argument("--reward-module", default="claude_code.my_reward",
                   help="보상 모듈 경로 (기본: claude_code.my_reward). 빈 값이면 프레임워크 기본 보상")
    p.add_argument("--observation-module", default="claude_code.my_observation",
                   help="관측 모듈 경로 (기본: claude_code.my_observation). 빈 값이면 tactical16")
    p.add_argument("--distance-reward-scale", type=float, default=None,
                   help="my_reward 의 distance_reward_scale 덮어쓰기. phase2 에서 거리 항을 "
                        "끄려면 0 을 준다. None 이면 모듈 기본값(0.001) 사용.")
    p.add_argument("--aim-reward-scale", type=float, default=None,
                   help="my_reward 의 aim_reward_scale 덮어쓰기. 직전 step 대비 줄어든 "
                        "|ATA|[deg] * 이 값. 0 이면 끔. None 이면 모듈 기본값(0.02) 사용.")
    p.add_argument("--resume-from", default="",
                   help="이어서 학습할 snapshot(.pt) 경로. actor+critic 가중치+obs_rms 를 불러와 "
                        "그 상태에서 학습 시작(phase1 → phase2). optimizer 모멘트는 새로 시작.")
    p.add_argument("--resume-state", default="",
                   help="전체 학습 상태 checkpoint(.pt) 에서 이어서 학습. model(actor+critic)+optimizer+"
                        "obs_rms+opponent pool(모든 후보 net & EMA)+global_step+iteration 을 복원해 "
                        "직전 iteration 다음부터 --iterations 까지 계속한다. (--resume-from 보다 우선)")
    p.add_argument("--checkpoint-path", default="",
                   help="매 iteration 직전/학습 종료 시 저장할 전체 학습 상태 .pt 경로. "
                        "비우면 claude_code/models/<name>/<tag>/train_state.pt 에 저장.")
    p.add_argument("--frozen-opponent", action="store_true",
                   help="self-play 상대를 학습 시작 시점의 actor net 으로 고정(학습 agent 와 분리). "
                        "phase2(--resume-from)와 함께 쓰면 상대=phase1 마지막 net 으로 고정.")
    # (구) 주기적 20판 evaluation 은 제거됨. 대신 매 iter rollout 게임의 opponent 대비
    # 승률로 EMA 를 갱신하고, EMA 가 임계값을 넘으면 opponent 를 현재 정책으로 승격한다.
    p.add_argument("--eval-interval", type=int, default=0, help="(미사용; 호환용)")
    p.add_argument("--eval-games", type=int, default=20, help="(미사용; 호환용)")
    p.add_argument("--eval-episodes", type=int, default=2, help="(미사용; 호환용)")
    p.add_argument("--selfplay-gate-threshold", type=float, default=0.6,
                   help="opponent pool: 모든 후보에 대한 EMA 승률 중 '최소값'이 이 값 이상이면 "
                        "현재 actor net 을 pool 에 새 후보로 추가(새 후보 EMA=0.5). "
                        "(--frozen-opponent 이면 무시=영구 고정)")
    p.add_argument("--selfplay-ema-alpha", type=float, default=0.1,
                   help="opponent 별 EMA 계수 α. ema_i = (1-α)·ema_i + α·(이번 iter 후보 i 상대 raw 승률). 초기 ema=0.5.")
    p.add_argument("--pool-size", type=int, default=5,
                   help="opponent pool 최대 크기. 초과 시 가장 오래 전에 추가된 후보를 제거(FIFO).")
    p.add_argument("--pool-sample-temp", type=float, default=0.3,
                   help="opponent 샘플링 softmax 온도 τ. weight_i ∝ exp(-ema_i/τ) → EMA 낮은 후보가 "
                        "더 자주 뽑힘. 작을수록 최저 EMA 후보를 강하게 선호.")
    p.add_argument("--seed", type=int, default=0)
    # loiter: 표적이 선회하며 고도를 유지(자기파괴 없음) → episode 가 timeout(terminal=0)
    # 으로 끝나므로 return 이 ownship 의 추격/사격 성과로만 결정돼 학습 신호가 깨끗하다.
    # fixed/autopilot 표적은 스스로 하강·추락해 매 episode 무승부(-30)로 끝나서 return 이
    # 정책 품질과 무관하게 ~-30 에 고정되므로 학습 시연에는 부적합하다.
    # behavior_tree 는 실제 대회형 강한 상대(처음부터 학습은 매우 어려움).
    p.add_argument("--target-mode", default="loiter",
                   choices=["fixed", "behavior_tree", "loiter", "autopilot"],
                   help="--no-self-play 일 때만 사용하는 스크립트 상대 종류")
    # 상대를 같은 actor network 로 조종(완전 self-learning). 기본 켜짐.
    p.add_argument("--self-play", dest="self_play", action="store_true", default=True,
                   help="상대를 같은 actor network 로 조종 (기본값)")
    p.add_argument("--no-self-play", dest="self_play", action="store_false",
                   help="self-play 끄고 --target-mode 스크립트 상대 사용")
    # Ray 병렬 데이터 수집. 기본 worker 수 = 논리 CPU 수(하이퍼스레딩 포함). 1 이면 단일 프로세스.
    p.add_argument("--num-workers", type=int, default=logical_cpu_count(),
                   help="Ray rollout worker 수 (기본=논리 코어 수). 1 이면 Ray 미사용")
    p.add_argument("--device", default="cpu",
                   help="driver update 디바이스 (큰 모델은 cuda). worker 는 항상 CPU 추론")
    p.add_argument("--output-name", default="team01")
    p.add_argument("--output-tag", default="ppo_mlp_v1")
    p.add_argument("--artifacts-dir", default="artifacts")
    # wandb 로깅 (기본 켜짐). 네트워크/키 문제로 실패하면 경고만 내고 학습은 계속된다.
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                   help="wandb 로깅 사용 (기본 켜짐)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help="wandb 로깅 끄기")
    p.add_argument("--wandb-project", default="AIP contest", help="wandb 프로젝트명")
    p.add_argument("--wandb-run-name", default="",
                   help="wandb run 이름 (비우면 output-name/output-tag)")
    return p.parse_args()


def main():
    args = parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    critic_hidden = (tuple(int(x) for x in args.critic_hidden.split(",") if x.strip())
                     if args.critic_hidden else None)
    critic_activation = args.critic_activation or None
    # phase2: 거리 보상 끄기 / 조준 shaping 켜기 등 reward 계수 덮어쓰기.
    reward_overrides = {}
    if args.distance_reward_scale is not None:
        reward_overrides["distance_reward_scale"] = args.distance_reward_scale
    if args.aim_reward_scale is not None:
        reward_overrides["aim_reward_scale"] = args.aim_reward_scale
    reward_overrides = reward_overrides or None
    # "opponent 에게 준 damage" 원값 복원용 damage_scale (my_reward: r_damage = 준damage×scale,
    # 받은damage 가중치 0 이므로 준damage = damage_reward / scale).
    dmg_scale = 10.0
    try:
        if args.reward_module == "claude_code.my_reward":
            from claude_code.my_reward import MY_REWARD_CONFIG
            dmg_scale = float(MY_REWARD_CONFIG.get("damage_scale", 10.0))
        if reward_overrides and "damage_scale" in reward_overrides:
            dmg_scale = float(reward_overrides["damage_scale"])
    except Exception:
        dmg_scale = 10.0
    env_kwargs = dict(
        overrides={"target_mode": args.target_mode},
        reward_module=args.reward_module,
        observation_module=args.observation_module,
        reward_overrides=reward_overrides,
    )
    env = make_env(**env_kwargs)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    parallel = args.num_workers > 1
    # snapshot/평가에서 과거 network 를 동일 구조로 재생성하기 위한 kwargs.
    model_kwargs = dict(
        obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, activation=args.activation,
        num_bins=args.action_bins,
        critic_hidden=critic_hidden, critic_activation=critic_activation,
    )
    meta_obs_mode = (env.config.get("observation_mode") if args.observation_module else "tactical16")
    print(f"[claude_code/PPO] obs={env.observation_space.shape} act={env.action_space.shape} "
          f"target_mode={args.target_mode} "
          f"reward_module={args.reward_module or '(default)'} "
          f"observation_module={args.observation_module or '(tactical16)'}")

    cfg = PPOConfig(
        total_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        hidden=hidden,
        activation=args.activation,
        log_std_init=args.log_std_init,
        num_bins=args.action_bins,
        critic_hidden=critic_hidden,
        critic_activation=critic_activation,
        critic_lr=args.critic_lr,
        normalize_obs=not args.no_normalize_obs,
        reconstruct_state=(args.observation_module == "claude_code.my_observation"),
        seed=args.seed,
        device=args.device,
    )

    # 데이터 수집: num_workers>1 이면 Ray 병렬, 아니면 단일 프로세스.
    if args.num_workers > 1:
        from claude_code.parallel import ParallelPPOTrainer
        env.close()   # driver 는 rollout env 를 step 하지 않음 (worker 가 가짐)
        trainer = ParallelPPOTrainer(env_kwargs, cfg, args.num_workers,
                                     args.self_play, obs_dim, act_dim)
    else:
        trainer = PPOTrainer(env, cfg)
    # gated self-play 상대는 resume(weights 확정) 후에 '현재 정책의 frozen copy' 로 설치한다.
    if args.self_play:
        print("[claude_code/PPO] 상대 = OPPONENT POOL self-play (매 게임 pool 에서 EMA 낮은 "
              f"후보 우대 샘플링, min-EMA≥{args.selfplay_gate_threshold:.2f} 시 현재 정책 추가; 둘 다 stochastic)")
    else:
        print(f"[claude_code/PPO] 상대 = 스크립트 target_mode={args.target_mode}")

    # 주기적 evaluation 은 제거됨(gated self-play 가 매 iter rollout 승률로 진척을 측정).
    eval_env = None

    bundle_dir = Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag
    base_metadata = {
        "output_name": args.output_name,
        "output_tag": args.output_tag,
        "target_mode": args.target_mode,
        "self_play": bool(args.self_play),
        "train_iterations": args.iterations,
        # 추론(submission/evaluate)이 동일 관측을 만들기 위해 모듈 경로를 기록.
        "reward_module": args.reward_module,
        "reward_overrides": reward_overrides,
        "observation_module": args.observation_module,
        "observation_mode": meta_obs_mode,
        "num_workers": args.num_workers,
        "env_config": {k: STANDARD_ENV_CONFIG[k] for k in
                       ("step_ratio", "max_engage_time", "episode_step_limit")},
    }
    best = {"return": -float("inf"), "win_rate": -1.0, "iter": -1, "saved": False}

    # ── wandb 초기화 (실패해도 학습은 계속) ──────────────────────────────────
    wb = None
    if args.wandb:
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
            run_name = args.wandb_run_name or f"{args.output_name}/{args.output_tag}"
            wandb.init(
                project=args.wandb_project, name=run_name,
                config={
                    "iterations": args.iterations, "rollout_steps": args.rollout_steps,
                    "lr": args.lr, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                    "clip_coef": args.clip_coef, "update_epochs": args.update_epochs,
                    "minibatch_size": args.minibatch_size, "ent_coef": args.ent_coef,
                    "vf_coef": args.vf_coef, "target_kl": args.target_kl,
                    "hidden": args.hidden, "activation": args.activation,
                    "action_bins": args.action_bins, "num_workers": args.num_workers,
                    "self_play": bool(args.self_play), "target_mode": args.target_mode,
                    "frozen_opponent": bool(args.frozen_opponent),
                    "selfplay_gate_threshold": (None if args.frozen_opponent
                                                else args.selfplay_gate_threshold),
                    "selfplay_ema_alpha": args.selfplay_ema_alpha,
                    "pool_size": (1 if args.frozen_opponent else max(1, int(args.pool_size))),
                    "pool_sample_temp": args.pool_sample_temp,
                    "reward_module": args.reward_module,
                    "observation_module": args.observation_module,
                    "reward_overrides": reward_overrides, "resume_from": args.resume_from,
                    "obs_dim": obs_dim, "act_dim": act_dim, "damage_scale": dmg_scale,
                },
            )
            wb = wandb
            print(f"[claude_code/PPO] wandb 로깅 활성: project='{args.wandb_project}' run='{run_name}'")
        except Exception as e:  # 네트워크/키 문제 등 → 로깅 없이 진행
            print(f"[claude_code/PPO] wandb 초기화 실패({e}) → wandb 로깅 없이 진행")
            wb = None

    # 매 iter actor network snapshot 저장 디렉토리 (평가 상대 + 재현용).
    from claude_code import evaluation
    snapshot_dir = Path("claude_code") / "models" / args.output_name / args.output_tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _snap_path(it: int) -> Path:
        return snapshot_dir / f"iter_{it:04d}.pt"

    # 전체 학습 상태 checkpoint 경로 (매 iteration 직전/학습 종료 시 저장).
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (snapshot_dir / "train_state.pt")

    # 전체 학습 상태(train-state) 이어가기 (--resume-state, 우선). model(actor+critic)+
    # optimizer+obs_rms+opponent pool+global_step+iteration 을 복원해 직전 iteration 다음부터
    # 계속한다. 아래 pool 설치 블록에서 pool 도 checkpoint 값으로 복원한다.
    resume_ckpt = None
    start_iter = 1
    if args.resume_state:
        resume_ckpt = load_train_state(args.resume_state)
        for k in ("obs_dim", "act_dim", "num_bins"):
            if resume_ckpt["model_kwargs"].get(k) != model_kwargs.get(k):
                raise ValueError(
                    f"resume-state 구조 불일치: {k} ckpt={resume_ckpt['model_kwargs'].get(k)} != "
                    f"현재={model_kwargs.get(k)}. 같은 네트워크 구조로만 이어서 학습 가능.")
        trainer.model.load_state_dict(
            {k: torch.as_tensor(v) for k, v in resume_ckpt["model_state"].items()})
        trainer.actor_opt.load_state_dict(resume_ckpt["actor_opt"])
        trainer.critic_opt.load_state_dict(resume_ckpt["critic_opt"])
        if resume_ckpt["obs_rms"] is not None and trainer.obs_rms is not None:
            trainer.obs_rms.mean = np.asarray(resume_ckpt["obs_rms"]["mean"], dtype=np.float64)
            trainer.obs_rms.var = np.asarray(resume_ckpt["obs_rms"]["var"], dtype=np.float64)
            trainer.obs_rms.count = float(resume_ckpt["obs_rms"]["count"])
        trainer.global_step = int(resume_ckpt["global_step"])
        start_iter = int(resume_ckpt["iteration"]) + 1
        best.update(dict(resume_ckpt.get("best", {})))
        base_metadata["resumed_state"] = str(args.resume_state)
        print(f"[claude_code/PPO] resume-state: {args.resume_state} 복원 "
              f"(iter {resume_ckpt['iteration']} 완료 → iter {start_iter}부터, "
              f"step={trainer.global_step}, pool={len(resume_ckpt['pool'])}개, "
              f"model+optimizer+obs_rms 포함)")

    # phase 이어가기: snapshot 에서 actor+critic 가중치 + obs_rms 를 그대로 불러온다.
    # (snapshot 은 model.state_dict() 전체 = actor+critic 둘 다 포함하므로 value net 도 이어짐.)
    # load_state_dict 는 in-place 라 self-play SelfPlayProvider 의 model 참조에도 즉시 반영되고,
    # 병렬 모드는 학습 시작 시 driver→worker broadcast 로 전파된다. optimizer 모멘트는 새로 시작.
    # (--resume-state 가 있으면 그쪽이 우선이므로 --resume-from 은 무시한다.)
    if args.resume_from and resume_ckpt is None:
        resume_path = Path(args.resume_from)
        state_dict, snap_kwargs, snap_rms = evaluation.load_snapshot(resume_path)
        for k in ("obs_dim", "act_dim", "num_bins"):
            if snap_kwargs.get(k) != model_kwargs.get(k):
                raise ValueError(
                    f"resume 구조 불일치: {k} snapshot={snap_kwargs.get(k)} != "
                    f"현재={model_kwargs.get(k)}. 같은 네트워크 구조로만 이어서 학습 가능.")
        trainer.model.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})
        if snap_rms is not None and trainer.obs_rms is not None:
            trainer.obs_rms.mean = np.asarray(snap_rms["mean"], dtype=np.float64)
            trainer.obs_rms.var = np.asarray(snap_rms["var"], dtype=np.float64)
            trainer.obs_rms.count = float(snap_rms["count"])
        base_metadata["resumed_from"] = str(resume_path)
        print(f"[claude_code/PPO] resume: {resume_path} 에서 actor+critic+obs_rms 로드 "
              f"(이어서 학습; optimizer 모멘트는 새로 시작)")

    # gated self-play: 초기 opponent = '학습 시작 시점(resume 면 phase1 마지막) 정책' 의
    # frozen deep-copy(=iter0). resume 로 weights 가 확정된 뒤 설치해야 하므로 여기서 1회 설치한다.
    # 이후 매 iter EMA 승률이 임계값을 넘으면 on_iteration 에서 현재 정책으로 승격한다.
    # --frozen-opponent 이면 임계값을 무한대로 둬서 영구 고정(승격 안 함).
    gate_threshold = float("inf") if args.frozen_opponent else float(args.selfplay_gate_threshold)
    pool_max = 1 if args.frozen_opponent else max(1, int(args.pool_size))
    if resume_ckpt is not None:
        pool_max = int(resume_ckpt["pool_max"])   # checkpoint 의 pool 구성을 그대로 이어감

    def _pool_weights(emas) -> list:
        """EMA 낮은 후보가 더 자주 뽑히도록 softmax(-ema/τ) 가중치."""
        e = np.asarray(emas, dtype=np.float64)
        if e.size <= 1:
            return [1.0] * int(e.size or 1)
        logits = -e / max(float(args.pool_sample_temp), 1e-6)
        logits -= logits.max()
        w = np.exp(logits)
        return (w / w.sum()).tolist()

    # opponent pool EMA/net 메타데이터(train.py 소유, checkpoint 저장 대상).
    #   각 후보: {"gen", "ema", "state"(actor+critic net np), "rms"(obs_rms 스냅샷)}.
    pool: list = []
    pool_state = {"next_gen": 1, "n_added": 0}

    if args.self_play:
        if resume_ckpt is not None:
            # checkpoint 의 pool(모든 후보 net + EMA) 을 그대로 복원.
            pool = [{"gen": int(e["gen"]), "ema": float(e["ema"]),
                     "state": e["state"], "rms": e.get("rms")}
                    for e in resume_ckpt["pool"]]
            pool_state = {"next_gen": int(resume_ckpt["next_gen"]),
                          "n_added": int(resume_ckpt["n_added"])}
            trainer.set_opponent_pool(pool, _pool_weights([e["ema"] for e in pool]), pool_max)
            print(f"[claude_code/PPO] opponent pool 복원: {len(pool)}개 후보 "
                  f"(gen {[e['gen'] for e in pool]}, ema {[round(e['ema'],3) for e in pool]})")
        else:
            trainer.install_opponent_pool(pool_max)
            snap = trainer.snapshot_current()
            pool = [{"gen": 0, "ema": 0.5, "state": snap["state"], "rms": snap["rms"]}]
            pool_state = {"next_gen": 1, "n_added": 0}
            trainer.pool_set_weights(_pool_weights([0.5]))
            print(f"[claude_code/PPO] opponent pool 초기화 = iter0 정책 1개 (최대 {pool_max}개, "
                  f"EMA α={args.selfplay_ema_alpha}, 추가 임계 min-EMA≥{args.selfplay_gate_threshold:.2f}, "
                  f"초기 EMA=0.5, 샘플 τ={args.pool_sample_temp})")
        base_metadata["selfplay_gate_threshold"] = (None if args.frozen_opponent
                                                    else args.selfplay_gate_threshold)
        base_metadata["selfplay_ema_alpha"] = args.selfplay_ema_alpha
        base_metadata["frozen_opponent"] = bool(args.frozen_opponent)
        base_metadata["pool_size"] = pool_max
        base_metadata["pool_sample_temp"] = args.pool_sample_temp
        if args.frozen_opponent:
            print("[claude_code/PPO] opponent = 학습 시작 시점 정책으로 영구 고정(pool 크기 1, 추가 없음)")

    # iter 0 = 학습 시작 직전 network (resume 면 불러온 가중치, 아니면 완전 초기화).
    evaluation.save_snapshot(_snap_path(0), trainer.model, trainer.obs_rms, model_kwargs)
    print(f"[claude_code/PPO] snapshot 저장: {snapshot_dir} (iter 0 = 초기 network)")

    def _save_best(s: IterationStats, summary: dict):
        obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
        save_bundle(
            trainer.model, bundle_dir, obs_norm=obs_norm,
            extra_metadata={**base_metadata,
                            "selected_iteration": s.iteration,
                            "eval_return": summary["mean_return"],
                            "eval_win_rate": summary["win_rate"]},
        )
        best["return"] = summary["mean_return"]
        best["win_rate"] = summary["win_rate"]
        best["iter"] = s.iteration
        best["saved"] = True

    log_dir = Path(args.artifacts_dir) / "logs" / args.output_name / args.output_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ppo_training_log.csv"
    # resume-state 로 이어갈 때 기존 CSV 에 append(헤더 재기록 안 함).
    csv_append = resume_ckpt is not None and log_path.exists()
    log_file = log_path.open("a" if csv_append else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not csv_append:
        writer.writerow([
            "iteration", "global_step", "mean_return", "mean_length", "completed_episodes",
            "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance",
            "ep_pursuit", "ep_damage", "ep_distance", "ep_aim", "ep_terminal", "elapsed_sec",
            "win", "loss", "draw", "raw_win_rate",
            "ema_mean", "ema_min", "pool_size", "opp_added", "altitude_term",
        ])

    # 마지막으로 완료한 iteration 추적(학습 종료 시 최종 checkpoint 저장에 사용).
    progress = {"last_iter": start_iter - 1, "last_step": trainer.global_step}

    def on_iteration(s: IterationStats):
        pursuit = s.extra.get("pursuit", float("nan"))
        damage = s.extra.get("damage", float("nan"))
        distance = s.extra.get("distance", float("nan"))
        aim = s.extra.get("aim", float("nan"))
        terminal = s.extra.get("terminal", float("nan"))

        # 이번 iter rollout 게임들의 opponent 대비 승패 집계.
        wins = int(s.extra.get("win", 0))
        losses = int(s.extra.get("loss", 0))
        draws = int(s.extra.get("draw", 0))
        decided = wins + losses + draws
        raw_wr = (wins / decided) if decided > 0 else float("nan")
        # 이번 iter 에서 우리 기체 고도 하락으로 종료된 episode 수.
        alt_term = int(s.extra.get("alt_term", 0))

        # opponent 별 EMA 갱신 + 조건부 pool 추가. 우리팀·opponent 모두 stochastic rollout.
        per_opp = s.extra.get("per_opp", {}) or {}
        added = False
        if args.self_play:
            alpha = args.selfplay_ema_alpha
            # 이번 iter 에 실제로 게임이 있었던 후보만 EMA 갱신(안 뽑힌 후보는 유지).
            for i, entry in enumerate(pool):
                d = per_opp.get(i)
                if d and d.get("decided", 0) > 0:
                    entry["ema"] = (1.0 - alpha) * entry["ema"] + alpha * d["raw_win_rate"]
            # 모든 후보 EMA 중 '최소값'이 임계값 이상이면 현재 정책을 새 후보로 추가.
            emas = [e["ema"] for e in pool]
            if emas and min(emas) >= gate_threshold:
                trainer.pool_add_current()
                snap = trainer.snapshot_current()   # 추가된 후보 net(+obs_rms) 을 checkpoint 용으로 보관
                pool.append({"gen": pool_state["next_gen"], "ema": 0.5,
                             "state": snap["state"], "rms": snap["rms"]})
                pool_state["next_gen"] += 1
                if len(pool) > pool_max:
                    pool.pop(0)
                pool_state["n_added"] += 1
                _save_best(s, {"mean_return": s.mean_return, "win_rate": raw_wr})
                added = True
            # 다음 iteration 을 위한 샘플링 가중치 갱신(EMA 낮은 후보 우대).
            trainer.pool_set_weights(_pool_weights([e["ema"] for e in pool]))

        emas = [e["ema"] for e in pool]
        ema_mean = float(np.mean(emas)) if emas else float("nan")
        ema_min = float(np.min(emas)) if emas else float("nan")
        promoted = added   # 하위 print/wandb 호환

        writer.writerow([
            s.iteration, s.global_step, f"{s.mean_return:.4f}", f"{s.mean_length:.1f}",
            s.completed_episodes, f"{s.policy_loss:.5f}", f"{s.value_loss:.5f}",
            f"{s.entropy:.4f}", f"{s.approx_kl:.5f}", f"{s.explained_variance:.4f}",
            f"{pursuit:.4f}", f"{damage:.4f}", f"{distance:.4f}", f"{aim:.4f}",
            f"{terminal:.4f}", f"{s.elapsed_sec:.2f}",
            wins, losses, draws, f"{raw_wr:.4f}",
            f"{ema_mean:.4f}", f"{ema_min:.4f}", len(pool), int(added), alt_term,
        ])
        log_file.flush()

        # 매 iteration 의 actor network 를 snapshot 으로 저장(post-update 상태).
        evaluation.save_snapshot(_snap_path(s.iteration), trainer.model,
                                 trainer.obs_rms, model_kwargs)

        # 전체 학습 상태 checkpoint 저장(= 다음 iteration 시작 직전 상태). 원자적 교체라
        # 저장 도중 중단돼도 직전 checkpoint 가 보존된다. 이 파일로 이어서 학습 가능.
        progress["last_iter"] = s.iteration
        progress["last_step"] = s.global_step
        try:
            save_train_state(
                ckpt_path, trainer=trainer, pool=pool, pool_state=pool_state,
                pool_max=pool_max, iteration=s.iteration, global_step=s.global_step,
                best=best, model_kwargs=model_kwargs)
        except Exception as e:
            print(f"[claude_code/PPO] train-state 저장 실패({e})", flush=True)

        # opponent 에게 준 damage(원값) 평균 = damage_reward / damage_scale (받은damage 가중치 0).
        damage_dealt = (damage / dmg_scale) if (dmg_scale and damage == damage) else float("nan")

        if wb is not None:
            try:
                log_dict = {
                    "iteration": s.iteration,
                    "global_step": s.global_step,
                    "train/mean_return": s.mean_return,
                    "train/mean_length": s.mean_length,
                    "train/completed_episodes": s.completed_episodes,
                    "loss/policy_loss": s.policy_loss,
                    "loss/value_loss": s.value_loss,
                    "loss/entropy": s.entropy,
                    "policy/entropy": s.entropy,          # 현재 정책 entropy
                    "metrics/approx_kl": s.approx_kl,
                    "metrics/explained_variance": s.explained_variance,
                    "selfplay/raw_win_rate": raw_wr,
                    "selfplay/ema_mean": ema_mean,
                    "selfplay/ema_min": ema_min,
                    "selfplay/pool_size": len(pool),
                    "selfplay/n_added": pool_state["n_added"],
                    "selfplay/opp_added": int(added),
                    "selfplay/wins": wins,
                    "selfplay/losses": losses,
                    "selfplay/draws": draws,
                    "train/altitude_termination": alt_term,   # 고도 하락으로 종료된 episode 수
                    "reward/damage_reward": damage,
                    "reward/distance_reward": distance,
                    "reward/termination_reward": terminal,
                    "reward/aim_reward": aim,
                    "damage/dealt_per_episode": damage_dealt,
                }
                # 후보별 EMA (슬롯 0=가장 오래된 후보 … pool_max-1). 빈 슬롯은 로깅 생략.
                for i in range(pool_max):
                    if i < len(pool):
                        log_dict[f"selfplay/pool_ema_slot{i}"] = pool[i]["ema"]
                wb.log(log_dict, step=s.iteration)
            except Exception as e:
                print(f"[claude_code/PPO] wandb.log 실패({e})", flush=True)

        sp_msg = ""
        if args.self_play:
            sp_msg = (f" | pool{len(pool)} W/L/D {wins}/{losses}/{draws} "
                      f"raw_wr {raw_wr:.2f} ema[min {ema_min:.3f} mean {ema_mean:.3f}]"
                      f"{' *POOL+ (added current, best saved)*' if added else ''}")

        print(
            f"iter {s.iteration:3d} | step {s.global_step:7d} | "
            f"return {s.mean_return:8.3f} | len {s.mean_length:6.1f} | "
            f"damage {damage:6.3f} | dist {distance:6.3f} | aim {aim:6.3f} | "
            f"ent {s.entropy:6.3f} | kl {s.approx_kl:.4f} | ev {s.explained_variance:6.3f} | "
            f"altT {alt_term}"
            f"{sp_msg}",
            flush=True,
        )

    try:
        history = trainer.train(on_iteration=on_iteration, start_iteration=start_iter)
    finally:
        log_file.close()
        env.close()
        if eval_env is not None:
            eval_env.close()
        if hasattr(trainer, "close"):
            trainer.close()   # Ray shutdown (병렬 모드)
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass

    obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
    if not best["saved"]:
        # opponent 승격이 한 번도 없었던 경우(EMA 가 임계값에 도달 못함) 최종 정책을 저장.
        save_bundle(trainer.model, bundle_dir, obs_norm=obs_norm, extra_metadata=base_metadata)

    # best 와 별개로, 맨 마지막 iteration 의 파라미터를 항상 '_final' 번들로 저장.
    final_bundle_dir = bundle_dir.parent / f"{bundle_dir.name}_final"
    save_bundle(trainer.model, final_bundle_dir, obs_norm=obs_norm,
                extra_metadata={**base_metadata,
                                "selected_iteration": args.iterations,
                                "source": "final_iteration"})

    # 학습 종료 시 최종 전체 학습 상태 checkpoint 저장(다음 학습에서 --resume-state 로 이어감).
    if args.self_play:
        try:
            save_train_state(
                ckpt_path, trainer=trainer, pool=pool, pool_state=pool_state,
                pool_max=pool_max, iteration=progress["last_iter"],
                global_step=progress["last_step"], best=best, model_kwargs=model_kwargs)
            print(f"[claude_code/PPO] 최종 학습 상태 checkpoint: {ckpt_path} "
                  f"(iter {progress['last_iter']}까지; 다음에 --resume-state {ckpt_path} 로 이어서 학습)")
        except Exception as e:
            print(f"[claude_code/PPO] 최종 train-state 저장 실패({e})", flush=True)

    print(f"\n[claude_code/PPO] 번들 저장 완료: {bundle_dir}")
    print(f"[claude_code/PPO] 최종 iteration 번들: {final_bundle_dir} (iter {args.iterations})")
    if best["saved"]:
        print(f"  - best = 마지막 pool 추가 시점 iteration {best['iter']} "
              f"(추가 당시 raw 승률 {best['win_rate']:.2f}, mean return {best['return']:.3f})")
    else:
        print("  - pool 추가가 없어(min-EMA<임계값) 최종 iteration 정책을 best 로 저장")
    print("  - metadata.json")
    print("  - policy_weights.pkl.gz")

    if history:
        first = next((h.mean_return for h in history if h.mean_return == h.mean_return), None)
        last = next((h.mean_return for h in reversed(history) if h.mean_return == h.mean_return), None)
        if first is not None and last is not None:
            print(f"[claude_code/PPO] 평균 return: {first:.3f} (초기) → {last:.3f} (최종)")


if __name__ == "__main__":
    main()
