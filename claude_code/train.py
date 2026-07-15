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
import sys
from pathlib import Path

import numpy as np
import torch

# Release 루트/ src import 경로 등록 (단독 실행 대비).
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
from claude_code.model import save_bundle
from claude_code.parallel import physical_cpu_count
from claude_code.ppo import PPOConfig, PPOTrainer, IterationStats


def parse_args():
    p = argparse.ArgumentParser(description="claude_code standalone PPO trainer for DogFight 1v1")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--rollout-steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.0001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--hidden", default="512,512,512", help="actor hidden 크기, 예: 256,256")
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--log-std-init", type=float, default=-1.0, help="(이산 정책에서는 미사용)")
    p.add_argument("--action-bins", type=int, default=7,
                   help="각 행동 채널(roll/pitch/yaw/throttle)의 이산 카테고리 수 (균등 분할)")
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
    p.add_argument("--aim-reward-scale", type=float, default=0.5,
                   help="조준 dense shaping(potential-based) 계수. damage 보다 작게(보조). "
                        "예 0.5. 0 이면 끔. None 이면 모듈 기본값(0.0=off) 사용.")
    p.add_argument("--resume-from", default="",
                   help="이어서 학습할 snapshot(.pt) 경로. actor+critic 가중치+obs_rms 를 불러와 "
                        "그 상태에서 학습 시작(phase1 → phase2). optimizer 모멘트는 새로 시작.")
    p.add_argument("--frozen-opponent", action="store_true",
                   help="self-play 상대를 학습 시작 시점의 actor net 으로 고정(학습 agent 와 분리). "
                        "phase2(--resume-from)와 함께 쓰면 상대=phase1 마지막 net 으로 고정.")
    p.add_argument("--eval-interval", type=int, default=5,
                   help="평가 주기(iter). 현재 정책 vs eval-interval iter 전 정책. 최고 성능 정책을 번들로 저장.")
    p.add_argument("--eval-games", type=int, default=20,
                   help="평가 1회당 stochastic 대결 판 수 (멀티프로세스 분배)")
    p.add_argument("--eval-episodes", type=int, default=2, help="(미사용; 호환용)")
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
    # Ray 병렬 데이터 수집. 기본 worker 수 = 물리 CPU 코어 수(논리 아님). 1 이면 단일 프로세스.
    p.add_argument("--num-workers", type=int, default=physical_cpu_count(),
                   help="Ray rollout worker 수 (기본=물리 코어 수). 1 이면 Ray 미사용")
    p.add_argument("--device", default="cpu",
                   help="driver update 디바이스 (큰 모델은 cuda). worker 는 항상 CPU 추론")
    p.add_argument("--output-name", default="team01")
    p.add_argument("--output-tag", default="ppo_mlp_v1")
    p.add_argument("--artifacts-dir", default="artifacts")
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
        if args.self_play:
            print("[claude_code/PPO] 상대 = SELF-PLAY (worker 별 같은 actor network, stochastic)")
    else:
        trainer = PPOTrainer(env, cfg)
        if args.self_play:
            from claude_code.self_play import SelfPlayProvider
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            env._target_action_provider = SelfPlayProvider(
                trainer.model, trainer.obs_rms, env._observation_fn,
                env._observation_mode, sr, cfg.device, explore=True)
            print("[claude_code/PPO] 상대 = SELF-PLAY (학습 중인 같은 actor network, stochastic)")
        else:
            print(f"[claude_code/PPO] 상대 = 스크립트 target_mode={args.target_mode}")

    # 평가 환경: 병렬 모드는 worker 가 자체 env 로 평가하므로 driver eval_env 불필요.
    # 단일 프로세스 모드에서만 별도 평가 env 를 만든다(rollout env 와 분리).
    eval_env = None
    if not parallel:
        eval_env = make_env(
            overrides={"target_mode": args.target_mode},
            reward_module=args.reward_module,
            observation_module=args.observation_module,
            reward_overrides=reward_overrides,
            runner_index="eval",
        )

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

    # 매 iter actor network snapshot 저장 디렉토리 (평가 상대 + 재현용).
    from claude_code import evaluation
    snapshot_dir = Path("claude_code") / "models" / args.output_name / args.output_tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _snap_path(it: int) -> Path:
        return snapshot_dir / f"iter_{it:04d}.pt"

    # phase 이어가기: snapshot 에서 actor+critic 가중치 + obs_rms 를 그대로 불러온다.
    # (snapshot 은 model.state_dict() 전체 = actor+critic 둘 다 포함하므로 value net 도 이어짐.)
    # load_state_dict 는 in-place 라 self-play SelfPlayProvider 의 model 참조에도 즉시 반영되고,
    # 병렬 모드는 학습 시작 시 driver→worker broadcast 로 전파된다. optimizer 모멘트는 새로 시작.
    if args.resume_from:
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

    # self-play 상대를 '학습 시작 시점(resume 면 phase1 마지막) actor net' 으로 고정.
    # resume 직후(초기 weights 확정) 캡처해야 하므로 iter0 snapshot 직전에 설정한다.
    if args.frozen_opponent and args.self_play:
        if parallel:
            rms = trainer.obs_rms
            trainer.set_frozen_opponent(
                {k: v.detach().cpu().numpy() for k, v in trainer.model.state_dict().items()},
                rms.mean if rms is not None else None,
                rms.var if rms is not None else None,
                rms.count if rms is not None else 0.0)
        else:
            import copy
            from claude_code.self_play import SelfPlayProvider
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            frozen_model = copy.deepcopy(trainer.model).eval()
            frozen_rms = copy.deepcopy(trainer.obs_rms) if trainer.obs_rms is not None else None
            env._target_action_provider = SelfPlayProvider(
                frozen_model, frozen_rms, env._observation_fn,
                env._observation_mode, sr, cfg.device, explore=True)
        base_metadata["frozen_opponent"] = True
        print("[claude_code/PPO] self-play 상대 = 학습 시작 시점 actor net 으로 고정(frozen)")

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
    log_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow([
        "iteration", "global_step", "mean_return", "mean_length", "completed_episodes",
        "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance",
        "ep_pursuit", "ep_damage", "ep_distance", "ep_aim", "ep_terminal", "elapsed_sec",
    ])

    def on_iteration(s: IterationStats):
        pursuit = s.extra.get("pursuit", float("nan"))
        damage = s.extra.get("damage", float("nan"))
        distance = s.extra.get("distance", float("nan"))
        aim = s.extra.get("aim", float("nan"))
        terminal = s.extra.get("terminal", float("nan"))
        writer.writerow([
            s.iteration, s.global_step, f"{s.mean_return:.4f}", f"{s.mean_length:.1f}",
            s.completed_episodes, f"{s.policy_loss:.5f}", f"{s.value_loss:.5f}",
            f"{s.entropy:.4f}", f"{s.approx_kl:.5f}", f"{s.explained_variance:.4f}",
            f"{pursuit:.4f}", f"{damage:.4f}", f"{distance:.4f}", f"{aim:.4f}",
            f"{terminal:.4f}", f"{s.elapsed_sec:.2f}",
        ])
        log_file.flush()

        # 매 iteration 의 actor network 를 snapshot 으로 저장(post-update 상태).
        evaluation.save_snapshot(_snap_path(s.iteration), trainer.model,
                                 trainer.obs_rms, model_kwargs)

        eval_msg = ""
        is_last = s.iteration == args.iterations
        if args.eval_interval > 0 and (s.iteration % args.eval_interval == 0 or is_last):
            # 상대 = eval_interval iter 전의 self (없으면 iter 0).
            opp_iter = max(0, s.iteration - args.eval_interval)
            opp_path = _snap_path(opp_iter)
            if opp_path.exists():
                opp_state, opp_kwargs, opp_rms = evaluation.load_snapshot(opp_path)
                base_seed = 500000 + s.iteration * 1000
                if parallel:
                    summary = trainer.evaluate_vs(
                        opp_state, opp_kwargs, opp_rms,
                        args.eval_games, True, base_seed)
                else:
                    summary = trainer.evaluate_vs(
                        eval_env, opp_state, opp_kwargs, opp_rms,
                        args.eval_games, True, base_seed)
                eval_ret = summary["mean_return"]
                # 최고 승률 여부와 무관하게, 5 iter 전의 self 를 상대로 승률이
                # 0.5 를 넘으면 best 모델로 저장(덮어쓰기).
                saved = summary["win_rate"] > 0.5
                if saved:
                    _save_best(s, summary)
                eval_msg = (
                    f" | EVAL vs iter{opp_iter} ret {eval_ret:7.3f} "
                    f"W/L/D {summary['win']}/{summary['loss']}/{summary['draw']} "
                    f"(wr {summary['win_rate']:.2f}, n={summary['n']})"
                    f"{' *SAVED(wr>0.5)*' if saved else ''}")

        print(
            f"iter {s.iteration:3d} | step {s.global_step:7d} | "
            f"return {s.mean_return:8.3f} | len {s.mean_length:6.1f} | "
            f"damage {damage:6.3f} | dist {distance:6.3f} | aim {aim:6.3f} | "
            f"ent {s.entropy:6.3f} | kl {s.approx_kl:.4f} | ev {s.explained_variance:6.3f}"
            f"{eval_msg}",
            flush=True,
        )

    try:
        history = trainer.train(on_iteration=on_iteration)
    finally:
        log_file.close()
        env.close()
        if eval_env is not None:
            eval_env.close()
        if hasattr(trainer, "close"):
            trainer.close()   # Ray shutdown (병렬 모드)

    obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
    if not best["saved"]:
        # 평가에서 승률 0.5 초과가 한 번도 없었던(또는 eval_interval<=0) 경우 최종 정책 저장.
        save_bundle(trainer.model, bundle_dir, obs_norm=obs_norm, extra_metadata=base_metadata)

    # best 와 별개로, 맨 마지막 iteration 의 파라미터를 항상 '_final' 번들로 저장.
    final_bundle_dir = bundle_dir.parent / f"{bundle_dir.name}_final"
    save_bundle(trainer.model, final_bundle_dir, obs_norm=obs_norm,
                extra_metadata={**base_metadata,
                                "selected_iteration": args.iterations,
                                "source": "final_iteration"})

    print(f"\n[claude_code/PPO] 번들 저장 완료: {bundle_dir}")
    print(f"[claude_code/PPO] 최종 iteration 번들: {final_bundle_dir} (iter {args.iterations})")
    if best["saved"]:
        print(f"  - 선택된 iteration: {best['iter']} "
              f"(past-self 승률 {best['win_rate']:.2f} > 0.5, mean return {best['return']:.3f})")
    else:
        print("  - 평가에서 승률 0.5 초과가 없어 최종 iteration 정책을 저장")
    print("  - metadata.json")
    print("  - policy_weights.pkl.gz")

    if history:
        first = next((h.mean_return for h in history if h.mean_return == h.mean_return), None)
        last = next((h.mean_return for h in reversed(history) if h.mean_return == h.mean_return), None)
        if first is not None and last is not None:
            print(f"[claude_code/PPO] 평균 return: {first:.3f} (초기) → {last:.3f} (최종)")


if __name__ == "__main__":
    main()
