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


def _deterministic_eval(model, obs_rms, eval_env, n_episodes: int, device: str,
                        reconstruct: bool = False) -> tuple[float, float]:
    """현재 정책을 탐험 없이(평균 action) 굴려 평균 return / 길이를 잰다.

    환경 초기 상태가 결정적이므로 적은 episode 로도 정책 품질을 대표한다. action 은
    raw([-1,1]^4) 로 env.step 에 전달하고 env 가 throttle 변환(_to_sim_action)을 수행해
    학습 rollout 과 동일한 경로를 쓴다.
    """
    def _norm(o):
        if obs_rms is None:
            return np.asarray(o, dtype=np.float32)
        n = (np.asarray(o, dtype=np.float64) - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        return np.clip(n, -10.0, 10.0).astype(np.float32)

    if reconstruct:
        from claude_code.my_observation import reset_reconstructor, advance_reconstructor
    returns, lengths = [], []
    for ep in range(n_episodes):
        if reconstruct:
            reset_reconstructor()
        o, _ = eval_env.reset(seed=100000 + ep)
        if reconstruct:
            reset_reconstructor()
        done, ret, steps = False, 0.0, 0
        while not done:
            with torch.no_grad():
                a = model.act_deterministic(
                    torch.as_tensor(_norm(o), dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze(0).cpu().numpy().astype(np.float32)
            o, r, term, trunc, _ = eval_env.step(a)
            if reconstruct:
                advance_reconstructor(eval_env._ownship_state, eval_env._target_state)
            ret += float(r)
            steps += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(steps)
    return float(np.mean(returns)), float(np.mean(lengths))


def parse_args():
    p = argparse.ArgumentParser(description="claude_code standalone PPO trainer for DogFight 1v1")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--hidden", default="256,256", help="콤마 구분 hidden 크기, 예: 256,256")
    p.add_argument("--activation", default="tanh", choices=["tanh", "relu", "elu"])
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--no-normalize-obs", action="store_true", help="관측 정규화 끄기")
    p.add_argument("--no-scale-reward", action="store_true", help="보상 스케일링 끄기")
    p.add_argument("--no-anneal-lr", action="store_true", help="학습률 감쇠 끄기")
    # 기본값으로 claude_code 의 my_reward / my_observation 을 사용한다.
    # 프레임워크 기본 보상/관측을 쓰려면 빈 문자열을 넘긴다: --reward-module "" --observation-module ""
    p.add_argument("--reward-module", default="claude_code.my_reward",
                   help="보상 모듈 경로 (기본: claude_code.my_reward). 빈 값이면 프레임워크 기본 보상")
    p.add_argument("--observation-module", default="claude_code.my_observation",
                   help="관측 모듈 경로 (기본: claude_code.my_observation). 빈 값이면 tactical16")
    p.add_argument("--eval-interval", type=int, default=5,
                   help="결정론적 평가 주기(iter). 최고 성능 정책을 번들로 저장.")
    p.add_argument("--eval-episodes", type=int, default=2)
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

    env_kwargs = dict(
        overrides={"target_mode": args.target_mode},
        reward_module=args.reward_module,
        observation_module=args.observation_module,
    )
    env = make_env(**env_kwargs)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
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
        normalize_obs=not args.no_normalize_obs,
        scale_reward=not args.no_scale_reward,
        anneal_lr=not args.no_anneal_lr,
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
            print("[claude_code/PPO] 상대 = SELF-PLAY (worker 별 같은 actor network)")
    else:
        trainer = PPOTrainer(env, cfg)
        if args.self_play:
            from claude_code.self_play import SelfPlayProvider
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            env._target_action_provider = SelfPlayProvider(
                trainer.model, trainer.obs_rms, env._observation_fn,
                env._observation_mode, sr, cfg.device)
            print("[claude_code/PPO] 상대 = SELF-PLAY (학습 중인 같은 actor network)")
        else:
            print(f"[claude_code/PPO] 상대 = 스크립트 target_mode={args.target_mode}")

    # 결정론적 평가용 별도 환경 (driver 에서 단일 실행).
    eval_env = make_env(
        overrides={"target_mode": args.target_mode},
        reward_module=args.reward_module,
        observation_module=args.observation_module,
        runner_index="eval",
    )
    if args.self_play:
        from claude_code.self_play import SelfPlayProvider
        sr = int(STANDARD_ENV_CONFIG["step_ratio"])
        eval_env._target_action_provider = SelfPlayProvider(
            trainer.model, trainer.obs_rms, eval_env._observation_fn,
            eval_env._observation_mode, sr, cfg.device)

    bundle_dir = Path(args.artifacts_dir) / "models" / args.output_name / args.output_tag
    base_metadata = {
        "output_name": args.output_name,
        "output_tag": args.output_tag,
        "target_mode": args.target_mode,
        "self_play": bool(args.self_play),
        "train_iterations": args.iterations,
        # 추론(submission/evaluate)이 동일 관측을 만들기 위해 모듈 경로를 기록.
        "reward_module": args.reward_module,
        "observation_module": args.observation_module,
        "observation_mode": meta_obs_mode,
        "num_workers": args.num_workers,
        "env_config": {k: STANDARD_ENV_CONFIG[k] for k in
                       ("step_ratio", "max_engage_time", "episode_step_limit")},
    }
    best = {"return": -float("inf"), "iter": -1, "saved": False}

    def _save_best(s: IterationStats, eval_return: float):
        obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
        save_bundle(
            trainer.model, bundle_dir, obs_norm=obs_norm,
            extra_metadata={**base_metadata,
                            "selected_iteration": s.iteration,
                            "eval_return": eval_return},
        )
        best["return"] = eval_return
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
        "ep_pursuit", "ep_damage", "ep_terminal", "elapsed_sec",
    ])

    def on_iteration(s: IterationStats):
        pursuit = s.extra.get("pursuit", float("nan"))
        damage = s.extra.get("damage", float("nan"))
        terminal = s.extra.get("terminal", float("nan"))
        writer.writerow([
            s.iteration, s.global_step, f"{s.mean_return:.4f}", f"{s.mean_length:.1f}",
            s.completed_episodes, f"{s.policy_loss:.5f}", f"{s.value_loss:.5f}",
            f"{s.entropy:.4f}", f"{s.approx_kl:.5f}", f"{s.explained_variance:.4f}",
            f"{pursuit:.4f}", f"{damage:.4f}", f"{terminal:.4f}", f"{s.elapsed_sec:.2f}",
        ])
        log_file.flush()

        eval_msg = ""
        is_last = s.iteration == args.iterations
        if args.eval_interval > 0 and (s.iteration % args.eval_interval == 0 or is_last):
            eval_ret, eval_len = _deterministic_eval(
                trainer.model, trainer.obs_rms, eval_env, args.eval_episodes, cfg.device,
                reconstruct=cfg.reconstruct_state,
            )
            improved = eval_ret > best["return"]
            if improved:
                _save_best(s, eval_ret)
            eval_msg = (f" | EVAL ret {eval_ret:7.3f} len {eval_len:5.1f}"
                        f"{' *BEST(saved)*' if improved else ''}")

        print(
            f"iter {s.iteration:3d} | step {s.global_step:7d} | "
            f"return {s.mean_return:8.3f} | len {s.mean_length:6.1f} | "
            f"pursuit {pursuit:6.3f} | damage {damage:6.3f} | "
            f"ent {s.entropy:6.3f} | kl {s.approx_kl:.4f} | ev {s.explained_variance:6.3f}"
            f"{eval_msg}",
            flush=True,
        )

    try:
        history = trainer.train(on_iteration=on_iteration)
    finally:
        log_file.close()
        env.close()
        eval_env.close()
        if hasattr(trainer, "close"):
            trainer.close()   # Ray shutdown (병렬 모드)

    if not best["saved"]:
        # 평가가 한 번도 수행되지 않은 경우(eval_interval<=0) 최종 정책 저장.
        obs_norm = trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None
        save_bundle(trainer.model, bundle_dir, obs_norm=obs_norm, extra_metadata=base_metadata)

    print(f"\n[claude_code/PPO] 번들 저장 완료: {bundle_dir}")
    print(f"  - 선택된 iteration: {best['iter']} (deterministic eval return {best['return']:.3f})")
    print("  - metadata.json")
    print("  - policy_weights.pkl.gz")

    if history:
        first = next((h.mean_return for h in history if h.mean_return == h.mean_return), None)
        last = next((h.mean_return for h in reversed(history) if h.mean_return == h.mean_return), None)
        if first is not None and last is not None:
            print(f"[claude_code/PPO] 평균 return: {first:.3f} (초기) → {last:.3f} (최종)")


if __name__ == "__main__":
    main()
