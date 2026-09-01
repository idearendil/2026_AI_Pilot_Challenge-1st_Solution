# -*- coding: utf-8 -*-
"""GPU 벡터화 PPO 학습 엔트리 (GpuDogfightVecEnv + PPOGPUTrainer).

예:
  python -m cuda_fdm.train_gpu --nenv 4096 --iters 2000 --rollout 32 \
      --save runs/gpu_ppo.pt --log runs/gpu_ppo.csv

한 iteration = rollout(T) 스텝 × nac(=2·nenv) 에이전트 병렬 수집.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nenv", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10000, help="총 iteration(<=0 이면 무한)")
    ap.add_argument("--rollout", type=int, default=64, help="iteration 당 env step 수 T")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8,
                    help="배치(N=nenv×rollout)를 몇 조각으로 쪼갤지(=gradient step 수/epoch). "
                         "미니배치 크기가 아님. 클수록 mb 작아지고 step 많아짐")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--no-norm-obs", action="store_true")
    ap.add_argument("--num-bins", type=int, default=21, help="채널별 discrete 행동 격자 수(원본 train.py 기본=21)")
    # ── iteration 스케줄 (sched-period iter 마다 단계 상승) ──
    ap.add_argument("--sched-period", type=int, default=2000,
                    help="이 iter 수마다 단계 k↑: lr·ent-coef ×= 각 decay, rollout += increment (0이면 비활성)")
    ap.add_argument("--sched-lr-decay", type=float, default=1.0 / 3.0, help="단계마다 lr 에 곱할 계수")
    ap.add_argument("--sched-ent-decay", type=float, default=1.0 / 3.0, help="단계마다 ent-coef 에 곱할 계수")
    ap.add_argument("--sched-rollout-increment", type=int, default=8, help="단계마다 rollout 에 더할 값")
    # ── opponent pool / gated self-play (원본과 동일 규약) ──
    ap.add_argument("--pool-evict-cap", type=int, default=4,
                    help="evictable(net) opponent snapshot 최대 수")
    ap.add_argument("--selfplay-gate-threshold", type=float, default=0.6,
                    help="evictable 최소 승률 EMA ≥ 이 값이면 현재 main 을 snapshot 추가")
    ap.add_argument("--selfplay-ema-alpha", type=float, default=0.1, help="승률 EMA 갱신율")
    ap.add_argument("--pool-sample-temp", type=float, default=0.3, help="opponent softmax 온도 τ")
    ap.add_argument("--pool-uniform-floor", type=float, default=0.5, help="샘플 균등 분배 비율 f")
    ap.add_argument("--milestone-period", type=int, default=500,
                    help="permanent snapshot(+capacity) + exploiter 학습 주기(iter, 0이면 비활성)")
    ap.add_argument("--no-opp-sample", action="store_true",
                    help="opponent 행동을 deterministic(argmax)으로")
    # ── exploiter ──
    ap.add_argument("--exploiter-iters", type=int, default=1000,
                    help="exploiter 1회 학습 최대 iter(0 이하면 비활성)")
    ap.add_argument("--exploiter-win-target", type=float, default=0.7)
    ap.add_argument("--exploiter-lr", type=float, default=1e-4)
    ap.add_argument("--exploiter-ent-coef", type=float, default=0.0001)
    ap.add_argument("--exploiter-clip-coef", type=float, default=0.2)
    ap.add_argument("--exploiter-init-iteration-first", type=int, default=500,
                    help="first iter 의 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--exploiter-init-iteration-rest", type=int, default=1000,
                    help="그 외 모든 exploiter 를 이 iter 시점 main net 으로 초기화")
    ap.add_argument("--exploiter-alt-hunt-coef", type=float, default=5.0,
                    help="exploiter 상대고도 log 사냥 보상 계수 C: C*(ln(상대 이전고도)-ln(상대 현재고도))")
    ap.add_argument("--no-critic-opp-actions", action="store_true",
                    help="critic 에 상대 과거 5-step action(20dim) 추가 입력을 주지 않음(기본은 줌)")
    ap.add_argument("--no-aux-pred", action="store_true",
                    help="미래위치 aux 예측(actor·critic head)을 끔(기본은 켬)")
    ap.add_argument("--aux-coef", type=float, default=0.1,
                    help="aux 미래위치 예측 MSE 를 actor/critic loss 에 더할 계수")
    ap.add_argument("--substeps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default=None, help="체크포인트 경로(.pt)")
    ap.add_argument("--save-every", type=int, default=100, help="N iteration 마다 저장")
    ap.add_argument("--log", type=str, default=None, help="iteration 통계 CSV 경로")
    ap.add_argument("--resume", type=str, default=None, help="이어서 학습할 .pt")
    # ── wandb (기본 켜짐; 네트워크/키 실패해도 학습은 계속) ──
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                    help="wandb 로깅 사용 (기본 켜짐)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", help="wandb 로깅 끄기")
    ap.add_argument("--wandb-project", default="AIP contest", help="wandb 프로젝트명")
    ap.add_argument("--wandb-run-name", default="", help="wandb run 이름 (비우면 gpu-ppo/seed)")
    args = ap.parse_args()

    torch.zeros(1, device=args.device)   # CUDA 워밍업
    env = GpuDogfightVecEnv(args.nenv, substeps=args.substeps, seed=args.seed,
                            device=args.device)
    cfg = PPOGPUConfig(
        total_iterations=args.iters, rollout_steps=args.rollout,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_coef=args.clip,
        update_epochs=args.epochs, num_minibatches=args.minibatches,
        lr=args.lr, critic_lr=args.critic_lr, ent_coef=args.ent_coef,
        target_kl=args.target_kl, normalize_obs=not args.no_norm_obs, num_bins=args.num_bins,
        sched_period=args.sched_period, sched_lr_decay=args.sched_lr_decay,
        sched_ent_decay=args.sched_ent_decay, sched_rollout_increment=args.sched_rollout_increment,
        pool_evict_cap=args.pool_evict_cap, selfplay_gate_threshold=args.selfplay_gate_threshold,
        selfplay_ema_alpha=args.selfplay_ema_alpha, pool_sample_temp=args.pool_sample_temp,
        pool_uniform_floor=args.pool_uniform_floor, milestone_period=args.milestone_period,
        opp_sample=not args.no_opp_sample,
        exploiter_iters=args.exploiter_iters, exploiter_win_target=args.exploiter_win_target,
        exploiter_lr=args.exploiter_lr, exploiter_ent_coef=args.exploiter_ent_coef,
        exploiter_clip_coef=args.exploiter_clip_coef,
        exploiter_init_iteration_first=args.exploiter_init_iteration_first,
        exploiter_init_iteration_rest=args.exploiter_init_iteration_rest,
        exploiter_alt_hunt_coef=args.exploiter_alt_hunt_coef,
        critic_opp_actions=not args.no_critic_opp_actions,
        aux_pred=not args.no_aux_pred, aux_coef=args.aux_coef,
        seed=args.seed, device=args.device)
    trainer = PPOGPUTrainer(env, cfg)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    start_it = 1
    if args.resume:
        ckpt = trainer.load(args.resume)
        start_it = ckpt.get("iteration", 1) + 1
        print(f"[resume] {args.resume} 에서 iter {start_it} 부터 재개", flush=True)

    # ── wandb 초기화 (실패해도 학습 계속) ─────────────────────────────────────
    # 하드코딩 API 키는 새 파일에 복제하지 않고 claude_code/train.py 것을 재사용한다
    # (그 파일이 유일한 키 보관처이자 '공개 repo 커밋 금지' 대상). 환경변수가 있으면 우선.
    wb = None
    if args.wandb:
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY"):
                try:
                    from claude_code.train import _WANDB_API_KEY
                    os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
                except Exception:
                    pass   # 키 없으면 wandb 로그인/오프라인 설정에 위임
            run_name = args.wandb_run_name or f"gpu-ppo/seed{args.seed}"
            # run id: '--save 파일명 stem' 같은 고정 id 는 서버에서 삭제된 run 과 충돌한다
            # (resume 으로 삭제된 id 재사용 금지). 대신 매 새 run 마다 unique id 를 생성하고
            # 체크포인트 옆 사이드카(<save>.wandbid)에 저장 → resume 시 읽어 같은 run 에 이어붙인다.
            id_path = Path(args.save).with_suffix(".wandbid") if args.save else None
            run_id = None
            if args.resume and id_path is not None and id_path.exists():
                run_id = (id_path.read_text(encoding="utf-8").strip() or None)
            if run_id is None:
                run_id = wandb.util.generate_id()
                if id_path is not None:
                    id_path.write_text(run_id, encoding="utf-8")
            wb = wandb.init(project=args.wandb_project, name=run_name, id=run_id,
                            resume="allow", config=vars(args))
            print(f"[gpu-ppo] wandb 활성: project='{args.wandb_project}' run='{run_name}' id={run_id}", flush=True)
        except Exception as e:
            print(f"[gpu-ppo] wandb 초기화 실패({e}) → wandb 없이 진행", flush=True)
            wb = None

    # ── wandb x축 정의 ──────────────────────────────────────────────────────────
    # main 지표는 전부 x축=iteration. exploiter 는 milestone 마다 별도 섹터(exploiter@500 …)
    # 를 만들고 각 섹터의 x축을 그 섹터 자체의 exp_iter(=exploiter 내부 iteration)로 둔다.
    # 이렇게 하면 explicit step 을 안 넘겨도(내부 _step 은 매 log 호출마다 단조 증가) 패널
    # x축이 섞이지 않는다 — milestone 에서 exploiter 가 수백 iter 돌아도 main step 과 충돌 없음.
    _exp_sections = set()
    if wb is not None:
        try:
            wb.define_metric("iteration")
            for _pre in ("charts", "losses", "pool", "pool_ema", "perf", "hparams"):
                wb.define_metric(f"{_pre}/*", step_metric="iteration")
            wb.define_metric("global_step", step_metric="iteration")
        except Exception as e:
            print(f"[gpu-ppo] wandb define_metric 실패({e})", flush=True)

    log_f = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "a", encoding="utf-8")
        if Path(args.log).stat().st_size == 0:
            log_f.write("iter,gstep,mean_ret,mean_len,eps,win_rate,pl,vl,ent,kl,clipfrac,"
                        "ev,sps,elapsed,pool_size,pool_perm,ema_min,ema_mean\n")

    def on_iter(s):
        wr = s.win_rate if s.win_rate == s.win_rate else float("nan")   # nan-safe
        msg = (f"it {s.iteration:5d} | ret {s.mean_return:8.3f} len {s.mean_length:6.1f} "
               f"eps {int(s.completed_episodes):5d} wr {wr:.3f} | pl {s.policy_loss:+.4f} "
               f"vl {s.value_loss:.3f} ent {s.entropy:.3f} kl {s.approx_kl:.4f} cf {s.clipfrac:.3f} "
               f"ev {s.explained_variance:+.3f} aux {s.extra.get('aux_loss', 0.0):.4f} | pool {s.extra['pool_size']}"
               f"(p{s.extra['pool_perm']}) emin {s.extra['ema_min']:.3f} "
               f"| {s.steps_per_sec/1e6:.2f}M sps {s.elapsed_sec*1e3:.0f}ms")
        if s.extra.get("early_stop"):
            msg += f" [kl-stop @ep{s.extra['epochs']}]"
        if s.extra.get("pool_event"):
            msg += f" [{s.extra['pool_event']}]"
        print(msg, flush=True)
        if log_f is not None:
            log_f.write(f"{s.iteration},{s.global_step},{s.mean_return},{s.mean_length},"
                        f"{int(s.completed_episodes)},{s.win_rate},{s.policy_loss},{s.value_loss},"
                        f"{s.entropy},{s.approx_kl},{s.clipfrac},{s.explained_variance},"
                        f"{s.steps_per_sec},{s.elapsed_sec},{s.extra['pool_size']},"
                        f"{s.extra['pool_perm']},{s.extra['ema_min']},{s.extra['ema_mean']}\n")
            log_f.flush()
        if wb is not None:
            try:
                logd = {
                    "charts/mean_return": s.mean_return, "charts/mean_length": s.mean_length,
                    "charts/win_rate": s.win_rate, "charts/completed_episodes": int(s.completed_episodes),
                    "charts/alt_loss_rate": s.extra["alt_loss_rate"],
                    "losses/policy_loss": s.policy_loss, "losses/value_loss": s.value_loss,
                    "losses/entropy": s.entropy, "losses/approx_kl": s.approx_kl,
                    "losses/clipfrac": s.clipfrac, "losses/explained_variance": s.explained_variance,
                    "losses/aux_pred_mse": s.extra.get("aux_loss", 0.0),
                    "pool/size": s.extra["pool_size"], "pool/permanent": s.extra["pool_perm"],
                    "pool/ema_min": s.extra["ema_min"], "pool/ema_mean": s.extra["ema_mean"],
                    "perf/steps_per_sec": s.steps_per_sec, "perf/elapsed_sec": s.elapsed_sec,
                    "hparams/lr": s.extra["lr"], "hparams/ent_coef": s.extra["ent_coef"],
                    "hparams/rollout": s.extra["rollout"],
                    "global_step": s.global_step,
                }
                # opponent 별 승률 EMA. evict 슬롯은 슬롯 위치(FIFO) 기준으로 로깅해 opponent 가
                # 교체돼도 그래프 수가 안 늘어난다. permanent 는 추가 순 고정 identity 로 로깅.
                for i, ema in enumerate(s.extra.get("evict_slot_emas", [])):
                    logd[f"pool_ema/evict_slot{i}"] = ema
                for i, ema in enumerate(s.extra.get("perm_slot_emas", [])):
                    logd[f"pool_ema/perm_slot{i}"] = ema
                logd["iteration"] = s.iteration   # x축(step= 대신 step_metric 사용)
                wb.log(logd)
            except Exception as e:
                print(f"[gpu-ppo] wandb.log 실패({e})", flush=True)
        if args.save and s.iteration % args.save_every == 0:
            trainer.save(args.save)

    def on_exploiter(ms_it, i, m):
        """milestone(ms_it)의 exploiter 학습 iteration(i) 지표를 exploiter@{ms_it} 섹터에 로깅."""
        if wb is None:
            return
        sec = f"exploiter@{ms_it}"
        try:
            if sec not in _exp_sections:
                wb.define_metric(f"{sec}/exp_iter")
                wb.define_metric(f"{sec}/*", step_metric=f"{sec}/exp_iter")
                _exp_sections.add(sec)
            d = {f"{sec}/{k}": v for k, v in m.items()}
            d[f"{sec}/exp_iter"] = i
            wb.log(d)
        except Exception as e:
            print(f"[gpu-ppo] wandb exploiter.log 실패({e})", flush=True)

    t0 = time.time()
    try:
        trainer.train(on_iteration=on_iter, start_iteration=start_it,
                      on_exploiter_iter=on_exploiter)
    except KeyboardInterrupt:
        print("\n[중단] 학습 중지, 체크포인트 저장 중...", flush=True)
    finally:
        if args.save:
            trainer.save(args.save)
            print(f"[save] {args.save}", flush=True)
        if log_f is not None:
            log_f.close()
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass
    print(f"총 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
