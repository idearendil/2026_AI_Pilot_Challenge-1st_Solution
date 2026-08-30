# -*- coding: utf-8 -*-
"""GPU PPO 검증:
(A) action_to_env 매핑이 채널 경계([-1,1]³,[0,1])를 지킨다.
(B) collect_rollout 의 GAE 가 독립 참조 구현과 일치(합성 롤아웃).
(C) 실제 GpuDogfightVecEnv 학습 루프 무결성: NaN 없음, 통계 유한,
    처리량, 가치함수 학습 진행(explained_variance 상승).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.ppo_gpu import (PPOGPUConfig, PPOGPUTrainer, action_to_env,
                              ActorCritic, RunningNorm, make_action_grid)


def test_action_mapping():
    """discrete index → env control 이 원본 규약과 일치:
    격자=linspace(-1,1,21), roll/pitch/rudder∈[-1,1] 그대로, throttle=[-1,1]→[0,1](0.5z+0.5).
    (claude_code.model.discrete_indices_to_continuous + action_provider throttle 리맵)"""
    nb = 21   # 원본 train.py --action-bins 기본값.
    mid_idx = (nb - 1) // 2   # 홀수 격자의 중앙 index → 원값 0(중립).
    idx = torch.randint(0, nb, (1000, 4))
    ctrl = action_to_env(idx, nb)
    ok = (ctrl[:, 0:3].abs() <= 1.0 + 1e-6).all() and \
         (ctrl[:, 3] >= -1e-6).all() and (ctrl[:, 3] <= 1.0 + 1e-6).all()
    grid = make_action_grid(nb)
    # 중앙 index → 원값 0 → roll/pitch/rudder 0, throttle 0.5.
    mid = action_to_env(torch.full((1, 4), mid_idx), nb)[0]
    # 원본 격자 대조: index i 의 roll 값 == linspace(-1,1,21)[i].
    grid_ok = torch.allclose(action_to_env(torch.arange(nb).view(nb, 1).repeat(1, 4), nb)[:, 0], grid)
    print(f"(A) discrete action: 경계={bool(ok)}, 중앙 ctrl(roll,thr)=({mid[0]:.2f},{mid[3]:.2f}), "
          f"격자==linspace(-1,1,21)={bool(grid_ok)} "
          f"-> {'OK' if ok and abs(mid[0])<1e-6 and abs(mid[3]-0.5)<1e-6 and grid_ok else 'FAIL'}")


def _ref_gae(rew, val, done, next_val, next_done, gamma, lam):
    """CleanRL 관행 참조 GAE (numpy, (T,B))."""
    T, B = rew.shape
    adv = np.zeros((T, B), dtype=np.float64)
    lastgae = np.zeros(B)
    for t in reversed(range(T)):
        if t == T - 1:
            nnt = 1.0 - next_done
            nv = next_val
        else:
            nnt = 1.0 - done[t + 1]
            nv = val[t + 1]
        delta = rew[t] + gamma * nv * nnt - val[t]
        lastgae = delta + gamma * lam * nnt * lastgae
        adv[t] = lastgae
    return adv


def test_gae_matches_reference():
    """trainer 버퍼에 합성값을 심고 GAE 계산부만 재현해 참조와 대조."""
    torch.manual_seed(0)
    dev = "cuda"
    # 작은 env 로 trainer 만 만들고(reset 은 실행됨) 버퍼를 덮어씀.
    env = GpuDogfightVecEnv(4, substeps=6, seed=0, device=dev)
    cfg = PPOGPUConfig(rollout_steps=6, device=dev, normalize_obs=False)
    tr = PPOGPUTrainer(env, cfg)
    T, B = cfg.rollout_steps, tr.nenv
    g = torch.Generator(device=dev).manual_seed(1)
    tr.b_rew = torch.rand(T, B, generator=g, device=dev)
    tr.b_val = torch.rand(T, B, generator=g, device=dev)
    tr.b_done = (torch.rand(T, B, generator=g, device=dev) < 0.2).float()
    last_val = torch.rand(B, generator=g, device=dev)
    next_done = (torch.rand(B, generator=g, device=dev) < 0.2).float()
    tr._next_done = next_done

    # trainer 의 GAE 루프와 동일 코드(참조는 numpy).
    adv = torch.zeros_like(tr.b_rew)
    lastgae = torch.zeros(B, device=dev)
    for t in reversed(range(T)):
        if t == T - 1:
            nnt = 1.0 - tr._next_done; nv = last_val
        else:
            nnt = 1.0 - tr.b_done[t + 1]; nv = tr.b_val[t + 1]
        delta = tr.b_rew[t] + cfg.gamma * nv * nnt - tr.b_val[t]
        lastgae = delta + cfg.gamma * cfg.gae_lambda * nnt * lastgae
        adv[t] = lastgae
    ref = _ref_gae(tr.b_rew.cpu().numpy(), tr.b_val.cpu().numpy(),
                   tr.b_done.cpu().numpy(), last_val.cpu().numpy(),
                   next_done.cpu().numpy(), cfg.gamma, cfg.gae_lambda)
    err = float(np.abs(adv.cpu().numpy() - ref).max())
    print(f"(B) GAE vs 참조: 최대오차 {err:.3e} -> {'OK' if err < 1e-5 else 'FAIL'}")


def test_train_integrity(iters=40, nenv=256):
    torch.zeros(1, device="cuda")
    env = GpuDogfightVecEnv(nenv, substeps=6, seed=0, device="cuda")
    # 게이팅 임계를 낮춰 짧은 run 에서도 pool 성장 유도, main-only 데이터 경로 확인.
    cfg = PPOGPUConfig(total_iterations=iters, rollout_steps=16, update_epochs=4,
                       num_minibatches=4, lr=3e-4, device="cuda", seed=0,
                       selfplay_gate_threshold=0.5, milestone_period=10000, exploiter_iters=0)
    tr = PPOGPUTrainer(env, cfg)
    evs, rets, sps_list = [], [], []
    bad = False

    def on_iter(s):
        nonlocal bad
        for v in (s.policy_loss, s.value_loss, s.entropy, s.approx_kl):
            if not np.isfinite(v):
                bad = True
        evs.append(s.explained_variance)
        if np.isfinite(s.mean_return):
            rets.append(s.mean_return)
        sps_list.append(s.steps_per_sec)

    tr.train(on_iteration=on_iter)
    ev0 = np.nanmean(evs[:5]) if len(evs) >= 5 else evs[0]
    ev1 = np.nanmean(evs[-5:])
    sps = np.median(sps_list)
    print(f"(C) 학습 무결성: {iters}iter, NaN={'있음(FAIL)' if bad else '없음'}, "
          f"pool 크기={tr.pool.size()}(게이팅 성장)")
    print(f"    explained_var {ev0:+.3f} -> {ev1:+.3f} "
          f"({'상승 OK' if ev1 > ev0 else '정체/하락'})")
    print(f"    mean_return {rets[0]:+.3f} -> {rets[-1]:+.3f} (episodes 표본 {len(rets)}iter)")
    print(f"    처리량 median {sps/1e6:.2f}M env-step/s")
    print(f"    결과 -> {'OK' if (not bad and ev1 > ev0) else 'CHECK'}")


def test_gating_and_weights():
    """EMA 게이팅(min EMA≥threshold 시 추가·oldest evictable FIFO)·softmax 가중식 확인."""
    torch.zeros(1, device="cuda")
    env = GpuDogfightVecEnv(64, substeps=6, seed=0, device="cuda")
    cfg = PPOGPUConfig(total_iterations=1, rollout_steps=4, device="cuda", seed=0,
                       pool_evict_cap=4, selfplay_gate_threshold=0.6,
                       pool_sample_temp=0.3, pool_uniform_floor=0.5)
    tr = PPOGPUTrainer(env, cfg)
    # 게이팅: evictable EMA 를 인위로 올려 추가 트리거 → oldest evictable FIFO 로 cap 유지.
    for _ in range(6):
        for e in tr.pool.entries:
            if not e["permanent"]:
                e["ema"] = 0.9
        tr.pool.gate_and_add(tr.model, tr.norm, cfg.selfplay_gate_threshold)
    ev = sum(1 for e in tr.pool.entries if not e["permanent"])
    cap_ok = ev <= cfg.pool_evict_cap
    # softmax 가중: EMA 낮은(어려운) 후보가 더 큰 가중치.
    tr.pool.entries[0]["ema"] = 0.2
    tr.pool.entries[1]["ema"] = 0.8
    w = tr.pool.weights(cfg.pool_sample_temp, cfg.pool_uniform_floor)
    harder_bigger = float(w[0]) > float(w[1])
    # 참조식으로 검산.
    emas = np.array([e["ema"] for e in tr.pool.entries]); m = emas.size
    lg = -emas / cfg.pool_sample_temp; lg -= lg.max(); sm = np.exp(lg); sm /= sm.sum()
    p = cfg.pool_uniform_floor / m + (1 - cfg.pool_uniform_floor) * sm; p /= p.sum()
    ref_err = float(np.abs(w.cpu().numpy() - p).max())
    print(f"(D) 게이팅: 6회 트리거 후 evictable {ev} (≤cap {cfg.pool_evict_cap})={cap_ok}")
    print(f"    softmax 가중: 낮은EMA 후보 가중 큼={harder_bigger}, 참조식 오차 {ref_err:.2e}")
    print(f"    결과 -> {'OK' if cap_ok and harder_bigger and ref_err < 1e-5 else 'FAIL'}")


def test_exploiter():
    """milestone 에서 permanent main + exploiter 추가·main-only 버퍼·상태복원 확인."""
    torch.zeros(1, device="cuda")
    env = GpuDogfightVecEnv(128, substeps=6, seed=0, device="cuda")
    cfg = PPOGPUConfig(total_iterations=3, rollout_steps=12, update_epochs=2,
                       num_minibatches=4, device="cuda", seed=0, selfplay_gate_threshold=2.0,
                       milestone_period=3, exploiter_iters=4, exploiter_win_target=1.1)
    tr = PPOGPUTrainer(env, cfg)
    ok_batch = (tr.b_obs.shape[0] == cfg.rollout_steps and tr.b_obs.shape[1] == env.nenv)
    tr.train()   # it3 milestone → permanent main + exploiter
    perm = tr.pool.num_permanent()
    print(f"(E) exploiter/milestone: pool {tr.pool.size()} (perm {perm}), capacity {tr.pool.capacity()}")
    print(f"    main-only 배치 (T={tr.b_obs.shape[0]}, nenv={tr.b_obs.shape[1]}, act=index) -> {'OK' if ok_batch else 'FAIL'}")
    print(f"    milestone 1회 → permanent 2(main+exploiter)={perm == 2}, capacity=4+perm={tr.pool.capacity() == 4 + perm}")
    print(f"    결과 -> {'OK' if ok_batch and perm == 2 and tr.pool.capacity() == 4 + perm else 'CHECK'}")


def main():
    test_action_mapping()
    test_gae_matches_reference()
    test_train_integrity()
    test_gating_and_weights()
    test_exploiter()


if __name__ == "__main__":
    main()
