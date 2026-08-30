# -*- coding: utf-8 -*-
"""obs_reward.py (GPU 벡터화 claude164r obs + my_reward) 를 CPU 참조와 대조.

(A) 배치 기하 vs GeometryInfo (distance/ata/aa/los az·el).
(B) 전체 관측: 실제 GPU 궤적을 CPU StateReconstructor+build_observation 과 step 별 대조.
(C) 보상 formula: 관점 기체 0 궤적을 my_reward.compute_reward 과 대조(damage+shaping).
(D) terminal 보상: 고도 이탈/HP 종료 합성 케이스 대조.
"""
import sys
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "claude_code"))

from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm import obs_reward as OR
from GeoMathUtil import GeometryInfo
import my_observation as MO
import my_reward as MR


def _rand_states(M, rng):
    s = np.zeros((M, 9))
    s[:, 0:3] = rng.uniform(-3000, 3000, size=(M, 3))
    s[:, 2] = rng.uniform(-9000, -300, size=M)          # D (고도 300~9000m)
    s[:, 3] = rng.uniform(-180, 180, size=M)            # roll
    s[:, 4] = rng.uniform(-80, 80, size=M)              # pitch
    s[:, 5] = rng.uniform(0, 360, size=M)               # yaw
    s[:, 6:9] = rng.uniform(-300, 300, size=(M, 3))
    s[:, 6] = rng.uniform(150, 300, size=M)             # u 전방 양수 위주
    return s


def test_geometry():
    rng = np.random.default_rng(0)
    M = 512
    own = _rand_states(M, rng)
    tgt = _rand_states(M, rng)
    geo = GeometryInfo()
    ownt = torch.tensor(own, device="cuda")
    tgtt = torch.tensor(tgt, device="cuda")
    g_dist = OR.distance_m(ownt, tgtt).cpu().numpy()
    g_ata = OR.ata_deg(ownt, tgtt).cpu().numpy()
    g_aa = OR.aspect_deg(ownt, tgtt).cpu().numpy()
    g_az, g_el = OR.los_az_el_deg(ownt, tgtt)
    g_az = g_az.cpu().numpy(); g_el = g_el.cpu().numpy()
    e_dist = e_ata = e_aa = e_az = e_el = 0.0
    for i in range(M):
        e_dist = max(e_dist, abs(g_dist[i] - geo._get_distance(own[i], tgt[i])))
        e_ata = max(e_ata, abs(g_ata[i] - geo._get_antenna_train_angle(own[i], tgt[i], False)))
        e_aa = max(e_aa, abs(g_aa[i] - geo._get_aspect_angle(own[i], tgt[i], False)))
        az, el = geo._get_los_angle(own[i], tgt[i])
        e_az = max(e_az, abs(g_az[i] - az)); e_el = max(e_el, abs(g_el[i] - el))
    ok = max(e_dist, e_ata, e_aa, e_az, e_el) < 1e-6
    print(f"(A) 기하 vs GeometryInfo: dist {e_dist:.2e}m ata {e_ata:.2e} aa {e_aa:.2e} "
          f"az {e_az:.2e} el {e_el:.2e} deg -> {'OK' if ok else 'FAIL'}")
    return ok


def test_obs_and_reward_trajectory(nenv=8, K=40, seed=3):
    torch.zeros(1, device="cuda")
    env = GpuDogfightVecEnv(nenv, substeps=6, seed=seed)
    env.reset(stagger=False)
    nac = env.nac
    bor = OR.BatchObsReward(nenv)
    geo = GeometryInfo()
    cpu_recs = [MO.StateReconstructor() for _ in range(nac)]
    MR.reset_distance_tracker()

    rng = np.random.default_rng(seed + 100)
    max_obs_err = 0.0
    max_rew_err = 0.0
    for step in range(K):
        acts = np.zeros((nac, 4))
        acts[:, 0:3] = rng.uniform(-0.3, 0.3, size=(nac, 3))
        acts[:, 3] = rng.uniform(0.6, 0.9, size=nac)
        acts_t = torch.tensor(acts, device="cuda")

        bor.push_actions(acts_t)
        for a in range(nac):
            cpu_recs[a].push_action(acts[a])

        env.step(acts_t)
        s9 = env.state9().reshape(nac, 9)
        s9_np = s9.cpu().numpy()

        bor.advance(s9)
        for a in range(nac):
            cpu_recs[a].advance(s9_np[a], s9_np[a ^ 1])

        gpu_obs = bor.build_obs(s9).cpu().numpy()
        for a in range(nac):
            cpu_obs = MO.build_observation(s9_np[a], s9_np[a ^ 1], geo,
                                           reconstructor=cpu_recs[a])
            max_obs_err = max(max_obs_err, float(np.max(np.abs(gpu_obs[a] - cpu_obs))))

        # 보상: 관점 기체 0 만 CPU 대조(전역 상태 충돌 회피)
        term_env = torch.zeros(nenv, dtype=torch.bool, device="cuda")
        gpu_rew = bor.compute_reward(s9, term_env).cpu().numpy()
        own_full = np.zeros(46); tgt_full = np.zeros(46)
        own_full[0:9] = s9_np[0]; tgt_full[0:9] = s9_np[1]
        own_full[MO.StateIndex.HEALTH] = float(bor.hp[0].item())
        tgt_full[MO.StateIndex.HEALTH] = float(bor.hp[1].item())
        own_full[MO.StateIndex.SIM_TIME] = float(bor.t_sec[0].item())
        own_full[MO.StateIndex.ALT] = -s9_np[0][2]
        loss_o = float(bor.hp_loss[0].item()); loss_t = float(bor.hp_loss[1].item())
        cpu_r, _ = MR.compute_reward(own_full, tgt_full, loss_o, loss_t, geo, {},
                                     MR.MY_REWARD_CONFIG, False, False, "")
        max_rew_err = max(max_rew_err, abs(gpu_rew[0] - cpu_r))

    ok_obs = max_obs_err < 3e-4
    ok_rew = max_rew_err < 2e-3
    print(f"(B) 관측 vs build_observation ({K}스텝×{nac}기): 최대오차 {max_obs_err:.2e} "
          f"-> {'OK' if ok_obs else 'FAIL'}")
    print(f"(C) 보상 vs compute_reward (damage+shaping): 최대오차 {max_rew_err:.2e} "
          f"-> {'OK' if ok_rew else 'FAIL'}")
    return ok_obs and ok_rew


def test_terminal_reward():
    """고도 이탈/HP 종료 terminal 항 대조(합성)."""
    nenv = 4
    bor = OR.BatchObsReward(nenv)
    geo = GeometryInfo()
    # env0: own 고도 이탈, env1: tgt 고도 이탈, env2: own HP 0, env3: 정상 종료(HP 미결)
    s9 = torch.zeros(2 * nenv, 9, device="cuda", dtype=torch.float64)
    for e in range(nenv):
        s9[2 * e, 0] = 0.0; s9[2 * e + 1, 0] = 1000.0    # 거리
    s9[0, 2] = -100.0   # env0 own alt 100m (<300 이탈)
    s9[1, 2] = -5000.0
    s9[2, 2] = -5000.0; s9[3, 2] = -100.0                # env1 tgt(=ac3) alt 이탈
    s9[4, 2] = -5000.0; s9[5, 2] = -5000.0
    s9[6, 2] = -5000.0; s9[7, 2] = -5000.0
    # HP 설정
    bor.hp[4] = 0.0     # env2 own(ac4) HP 0
    # advance 없이 prev_x_valid=False → shaping 0. hp_loss=0.
    term = torch.ones(nenv, dtype=torch.bool, device="cuda")
    gpu_r = bor.compute_reward(s9, term).cpu().numpy()
    cfg = MR.MY_REWARD_CONFIG
    exp = {0: cfg["ownship_alt_reward"], 3: cfg["target_alt_reward"],
           4: cfg["loss_reward"]}
    # ac0: own 이탈 -20; ac3: 자기 own alt 이탈? ac3 own=s9[3] alt 100 → own_below -20.
    # (env1 에서 관점 ac2 의 target=ac3 이 이탈 → ac2 는 +5)
    exp = {0: -20.0, 1: 5.0,      # ac0 own이탈; ac1 의 target(ac0) 이탈 → +5
           2: 5.0,               # ac2 의 target(ac3) 이탈 → +5
           3: -20.0}             # ac3 own 이탈 → -20
    err = 0.0
    for a, v in exp.items():
        err = max(err, abs(gpu_r[a] - v))
    # ac4: own HP0 → loss_reward(0) + (own alt 5000 정상) → 0
    err = max(err, abs(gpu_r[4] - cfg["loss_reward"]))
    ok = err < 1e-9
    print(f"(D) terminal 보상 합성 케이스: 최대오차 {err:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_kernel_vs_torch(nenv=16, K=50, seed=11):
    """융합 커널(advance+reward, build_obs) vs torch 경로(=CPU참조와 비트일치)."""
    from cuda_fdm.gpu_env import GpuDogfight
    torch.zeros(1, device="cuda")
    sim = GpuDogfight(nenv, substeps=6, planes_per_env=2)
    env = GpuDogfightVecEnv(nenv, substeps=6, seed=seed)   # IC 샘플용
    seeds = env._build_all_seeds()
    sim.load_seed(seeds)
    nac = 2 * nenv
    bor_t = OR.BatchObsReward(nenv, enable_kernel=False)   # torch 경로
    bor_k = OR.BatchObsReward(nenv, enable_kernel=True)    # 커널 경로
    rng = np.random.default_rng(seed + 5)
    max_obs = max_rew = max_hp = max_pqr = 0.0
    term_mismatch = 0
    cfg = MR.MY_REWARD_CONFIG
    for step in range(K):
        acts = np.zeros((nac, 4))
        acts[:, 0:3] = rng.uniform(-0.4, 0.4, size=(nac, 3))
        acts[:, 3] = rng.uniform(0.5, 0.95, size=nac)
        acts_t = torch.tensor(acts, device="cuda")
        sim.step(acts_t, substeps=6)
        # torch 경로 (state9 는 rl_env 계산 재사용)
        from cuda_fdm.rl_env import kinematics_state, ned_from_ecef_altasl, FT2M, R2D
        ecef_m, euler, vUVW, alt_ft = kinematics_state(sim.states)
        n, e, d = ned_from_ecef_altasl(ecef_m, alt_ft * FT2M)
        s9 = torch.cat([torch.stack([n, e, d], 1), euler * R2D, vUVW * FT2M], 1)
        bor_t.push_actions(acts_t); bor_t.advance(s9)
        obs_t = bor_t.build_obs(s9)
        term_env = torch.zeros(nenv, dtype=torch.bool, device="cuda")  # 비교는 공식만
        rew_t = bor_t.compute_reward(s9, term_env, cfg=cfg)
        # 커널 경로
        rew_k, term_k, trunc_k = bor_k.kernel_advance(sim.states, acts_t, cfg=cfg)
        obs_k = bor_k.kernel_build_obs(sim.states)
        max_obs = max(max_obs, (obs_k - obs_t).abs().max().item())
        # reward: terminal 항을 뺀 damage+shaping 비교(둘 다 term=False 로)
        # 커널 rew_k 는 실제 종료 반영 → term=False env 만 비교
        keep = (~term_k.bool()).repeat_interleave(2)
        if keep.any():
            max_rew = max(max_rew, (rew_k[keep] - rew_t[keep]).abs().max().item())
        max_hp = max(max_hp, (bor_k.hp - bor_t.hp).abs().max().item())
        max_pqr = max(max_pqr, (bor_k.pqr - bor_t.pqr).abs().max().item())
    ok = max_obs < 5e-4 and max_rew < 5e-3 and max_hp < 1e-9 and max_pqr < 1e-6
    print(f"(E) 커널 vs torch: obs {max_obs:.2e} reward {max_rew:.2e} hp {max_hp:.2e} "
          f"pqr {max_pqr:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    r = []
    r.append(test_geometry())
    r.append(test_obs_and_reward_trajectory())
    r.append(test_terminal_reward())
    r.append(test_kernel_vs_torch())
    print("ALL PASS" if all(r) else "FAIL")
    sys.exit(0 if all(r) else 1)


if __name__ == "__main__":
    main()
