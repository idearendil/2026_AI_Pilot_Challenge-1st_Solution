# -*- coding: utf-8 -*-
"""GpuDogfightVecEnv 검증:
(1) NED 브릿지가 pymap3d 참조(FighterSim 규약)와 일치.
(2) torch kinematics_state 의 euler/vUVW 가 커널 obs 와 일치.
(3) IC 왕복: reset(stagger=False) 후 state9 가 의도한 N/E/D·heading·speed 복원.
(4) staggered reset 이 env 별 위상을 분산.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import pymap3d as pm

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from cuda_fdm.rl_env import (GpuDogfightVecEnv, kinematics_state, ned_from_ecef_altasl,
                             ORIGIN_LAT_DEG, ORIGIN_LON_DEG, FT2M, M2FT, R2D)
from cuda_fdm.gpu_env import OBS
from cuda_fdm.ref import jsb_frames as FR


def test_bridge_vs_pymap3d(env):
    """state9 N/E/D vs pymap3d.geodetic2ned(mGeodLat,mLon,alt_asl) (CPU 참조)."""
    st = env.sim.states.cpu().numpy()
    ecef_m, euler, vUVW, alt_asl_ft = kinematics_state(env.sim.states)
    n, e, d = ned_from_ecef_altasl(ecef_m, alt_asl_ft * FT2M)
    got = torch.stack([n, e, d], 1).cpu().numpy()
    maxerr = 0.0
    for a in range(env.nac):
        eci = st[a, 0:3]; epa = st[a, 13]
        ce, se = math.cos(epa), math.sin(epa)
        ecef = (ce * eci[0] + se * eci[1], -se * eci[0] + ce * eci[1], eci[2])
        loc = FR.location_derived(ecef)
        radius = loc["radius"]; alt_asl = (radius - FR.SEMI_MAJOR) * FT2M
        ref = pm.geodetic2ned(math.degrees(loc["mGeodLat"]), math.degrees(loc["mLon"]),
                              alt_asl, ORIGIN_LAT_DEG, ORIGIN_LON_DEG, 0.0)
        err = max(abs(got[a][i] - ref[i]) for i in range(3))
        maxerr = max(maxerr, err)
    print(f"(1) NED 브릿지 vs pymap3d: 최대오차 {maxerr:.3e} m  "
          f"-> {'OK' if maxerr < 1e-3 else 'FAIL'}")


def test_kinematics_vs_kernel(env):
    """torch kinematics_state euler/vUVW == 커널 obs (step 후)."""
    env.sim.step(torch.zeros(env.nac, 4, dtype=torch.float64, device="cuda").add_(
        torch.tensor([0, 0, 0, 0.8], dtype=torch.float64, device="cuda")), substeps=1)
    _, euler, vUVW, alt = kinematics_state(env.sim.states)
    ob = env.sim.obs
    de = (euler - ob[:, OBS["euler"]]).abs().max().item()
    dv = (vUVW - ob[:, OBS["vUVW"]]).abs().max().item()
    da = (alt - ob[:, OBS["alt_asl"]]).abs().max().item()
    print(f"(2) torch kinematics vs 커널 obs: euler {de:.2e}rad  vUVW {dv:.2e}fps  "
          f"alt {da:.2e}ft  -> {'OK' if de<1e-9 and dv<1e-6 and da<1e-6 else 'FAIL'}")


def test_ic_roundtrip():
    """reset(stagger=False) 후 state9 가 의도한 N/E/D·heading·speed 복원."""
    env = GpuDogfightVecEnv(64, substeps=6, seed=0)
    # 알려진 IC 로 직접 구성(샘플 대신)
    from cuda_fdm.ic import build_seed_vector
    from cuda_fdm.rl_env import ned_to_geodetic_np, STATE_N
    cases = [(3500 + 300, 0, -22966 * FT2M, 90.0, 250.0),   # 북쪽, 동향
             (3500 - 300, 0, -22966 * FT2M, 270.0, 250.0),  # 남쪽, 서향
             (3500, 200, -10000 * FT2M, 180.0, 220.0)]
    seeds = np.zeros((env.nac, STATE_N))
    for i in range(env.nac):
        n, e, d, hdg, spd = cases[i % len(cases)]
        lat, lon, alt_ft = ned_to_geodetic_np(n, e, d)
        seeds[i] = build_seed_vector(lat_deg=lat, lon_deg=lon, alt_ft=alt_ft,
                                     vt_fps=spd * M2FT, psi_deg=hdg,
                                     phi_deg=0, theta_deg=0, alpha_deg=0, beta_deg=0,
                                     fuel_lbs=6000.0, throttle=0.8)
    env.sim.load_seed(seeds)
    s9 = env.state9().view(env.nac, 9).cpu().numpy()
    maxpos = maxhdg = maxspd = 0.0
    for i in range(env.nac):
        n, e, d, hdg, spd = cases[i % len(cases)]
        dp = max(abs(s9[i][0] - n), abs(s9[i][1] - e), abs(s9[i][2] - d))
        yaw = s9[i][5] % 360.0
        dh = min(abs(yaw - hdg), 360 - abs(yaw - hdg))
        spd_got = math.sqrt(sum(s9[i][6 + k] ** 2 for k in range(3)))
        maxpos = max(maxpos, dp); maxhdg = max(maxhdg, dh)
        maxspd = max(maxspd, abs(spd_got - spd))
    print(f"(3) IC 왕복: pos {maxpos:.3e}m  heading {maxhdg:.3e}deg  speed {maxspd:.3e}m/s  "
          f"-> {'OK' if maxpos<1.0 and maxhdg<0.1 and maxspd<1.0 else 'FAIL'}")


def test_stagger():
    """staggered reset 이 env 별 sim 진행량을 분산(state 다양성 확인)."""
    env = GpuDogfightVecEnv(256, substeps=6, seed=1)
    env.reset(stagger=True, max_stagger_steps=200)
    s9 = env.state9()[:, 0, :].cpu().numpy()   # ownship
    # 위상 분산 대리지표: N 위치 표준편차(같은 IC 분포라도 오프셋 다르면 퍼짐)
    n_std = float(np.std(s9[:, 0]))
    alt = -s9[:, 2]
    print(f"(4) stagger 후 ownship N std={n_std:.1f}m, alt범위=[{alt.min():.0f},{alt.max():.0f}]m "
          f"-> env들이 서로 다른 위상 {'OK' if n_std > 50 else '(분산작음)'}")
    # 무stagger 대비: 같은 시드로 stagger=False 면 위상 0
    env2 = GpuDogfightVecEnv(256, substeps=6, seed=1)
    env2.reset(stagger=False)
    s9b = env2.state9()[:, 0, :].cpu().numpy()
    print(f"    (참고) stagger=False N std={float(np.std(s9b[:,0])):.1f}m (IC 분포만)")


def main():
    torch.zeros(1, device="cuda")
    env = GpuDogfightVecEnv(64, substeps=6, seed=0)
    env.reset(stagger=False)
    test_bridge_vs_pymap3d(env)
    test_kinematics_vs_kernel(env)
    test_ic_roundtrip()
    test_stagger()


if __name__ == "__main__":
    main()
