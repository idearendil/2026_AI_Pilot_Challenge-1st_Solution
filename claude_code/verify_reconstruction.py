"""학습 환경에서 '재구성한 state' vs '실제 env state' 검증 + delta_t/시간게이팅 확인.

  python claude_code/verify_reconstruction.py --bundle-dir artifacts/models/claude_demo/ppo_mlp_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch

from dogfight.sim.state_schema import StateIndex

from claude_code.env_utils import make_env
from claude_code.model import load_bundle, make_obs_normalizer
from claude_code.my_observation import (
    StateReconstructor, reconstruct_altitude, reconstruct_speed, damage_rate,
    DT_PER_STEP, TIER2_START_SEC, TIER3_START_SEC,
)

KTAS_INDEX = 27  # FighterSim: state[27] = True Air Speed (m/s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", default="artifacts/models/claude_demo/ppo_mlp_v1")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--target-mode", default="loiter")
    return p.parse_args()


def main():
    args = parse_args()
    env = make_env(overrides={"target_mode": args.target_mode}, runner_index="verify")
    print(f"DT_PER_STEP = {DT_PER_STEP} s  (env._delta_t={env._delta_t:.5f} * step_ratio={env._step_ratio})")
    print(f"시간 게이팅: tier2 from {TIER2_START_SEC}s, tier3 from {TIER3_START_SEC}s\n")

    model, meta = load_bundle(args.bundle_dir)
    normalizer = make_obs_normalizer(meta.get("obs_normalization"))
    model_dim = int(meta.get("observation_size", 16))

    recon = StateReconstructor()
    recon.reset()
    obs, _ = env.reset(seed=7)
    geo = env._geo_info

    alt_err, spd_err, hp_tgt_diff = [], [], []
    env_late_tier_damage = 0   # env 가 ATA 1~3deg(tier2/3 영역)에서 damage 준 횟수
    rows = []
    for t in range(args.steps):
        if obs.shape[0] == model_dim:
            with torch.no_grad():
                a = model.act_deterministic(
                    torch.as_tensor(normalizer(obs), dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy().astype(np.float32)
        else:
            a = np.zeros(4, dtype=np.float32)

        obs, r, term, trunc, info = env.step(a)
        own = np.asarray(env._ownship_state, dtype=np.float64)
        tgt = np.asarray(env._target_state, dtype=np.float64)
        recon.advance(own, tgt)

        recon_alt, real_alt = reconstruct_altitude(own), float(own[StateIndex.ALT])
        recon_spd, real_ktas = reconstruct_speed(own), float(own[KTAS_INDEX])
        real_hp_t = float(tgt[StateIndex.HEALTH])
        alt_err.append(abs(recon_alt - real_alt))
        spd_err.append(abs(recon_spd - real_ktas))
        hp_tgt_diff.append(abs(recon.hp_tgt - real_hp_t))

        ata = abs(geo._get_antenna_train_angle(own, tgt, False))
        if 1.0 < ata < 3.0 and float(info.get("target_damage", 0.0)) > 0:
            env_late_tier_damage += 1

        if recon.last_dmg_dealt > 0 or real_hp_t < 1.0:
            rows.append((t, real_hp_t, recon.hp_tgt, recon.last_r_ft, recon.last_ata_own,
                         recon.last_dmg_dealt))
        if term or trunc:
            break

    print(f"=== 재구성 정확도 (steps={len(alt_err)}) ===")
    print(f"고도(-D vs ALT[44])   : mean |err| = {np.mean(alt_err):.4f} m,  max = {np.max(alt_err):.4f} m")
    print(f"속도(||v|| vs KTAS[27]): mean |err| = {np.mean(spd_err):.4f} m/s, max = {np.max(spd_err):.4f} m/s")
    print(f"표적 HP(재구성 vs env real): mean |diff| = {np.mean(hp_tgt_diff):.4f}, "
          f"max = {np.max(hp_tgt_diff):.4f}")
    print(f"env 가 ATA 1~3deg(tier2/3 영역)에서 damage 준 횟수 = {env_late_tier_damage} "
          f"(0 이면 env 는 tier1 전용=시간게이팅 없음 확인)")

    if rows:
        print("\n=== HP 변화 구간 (t, HPtgt real/recon, r_ft, ATA, rate) ===")
        for (t, ht, htr, rft, ata, rate) in rows[:20]:
            print(f"  t={t:4d} | real {ht:5.2f}  recon {htr:5.2f} | r_ft {rft:6.0f} "
                  f"ATA {ata:5.2f} rate {rate:.4f}")

    # 시간 게이팅 합성 데모: 동일 기하에서 시점만 바꿔 tier 활성화 확인
    print("\n=== 시간 게이팅 데모 (r=2000ft 고정) ===")
    for r_ft, th, label in [(2000, 0.5, "tier1(±1°)"), (2000, 1.5, "tier2(±2°)"),
                            (2000, 2.5, "tier3(±3°)")]:
        v0 = damage_rate(r_ft, th, 50.0)
        v120 = damage_rate(r_ft, th, 120.0)
        v160 = damage_rate(r_ft, th, 160.0)
        print(f"  {label}: t=50s→{v0:.3f}  t=120s→{v120:.3f}  t=160s→{v160:.3f}")

    print("\n[주석] 고도·속도는 위치/속도만으로 정확히 복원(오차≈0). 표적 HP 재구성은 이제")
    print("       env 와 동일한 시간적분(rate*DT_PER_STEP)이라 tier1 구간에서 env real HP 와")
    print("       근접. 시간 게이팅(tier2 100s, tier3 150s)은 대결 서버 규칙이며 학습 env 엔")
    print("       없음(위 'damage 준 횟수=0' 확인). 학습에서 tier2/3 를 겪으려면 episode 가")
    print("       100s/150s 를 넘어야 함(--max-engage-time 상향 필요).")
    env.close()


if __name__ == "__main__":
    main()
