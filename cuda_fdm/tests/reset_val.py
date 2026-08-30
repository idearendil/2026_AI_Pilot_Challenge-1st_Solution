# -*- coding: utf-8 -*-
"""reset_ic (IC→seed, CPU ref) 검증: golden IC 로 리셋 후 golden 스케줄 스텝 →
env0/plane0 vs golden. golden-exact 시드가 아니므로 '리셋 과도'를 정량화."""
import sys, struct, math
from pathlib import Path
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1].parent))
from cuda_fdm.gpu_env import GpuDogfight, OBS
BIN = HERE.parent / "_bin"


def action_at(k):
    t = k / 60.0
    for te, a in [(1.5, [0, 0, 0, 0.8]), (3.0, [0, 0.3, 0, 0.8]),
                  (4.5, [0.3, 0, 0, 0.8]), (6.0, [0, 0, 0.3, 0.8])]:
        if t < te:
            return a
    return [0, 0, 0.3, 0.8]


def wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def main():
    N = int((BIN / "meta.txt").read_text())
    gref_raw = (BIN / "golden_ref.bin").read_bytes()
    gref = [struct.unpack_from("<10d", gref_raw, k * 80) for k in range(N)]

    ic = dict(lat_deg=37.92355643001773, lon_deg=128.18188127777776,
              alt_ft=22966.137662702928, vt_fps=984.252, fuel_lbs=6972.0)
    nenv = 2
    env = GpuDogfight(nenv, substeps=1)
    env.reset_ic([ic] * env.nac)

    m = {"pos_ft": (0.0, 0), "att_deg": (0.0, 0), "vel_fps": (0.0, 0), "alpha_deg": (0.0, 0)}
    for k in range(N):
        obs = env.step(torch.tensor([action_at(k)] * env.nac, dtype=torch.float64))
        o = obs[0, 0].cpu().numpy(); g = gref[k]
        dp = math.sqrt(sum((o[OBS["eci_pos"]][i] - g[i]) ** 2 for i in range(3)))
        da = max(abs(math.degrees(wrap(o[OBS["euler"]][i] - g[3 + i]))) for i in range(3))
        dv = max(abs(o[OBS["vUVW"]][i] - g[6 + i]) for i in range(3))
        dal = abs(math.degrees(o[OBS["alpha"]] - g[9]))
        for key, val in [("pos_ft", dp), ("att_deg", da), ("vel_fps", dv), ("alpha_deg", dal)]:
            if val > m[key][0]:
                m[key] = (val, k + 1)
    print(f"reset_ic(golden IC) 후 golden 스케줄 vs golden (N={N}):")
    for key in m:
        print(f"    {key:10s}: {m[key][0]:.4e}  @row {m[key][1]}")


if __name__ == "__main__":
    main()
