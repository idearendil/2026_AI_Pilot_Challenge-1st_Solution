# -*- coding: utf-8 -*-
"""GpuDogfight(배치 env) 검증:
(1) 모든 aircraft 를 golden 시드로 로드, golden 액션 스케줄로 step() 360회 →
    env0/plane0 obs 가 golden 궤적과 일치하는지 (step API+배치+obs 경로 검증).
(2) 독립성: 같은 env 의 plane0/plane1 에 다른 throttle → Vt 갈라지는지.
"""
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
    seed = list(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()))
    gref_raw = (BIN / "golden_ref.bin").read_bytes()
    gref = [struct.unpack_from("<10d", gref_raw, k * 80) for k in range(N)]

    nenv = 3
    env = GpuDogfight(nenv, substeps=1)
    env.load_seed(seed)  # 모든 aircraft 동일 시드

    metrics = {"pos_ft": 0.0, "att_deg": 0.0, "vel_fps": 0.0, "alpha_deg": 0.0}
    for k in range(N):
        a = action_at(k)
        acts = torch.tensor([a] * env.nac, dtype=torch.float64)
        obs = env.step(acts)  # (nenv,ppe,17)
        o = obs[0, 0].cpu().numpy()
        g = gref[k]
        dp = math.sqrt(sum((o[OBS["eci_pos"]][i] - g[i]) ** 2 for i in range(3)))
        da = max(abs(math.degrees(wrap(o[OBS["euler"]][i] - g[3 + i]))) for i in range(3))
        dv = max(abs(o[OBS["vUVW"]][i] - g[6 + i]) for i in range(3))
        dal = abs(math.degrees(o[OBS["alpha"]] - g[9]))
        metrics["pos_ft"] = max(metrics["pos_ft"], dp)
        metrics["att_deg"] = max(metrics["att_deg"], da)
        metrics["vel_fps"] = max(metrics["vel_fps"], dv)
        metrics["alpha_deg"] = max(metrics["alpha_deg"], dal)
    print(f"(1) 배치 env0/plane0 vs golden (N={N}, nenv={nenv}):")
    for kk in metrics:
        print(f"    {kk:10s}: {metrics[kk]:.4e}")

    # (2) 독립성 테스트: 재시드 후 plane0=throttle0.8, plane1=throttle0.2
    env.load_seed(seed)
    acts = torch.zeros(env.nac, 4, dtype=torch.float64)
    acts[:, 3] = 0.8
    acts[1::2, 3] = 0.2  # 각 env 의 plane1 은 throttle 0.2
    for k in range(120):
        env.step(acts)
    obs = env.get_obs().view(env.nenv, env.ppe, -1)
    Vt0 = obs[0, 0, OBS["Vt"]].item()
    Vt1 = obs[0, 1, OBS["Vt"]].item()
    print(f"(2) 독립성 2s 후 Vt: plane0(thr0.8)={Vt0:.2f}  plane1(thr0.2)={Vt1:.2f}  "
          f"diff={Vt0-Vt1:.2f} fps  -> {'OK' if abs(Vt0-Vt1)>1.0 else 'FAIL'}")
    # 모든 env 동일한지(같은 시드/액션이므로 env0==env1==env2)
    same = torch.allclose(obs[0], obs[1]) and torch.allclose(obs[0], obs[2])
    print(f"    env 간 동일성(같은 시드/액션): {'OK' if same else 'FAIL'}")


if __name__ == "__main__":
    main()
