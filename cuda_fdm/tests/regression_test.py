# -*- coding: utf-8 -*-
"""Tolerance 기반 golden 회귀 테스트 (CI용).

세 경로를 golden 궤적(_bin/golden_ref.bin, 6초 4채널 기동)과 대조하고
각 지표를 TOL 에 대해 PASS/FAIL 판정, 하나라도 초과하면 exit(1).

  A) GPU 궤적 커널 fdm_run   (스레드=env, N step 내부루프)  ★필수
  B) 배치 env GpuDogfight.step()  (스레드=aircraft, obs 경로)  ★필수
  C) host-C out_c.bin        (있을 때만; g++ 빌드시 갱신)      선택

TOL 은 관측 오차(pos 0.036ft/att 0.003°/vel 0.017fps/alpha 5.5e-4°)에
약 3배 헤드룸 — 크로스-GPU/드라이버 변동은 통과, 실제 로직 회귀는 검출.

실행:  D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/regression_test.py
사전:  cuda_fdm/tests/export_seed.py 로 _bin/{seed,golden_ref,actions,meta} 생성.
"""
import sys, struct, math
from pathlib import Path
import ctypes as Ct
import torch

HERE = Path(__file__).resolve()
GEN = HERE.parents[1] / "gen"
BIN = HERE.parent / "_bin"
sys.path.insert(0, str(HERE.parent))              # cuda_rt
sys.path.insert(0, str(HERE.parents[1].parent))   # cuda_fdm.*
import cuda_rt

# 관측 최대오차 대비 ~3x 헤드룸. 초과시 회귀로 간주.
TOL = {"pos_ft": 0.10, "att_deg": 0.010, "vel_fps": 0.050, "alpha_deg": 0.005}
OBSERVED = {"pos_ft": 0.0356, "att_deg": 0.0030, "vel_fps": 0.017, "alpha_deg": 0.00055}
# FP32: 위치만 ECI 절대좌표 FP32 해상한계로 드리프트(~43ft/6s), 나머지는 FP64급.
TOL_FP32 = {"pos_ft": 60.0, "att_deg": 0.010, "vel_fps": 0.050, "alpha_deg": 0.005}
OBSERVED_FP32 = {"pos_ft": 42.8, "att_deg": 0.0031, "vel_fps": 0.015, "alpha_deg": 0.00059}


def wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def action_at(k):
    t = k / 60.0
    for te, a in [(1.5, [0, 0, 0, 0.8]), (3.0, [0, 0.3, 0, 0.8]),
                  (4.5, [0.3, 0, 0, 0.8]), (6.0, [0, 0, 0.3, 0.8])]:
        if t < te:
            return a
    return [0, 0, 0.3, 0.8]


def load_golden(N):
    raw = (BIN / "golden_ref.bin").read_bytes()
    return [struct.unpack_from("<10d", raw, k * 80) for k in range(N)]


def metrics_vs_golden(traj, gref):
    """traj[k] = (eci_pos3, euler3, vUVW3, alpha) 10-tuple. → 최대오차 dict(+argmax row)."""
    m = {"pos_ft": (0.0, 0), "att_deg": (0.0, 0), "vel_fps": (0.0, 0), "alpha_deg": (0.0, 0)}
    for k in range(len(gref)):
        o, g = traj[k], gref[k]
        dp = math.sqrt(sum((o[i] - g[i]) ** 2 for i in range(3)))
        da = max(abs(math.degrees(wrap(o[3 + i] - g[3 + i]))) for i in range(3))
        dv = max(abs(o[6 + i] - g[6 + i]) for i in range(3))
        dal = abs(math.degrees(o[9] - g[9]))
        for key, val in [("pos_ft", dp), ("att_deg", da), ("vel_fps", dv), ("alpha_deg", dal)]:
            if val > m[key][0]:
                m[key] = (val, k + 1)
    return m


# ---- A) GPU 궤적 커널 ----
def run_gpu_traj(N):
    src = cuda_rt.assemble_source(
        (GEN / "fdm_kernel.cu").read_text(encoding="utf-8"),
        (GEN / "fdm.cuh").read_text(encoding="utf-8"),
        (GEN / "f16_gen.cuh").read_text(encoding="utf-8"))
    cap = torch.cuda.get_device_capability()
    ptx = cuda_rt.compile_ptx(src, f"compute_{cap[0]}{cap[1]}")
    kern = cuda_rt.Kernel(ptx, "fdm_run")
    seed = torch.tensor(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()),
                        dtype=torch.float64)
    actions = torch.tensor(
        list(struct.unpack(f"<{N*4}d", (BIN / "actions.bin").read_bytes())),
        dtype=torch.float64).view(N, 4).contiguous().cuda()
    nenv = 4
    states = seed.repeat(nenv, 1).contiguous().cuda()
    traj = torch.zeros(N * 10, dtype=torch.float64, device="cuda")
    final = torch.zeros(nenv * 10, dtype=torch.float64, device="cuda")
    block = 128
    grid = ((nenv + block - 1) // block, 1, 1)
    args = [Ct.c_void_p(states.data_ptr()), Ct.c_void_p(actions.data_ptr()),
            Ct.c_int(N), Ct.c_int(nenv),
            Ct.c_void_p(traj.data_ptr()), Ct.c_void_p(final.data_ptr())]
    kern.launch(grid, (block, 1, 1), args)
    t = traj.cpu().numpy()
    return [tuple(t[k * 10:k * 10 + 10]) for k in range(N)]


# ---- B) 배치 env step() ----
def run_batch_env(N, precision="fp64"):
    from cuda_fdm.gpu_env import GpuDogfight, OBS
    seed = list(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()))
    env = GpuDogfight(3, substeps=1, precision=precision)
    env.load_seed(seed)
    traj = []
    for k in range(N):
        acts = torch.tensor([action_at(k)] * env.nac, dtype=torch.float64)
        o = env.step(acts)[0, 0].cpu().numpy()
        traj.append((o[0], o[1], o[2], o[6], o[7], o[8],
                     o[9], o[10], o[11], o[12]))  # pos3,euler3,vUVW3,alpha
    return traj


# ---- C) host-C (선택) ----
def load_host_c(N):
    p = BIN / "out_c.bin"
    if not p.exists():
        return None
    raw = p.read_bytes()
    return [struct.unpack_from("<10d", raw, k * 80) for k in range(N)]


def report(name, m, tol=TOL, observed=OBSERVED):
    print(f"\n[{name}]  vs golden")
    ok = True
    for key in tol:
        val, row = m[key]
        passed = val <= tol[key]
        ok = ok and passed
        flag = "PASS" if passed else "FAIL"
        print(f"    {key:10s}: {val:.4e} @row {row:<4d} "
              f"(tol {tol[key]:.0e}, obs {observed[key]:.1e})  {flag}")
    return ok


def main():
    if not (BIN / "meta.txt").exists():
        print("ERROR: _bin/ 아티팩트 없음. 먼저 export_seed.py 실행 (aip python).")
        sys.exit(2)
    if not torch.cuda.is_available():
        print("ERROR: CUDA 불가. aip_gpu python 으로 실행.")
        sys.exit(2)
    torch.zeros(1, device="cuda")
    N = int((BIN / "meta.txt").read_text())
    print(f"golden 회귀 테스트  N={N}  device={torch.cuda.get_device_name(0)}")
    gref = load_golden(N)

    # (name, metrics, tol, observed)
    checks = []
    checks.append(("A: GPU traj kernel FP64 (fdm_run)",
                   metrics_vs_golden(run_gpu_traj(N), gref), TOL, OBSERVED))
    checks.append(("B: batch env FP64 (GpuDogfight)",
                   metrics_vs_golden(run_batch_env(N, "fp64"), gref), TOL, OBSERVED))
    checks.append(("D: batch env FP32 (GpuDogfight)",
                   metrics_vs_golden(run_batch_env(N, "fp32"), gref), TOL_FP32, OBSERVED_FP32))
    hc = load_host_c(N)
    if hc is not None:
        checks.append(("C: host-C FP64 (out_c.bin)",
                       metrics_vs_golden(hc, gref), TOL, OBSERVED))
    else:
        print("\n(C host-C 스킵: out_c.bin 없음 — g++ 빌드시 검증)")

    all_ok = True
    for name, m, tol, obs in checks:
        all_ok = report(name, m, tol, obs) and all_ok

    print(f"\n{'='*52}\n결과: {'ALL PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
