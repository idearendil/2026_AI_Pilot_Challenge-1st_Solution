# -*- coding: utf-8 -*-
"""P5: --maxrregcount 스윕. register-bound(255regs,17%occ) 병목을 낮추면
occupancy↑ vs 스필↑ 트레이드오프. throughput 실측으로 최적 cap 결정.
정합은 유지(같은 소스, 레지스터 배치만 변화 — --fmad=false 그대로)."""
import sys, struct, time
from pathlib import Path
import ctypes as Ct
import torch

HERE = Path(__file__).resolve()
GEN = HERE.parents[1] / "gen"
BIN = HERE.parent / "_bin"
sys.path.insert(0, str(HERE.parent))
import cuda_rt

ATTR = {"LOCAL_BYTES": 3, "NUM_REGS": 4}


def fattr(kern, code):
    v = Ct.c_int(0)
    cuda_rt._cu.cuFuncGetAttribute(Ct.byref(v), Ct.c_int(code), kern.func)
    return v.value


def occ(kern, block, max_tpm):
    n = Ct.c_int(0)
    cuda_rt._cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        Ct.byref(n), kern.func, Ct.c_int(block), Ct.c_size_t(0))
    return 100.0 * n.value * block / max_tpm, n.value


_PTX = None


def build(regcap):
    global _PTX
    if _PTX is None:
        src = cuda_rt.assemble_source(
            (GEN / "fdm_kernel.cu").read_text(encoding="utf-8"),
            (GEN / "fdm.cuh").read_text(encoding="utf-8"),
            (GEN / "f16_gen.cuh").read_text(encoding="utf-8"))
        cap = torch.cuda.get_device_capability()
        _PTX = cuda_rt.compile_ptx(src, f"compute_{cap[0]}{cap[1]}")
    # 레지스터 제어는 드라이버 JIT(CU_JIT_MAX_REGISTERS)에서.
    return cuda_rt.Kernel(_PTX, "fdm_step_batch", max_registers=(regcap or 0))


def bench(kern, nac, block, ss, reps=40):
    seed = torch.tensor(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()),
                        dtype=torch.float64)
    states = seed.unsqueeze(0).expand(nac, -1).contiguous().cuda()
    states0 = states.clone()
    actions = torch.zeros(nac, 4, dtype=torch.float64, device="cuda"); actions[:, 3] = 0.8
    obs = torch.zeros(nac, 17, dtype=torch.float64, device="cuda")
    grid = ((nac + block - 1) // block, 1, 1)
    args = [Ct.c_void_p(states.data_ptr()), Ct.c_void_p(actions.data_ptr()),
            Ct.c_int(nac), Ct.c_int(ss), Ct.c_void_p(obs.data_ptr())]
    kern.launch(grid, (block, 1, 1), args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        states.copy_(states0)
        kern.launch(grid, (block, 1, 1), args)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    torch.zeros(1, device="cuda")
    dev = torch.cuda.get_device_properties(0)
    mtpm = dev.max_threads_per_multi_processor
    nac, block = 65536, 128
    print(f"{dev.name}  nac={nac} block={block}  (기준: cap없음 255regs/17%/29.5M)\n")
    print(f"{'regcap':>8} {'regs':>5} {'local':>6} {'occ%':>6} {'blk/SM':>7} "
          f"{'ss1 M/s':>9} {'ss4 M/s':>9}")
    for regcap in (None, 144, 128, 112, 96, 88):
        try:
            kern = build(regcap)
        except RuntimeError as e:
            print(f"{str(regcap):>8}  compile FAIL: {str(e)[:40]}")
            continue
        regs = fattr(kern, ATTR["NUM_REGS"])
        localb = fattr(kern, ATTR["LOCAL_BYTES"])
        o, nb = occ(kern, block, mtpm)
        dt1 = bench(kern, nac, block, 1)
        dt4 = bench(kern, nac, block, 4)
        print(f"{str(regcap):>8} {regs:>5} {localb:>6} {o:>6.0f} {nb:>7} "
              f"{nac/dt1/1e6:>9.1f} {nac*4/dt4/1e6:>9.1f}")


if __name__ == "__main__":
    main()
