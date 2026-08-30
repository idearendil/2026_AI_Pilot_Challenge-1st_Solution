# -*- coding: utf-8 -*-
"""P5 프로파일: fdm_step_batch 커널의 레지스터/local memory/occupancy 실측 +
block/nac 스윕 처리량 벤치. 최적화 방향(무엇에 bound 인지)을 데이터로 결정.

실행:  D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/profile_kernel.py
"""
import sys, struct, time
from pathlib import Path
import ctypes as Ct
import torch

HERE = Path(__file__).resolve()
GEN = HERE.parents[1] / "gen"
BIN = HERE.parent / "_bin"
sys.path.insert(0, str(HERE.parent))
import cuda_rt

# cuFuncGetAttribute enum
ATTR = {"MAX_THREADS_PER_BLOCK": 0, "SHARED_BYTES": 1, "CONST_BYTES": 2,
        "LOCAL_BYTES": 3, "NUM_REGS": 4}


def func_attr(kern, code):
    v = Ct.c_int(0)
    cuda_rt._cu.cuFuncGetAttribute(Ct.byref(v), Ct.c_int(code), kern.func)
    return v.value


def occupancy(kern, block):
    n = Ct.c_int(0)
    r = cuda_rt._cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        Ct.byref(n), kern.func, Ct.c_int(block), Ct.c_size_t(0))
    return n.value if r == 0 else -1


def build(kernel_name):
    src = cuda_rt.assemble_source(
        (GEN / "fdm_kernel.cu").read_text(encoding="utf-8"),
        (GEN / "fdm.cuh").read_text(encoding="utf-8"),
        (GEN / "f16_gen.cuh").read_text(encoding="utf-8"))
    cap = torch.cuda.get_device_capability()
    ptx = cuda_rt.compile_ptx(src, f"compute_{cap[0]}{cap[1]}")
    return cuda_rt.Kernel(ptx, kernel_name)


def bench_step(kern, nac, block, substeps, reps=20):
    states = torch.zeros(nac, 101, dtype=torch.float64, device="cuda")
    seed = torch.tensor(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()),
                        dtype=torch.float64)
    states.copy_(seed.unsqueeze(0).expand(nac, -1))
    states0 = states.clone()
    actions = torch.zeros(nac, 4, dtype=torch.float64, device="cuda")
    actions[:, 3] = 0.8
    obs = torch.zeros(nac, 17, dtype=torch.float64, device="cuda")
    grid = ((nac + block - 1) // block, 1, 1)
    args = [Ct.c_void_p(states.data_ptr()), Ct.c_void_p(actions.data_ptr()),
            Ct.c_int(nac), Ct.c_int(substeps), Ct.c_void_p(obs.data_ptr())]
    kern.launch(grid, (block, 1, 1), args)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        states.copy_(states0)
        kern.launch(grid, (block, 1, 1), args)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return dt


def main():
    assert torch.cuda.is_available()
    torch.zeros(1, device="cuda")
    dev = torch.cuda.get_device_properties(0)
    print(f"device={dev.name}  SM={dev.multi_processor_count}  "
          f"regs/SM=65536  maxThreads/SM={dev.max_threads_per_multi_processor}")

    kern = build("fdm_step_batch")
    regs = func_attr(kern, ATTR["NUM_REGS"])
    localb = func_attr(kern, ATTR["LOCAL_BYTES"])
    constb = func_attr(kern, ATTR["CONST_BYTES"])
    maxtpb = func_attr(kern, ATTR["MAX_THREADS_PER_BLOCK"])
    print(f"\nfdm_step_batch:  regs/thread={regs}  local={localb} B/thread  "
          f"const={constb} B  maxThreads/block={maxtpb}")
    print(f"  (FdmState=808 B; local {localb}B => "
          f"{'스필 있음' if localb > 0 else '스필 없음'})")

    print("\n[occupancy] block별 SM당 활성 블록/워프 (100% = "
          f"{dev.max_threads_per_multi_processor//32} warp):")
    for b in (32, 64, 96, 128, 192, 256):
        if b > maxtpb:
            continue
        nb = occupancy(kern, b)
        warps = nb * (b // 32)
        occ = 100.0 * warps * 32 / dev.max_threads_per_multi_processor
        print(f"    block={b:4d}: {nb} blk/SM  {warps} warp/SM  occ={occ:.0f}%")

    print("\n[throughput] nac x block 스윕 (substeps=1), M ac-steps/s:")
    nacs = [4096, 16384, 65536, 262144]
    blocks = [64, 128, 256]
    hdr = "  nac\\blk " + "".join(f"{b:>10d}" for b in blocks)
    print(hdr)
    for nac in nacs:
        row = f"  {nac:8d}"
        for b in blocks:
            dt = bench_step(kern, nac, b, 1)
            row += f"{nac/dt/1e6:>10.1f}"
        print(row)

    print("\n[substeps 스케일] nac=65536 block=128:")
    for ss in (1, 2, 4, 8):
        dt = bench_step(kern, 65536, 128, ss)
        print(f"    ss={ss}: {dt*1000:.3f} ms  {65536*ss/dt/1e6:.1f} M ac-steps/s  "
              f"({65536/dt/1e6:.1f} M launches/s)")


if __name__ == "__main__":
    main()
