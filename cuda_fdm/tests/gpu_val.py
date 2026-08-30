# -*- coding: utf-8 -*-
"""GPU 검증/성능: NVRTC 로 fdm_kernel.cu 컴파일 → nenv env 병렬 N step.
env0 전 궤적을 out_gpu.bin 으로 → compare.py 로 golden 대조.
사용: gpu_val.py [nenv]   (기본 4=검증, 큰 값=성능)"""
import sys, struct, time
from pathlib import Path
import ctypes as C
import torch

HERE = Path(__file__).resolve()
GEN = HERE.parents[1] / "gen"
BIN = HERE.parent / "_bin"
sys.path.insert(0, str(HERE.parent))
import cuda_rt


def main():
    nenv = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    assert torch.cuda.is_available()
    torch.zeros(1, device="cuda")  # torch primary context 초기화
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}"
    print(f"device={torch.cuda.get_device_name(0)}  arch={arch}  nenv={nenv}")

    src = cuda_rt.assemble_source(
        (GEN / "fdm_kernel.cu").read_text(encoding="utf-8"),
        (GEN / "fdm.cuh").read_text(encoding="utf-8"),
        (GEN / "f16_gen.cuh").read_text(encoding="utf-8"))
    t0 = time.time()
    ptx = cuda_rt.compile_ptx(src, arch)
    print(f"NVRTC compiled PTX ({len(ptx)} bytes) in {time.time()-t0:.2f}s")
    kern = cuda_rt.Kernel(ptx, "fdm_run")

    N = int((BIN / "meta.txt").read_text())
    seed = torch.tensor(struct.unpack("<101d", (BIN / "seed.bin").read_bytes()),
                        dtype=torch.float64)
    actions = torch.tensor(
        list(struct.unpack(f"<{N*4}d", (BIN / "actions.bin").read_bytes())),
        dtype=torch.float64).view(N, 4).contiguous().cuda()

    states0 = seed.repeat(nenv, 1).contiguous().cuda()  # (nenv,101) 원본 시드
    states = states0.clone()
    traj = torch.zeros(N * 10, dtype=torch.float64, device="cuda")
    final = torch.zeros(nenv * 10, dtype=torch.float64, device="cuda")

    block = 128
    grid = (nenv + block - 1) // block
    args = [C.c_void_p(states.data_ptr()), C.c_void_p(actions.data_ptr()),
            C.c_int(N), C.c_int(nenv),
            C.c_void_p(traj.data_ptr()), C.c_void_p(final.data_ptr())]
    # warmup + timing (재시드는 device-to-device copy)
    kern.launch((grid, 1, 1), (block, 1, 1), args)
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 5
    for _ in range(reps):
        states.copy_(states0)              # device-to-device 재시드
        kern.launch((grid, 1, 1), (block, 1, 1), args)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    steps = nenv * N
    print(f"per-run {dt*1000:.2f} ms  |  {steps/dt/1e6:.2f} M env-steps/s  "
          f"({nenv} env x {N} steps)")

    # env0 궤적 저장
    traj_c = traj.cpu().numpy().tobytes()
    (BIN / "out_gpu.bin").write_bytes(traj_c)
    print("wrote out_gpu.bin")


if __name__ == "__main__":
    main()
