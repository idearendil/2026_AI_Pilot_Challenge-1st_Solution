# -*- coding: utf-8 -*-
"""GPU 배치 F16 FDM env (1v1 = 2기/env). NVRTC 컴파일된 fdm_step_batch 커널을
torch cuda 텐서 상태로 구동. reset/step/get_state 벡터 API.

상태: states (nac,101) float64  (nac = 2*nenv; a=2*e+0 ownship, 2*e+1 opponent)
obs  : (nac,17) = eci_pos3,eci_vel3,euler3,vUVW3,alpha,beta,mach,Vt,alt_asl
"""
import sys
from pathlib import Path
import ctypes as C
import torch

HERE = Path(__file__).resolve().parent
GEN = HERE / "gen"
sys.path.insert(0, str(HERE / "tests"))
import cuda_rt

STATE_N = 101
OBS_N = 17
# obs 컬럼 인덱스
OBS = dict(eci_pos=slice(0, 3), eci_vel=slice(3, 6), euler=slice(6, 9),
           vUVW=slice(9, 12), alpha=12, beta=13, mach=14, Vt=15, alt_asl=16)


class GpuDogfight:
    # 커널은 순수 double 연산이라 255 레지스터 하드캡에 걸려 occupancy 17%.
    # 드라이버 JIT 레지스터 상한(CU_JIT_MAX_REGISTERS)으로 occupancy↑ → 처리량↑.
    # RTX 3070 Ti 스윕상 fp64=96(ss1 +24%/ss4 +12%), fp32=128 이 최적.
    # 레지스터 캡은 수치 불변(배치만 변경). None=드라이버 기본(255).
    DEFAULT_MAX_REGISTERS = {"fp64": 96, "fp32": 128}

    def __init__(self, nenv, substeps=1, planes_per_env=2, block=128,
                 precision="fp64", max_registers="auto"):
        """precision: 'fp64'(검증된 bit-정합 기본) 또는 'fp32'(소비자 GPU 에서 2.8~6.8x,
        단 위치가 ECI 절대좌표 FP32 해상한계로 ~43ft/6s 드리프트; 자세/속도/받음각은
        FP64 수준). RL 보상이 상대기하 수백~수천 ft 스케일이면 fp32 로 대량가속 권장."""
        assert torch.cuda.is_available()
        assert precision in ("fp64", "fp32")
        torch.zeros(1, device="cuda")  # primary ctx
        self.nenv = nenv
        self.ppe = planes_per_env
        self.nac = nenv * planes_per_env
        self.substeps = substeps
        self.block = block
        self.precision = precision
        self.dtype = torch.float32 if precision == "fp32" else torch.float64
        if max_registers == "auto":
            max_registers = self.DEFAULT_MAX_REGISTERS[precision]
        cap = torch.cuda.get_device_capability()
        arch = f"compute_{cap[0]}{cap[1]}"
        src = cuda_rt.assemble_source(
            (GEN / "fdm_kernel.cu").read_text(encoding="utf-8"),
            (GEN / "fdm.cuh").read_text(encoding="utf-8"),
            (GEN / "f16_gen.cuh").read_text(encoding="utf-8"))
        if precision == "fp32":
            src = cuda_rt.to_fp32(src)   # 검증된 FP64 소스를 기계변환(단일 진실원본)
        ptx = cuda_rt.compile_ptx(src, arch)
        self.kern = cuda_rt.Kernel(ptx, "fdm_step_batch",
                                   max_registers=(max_registers or 0))
        self.states = torch.zeros(self.nac, STATE_N, dtype=self.dtype, device="cuda")
        self.obs = torch.zeros(self.nac, OBS_N, dtype=self.dtype, device="cuda")
        self.actions = torch.zeros(self.nac, 4, dtype=self.dtype, device="cuda")

    # ---- reset ----
    def load_seed(self, seed):
        """seed: (nac,101) 또는 (101,) 브로드캐스트 또는 (ppe,101)/(nenv,ppe,101).
        torch.Tensor 또는 numpy 허용."""
        t = torch.as_tensor(seed, dtype=torch.float64)
        if t.ndim == 1:
            t = t.unsqueeze(0).expand(self.nac, -1)
        elif t.ndim == 3:
            t = t.reshape(self.nac, STATE_N)
        elif t.shape[0] != self.nac:
            # (ppe,101) -> 각 env 동일 배치
            t = t.unsqueeze(0).expand(self.nenv, -1, -1).reshape(self.nac, STATE_N)
        self.states.copy_(t.to("cuda"))

    def reset_ic(self, ic_list):
        """ic_list: nac 개(또는 (nenv,ppe) 중첩) IC dict. 각 dict 는 ic.build_seed_vector kwargs
        (lat_deg,lon_deg,alt_ft,vt_fps,gamma_deg,phi_deg,psi_deg,theta_deg,alpha_deg,beta_deg,
         fuel_lbs,throttle,lat_type). CPU(ref)로 seed 계산 후 업로드."""
        import numpy as np
        from cuda_fdm.ic import build_seed_vector
        flat = []
        for item in ic_list:
            if isinstance(item, dict):
                flat.append(item)
            else:
                flat.extend(item)  # (ppe,) 중첩
        assert len(flat) == self.nac, f"{len(flat)} != nac {self.nac}"
        seeds = np.array([build_seed_vector(**ic) for ic in flat], dtype=np.float64)
        self.load_seed(seeds)

    # ---- step ----
    def step(self, actions, substeps=None):
        """actions: (nac,4) 또는 (nenv,ppe,4) [aileron,elevator,rudder,throttle].
        반환 obs (nenv,ppe,OBS_N) (cuda 텐서 view)."""
        ss = self.substeps if substeps is None else substeps
        a = torch.as_tensor(actions, dtype=torch.float64, device="cuda").reshape(self.nac, 4)
        self.actions.copy_(a)
        grid = ((self.nac + self.block - 1) // self.block, 1, 1)
        args = [C.c_void_p(self.states.data_ptr()), C.c_void_p(self.actions.data_ptr()),
                C.c_int(self.nac), C.c_int(ss), C.c_void_p(self.obs.data_ptr())]
        self.kern.launch(grid, (self.block, 1, 1), args)
        return self.obs.view(self.nenv, self.ppe, OBS_N)

    def get_state(self):
        return self.states.view(self.nenv, self.ppe, STATE_N)

    def get_obs(self):
        return self.obs.view(self.nenv, self.ppe, OBS_N)
