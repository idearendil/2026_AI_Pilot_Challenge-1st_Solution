# -*- coding: utf-8 -*-
"""NVRTC(런타임 컴파일) + CUDA Driver API 를 ctypes 로 구동하는 최소 런타임.
CUDA 툴킷/nvcc 불필요 — torch 번들 nvrtc64_130_0.dll + 드라이버 nvcuda.dll 사용.
메모리는 torch cuda 텐서(.data_ptr())로 관리, 여기선 모듈로드+커널런치만."""
import ctypes as C
import os
import glob
import re
import torch

_nvrtc = None
_cu = None


def _load_libs():
    global _nvrtc, _cu
    if _nvrtc is not None:
        return
    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    cands = sorted(glob.glob(os.path.join(libdir, "nvrtc64_*_0.dll")))
    cands = [c for c in cands if ".alt." not in c] or cands
    if not cands:
        raise RuntimeError("nvrtc dll not found in torch/lib")
    os.add_dll_directory(libdir)
    _nvrtc = C.CDLL(cands[0])
    _cu = C.CDLL("nvcuda.dll")


def _nvrtc_check(res):
    if res != 0:
        fn = _nvrtc.nvrtcGetErrorString
        fn.restype = C.c_char_p
        raise RuntimeError("NVRTC error: " + fn(res).decode())


def _cu_check(res, what=""):
    if res != 0:
        name = C.c_char_p()
        try:
            _cu.cuGetErrorName(res, C.byref(name))
            msg = name.value.decode() if name.value else str(res)
        except Exception:
            msg = str(res)
        raise RuntimeError(f"CUDA driver error {what}: {msg}")


def compile_ptx(src: str, arch: str, extra_opts=None) -> bytes:
    _load_libs()
    prog = C.c_void_p()
    _nvrtc_check(_nvrtc.nvrtcCreateProgram(
        C.byref(prog), src.encode(), b"fdm.cu", 0, None, None))
    opts = [f"--gpu-architecture={arch}".encode(), b"--fmad=false"]
    if extra_opts:
        opts += [o.encode() for o in extra_opts]
    arr = (C.c_char_p * len(opts))(*opts)
    res = _nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    # 로그
    logsz = C.c_size_t()
    _nvrtc.nvrtcGetProgramLogSize(prog, C.byref(logsz))
    log = C.create_string_buffer(logsz.value)
    _nvrtc.nvrtcGetProgramLog(prog, log)
    if res != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log.value.decode(errors="replace"))
    ptxsz = C.c_size_t()
    _nvrtc_check(_nvrtc.nvrtcGetPTXSize(prog, C.byref(ptxsz)))
    ptx = C.create_string_buffer(ptxsz.value)
    _nvrtc_check(_nvrtc.nvrtcGetPTX(prog, ptx))
    _nvrtc.nvrtcDestroyProgram(C.byref(prog))
    return ptx.raw


# CUjit_option enum (subset)
CU_JIT_MAX_REGISTERS = 0


class Kernel:
    def __init__(self, ptx: bytes, name: str, max_registers: int = 0):
        """max_registers>0 이면 드라이버 PTX->SASS JIT 시 레지스터/스레드 상한 지정
        (cuModuleLoadDataEx + CU_JIT_MAX_REGISTERS). occupancy 튜닝용."""
        _load_libs()
        # torch 가 이미 primary context 를 만들었다고 가정하고 current 로.
        _cu_check(_cu.cuInit(0), "cuInit")
        dev = C.c_int(0)
        _cu_check(_cu.cuDeviceGet(C.byref(dev), 0), "cuDeviceGet")
        ctx = C.c_void_p()
        _cu_check(_cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "PrimaryCtxRetain")
        _cu_check(_cu.cuCtxSetCurrent(ctx), "CtxSetCurrent")
        self.ctx = ctx
        mod = C.c_void_p()
        if max_registers > 0:
            opts = (C.c_int * 1)(CU_JIT_MAX_REGISTERS)
            # optionValues: void* 배열, MAX_REGISTERS 는 값 자체를 포인터로 캐스팅.
            vals = (C.c_void_p * 1)(C.c_void_p(max_registers))
            _cu_check(_cu.cuModuleLoadDataEx(
                C.byref(mod), ptx, 1, opts, vals), "ModuleLoadDataEx")
        else:
            _cu_check(_cu.cuModuleLoadData(C.byref(mod), ptx), "ModuleLoadData")
        self.mod = mod
        func = C.c_void_p()
        _cu_check(_cu.cuModuleGetFunction(C.byref(func), mod, name.encode()), "GetFunction")
        self.func = func

    def launch(self, grid, block, args):
        """args: list of (ctypes value). device 포인터는 c_void_p(data_ptr)."""
        n = len(args)
        params = (C.c_void_p * n)()
        keep = []
        for i, a in enumerate(args):
            keep.append(a)
            params[i] = C.cast(C.byref(a), C.c_void_p)
        _cu_check(_cu.cuLaunchKernel(
            self.func,
            C.c_uint(grid[0]), C.c_uint(grid[1]), C.c_uint(grid[2]),
            C.c_uint(block[0]), C.c_uint(block[1]), C.c_uint(block[2]),
            C.c_uint(0), C.c_void_p(0), params, None), "LaunchKernel")
        _cu_check(_cu.cuCtxSynchronize(), "CtxSync")


def assemble_source(kernel_cu: str, fdm_cuh: str, gen_cuh: str) -> str:
    """#include 를 인라인 스플라이스해서 NVRTC 단일 소스로."""
    fdm = fdm_cuh.replace('#include "f16_gen.cuh"', gen_cuh)
    ker = kernel_cu.replace('#include "fdm.cuh"', fdm)
    return ker


# float 리터럴(소수점 또는 지수 포함)에 f 접미사. 식별자/이미 접미사 제외.
# 예: 1.0->1.0f, .5->.5f, 1e-9->1e-9f, 60.0->60.0f. 정수(60)는 미변경.
_FLOAT_LIT = re.compile(r'(?<![\w.])(?:\d+\.\d*|\.\d+|\d+\.?\d*[eE][+-]?\d+)(?![\w.fF])')


def to_fp32(src: str) -> str:
    """검증된 FP64 소스를 FP32 로 기계변환: 모든 float 리터럴에 f 접미사(승격 방지)
    후 double->float. 소비자 GPU 에서 FP64(1/64 rate) 병목을 회피(측정 2.8~6.8x).
    FP64 원본은 그대로 두고 assemble 시점에만 적용 — 단일 진실원본 유지."""
    src = _FLOAT_LIT.sub(lambda m: m.group(0) + 'f', src)
    src = re.sub(r'\bdouble\b', 'float', src)
    return src
