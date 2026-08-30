# -*- coding: utf-8 -*-
"""out_c.bin (또는 out_gpu.bin) vs golden_ref.bin 비교 (val_fulltraj 와 동일 지표)."""
import sys, struct, math
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "_bin"


def load(name, n, cols):
    data = (OUTDIR / name).read_bytes()
    out = []
    for k in range(n):
        out.append(struct.unpack_from(f"<{cols}d", data, k * cols * 8))
    return out


def wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "out_c.bin"
    N = int((OUTDIR / "meta.txt").read_text())
    ref = load("golden_ref.bin", N, 10)
    got = load(out_name, N, 10)
    metrics = {"pos_ft": 0.0, "att_deg": 0.0, "vel_fps": 0.0, "alpha_deg": 0.0}
    argm = {k: -1 for k in metrics}
    for k in range(N):
        g, o = ref[k], got[k]
        dp = math.sqrt(sum((o[i] - g[i]) ** 2 for i in range(3)))
        da = max(abs(math.degrees(wrap(o[3 + i] - g[3 + i]))) for i in range(3))
        dv = max(abs(o[6 + i] - g[6 + i]) for i in range(3))
        dal = abs(math.degrees(o[9] - g[9]))
        for key, val in [("pos_ft", dp), ("att_deg", da), ("vel_fps", dv), ("alpha_deg", dal)]:
            if val > metrics[key]:
                metrics[key] = val
                argm[key] = k + 1
    print(f"{out_name} vs golden  (N={N})")
    for key in metrics:
        print(f"  {key:10s}: {metrics[key]:.4e}  @row {argm[key]}")


if __name__ == "__main__":
    main()
