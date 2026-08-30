# -*- coding: utf-8 -*-
"""검증 1: 표준대기 + Auxiliary(alpha/beta/qbar/mach) 를 golden CSV로 독립검증.
golden 의 고도/body속도 를 입력으로, rho/a/T/mach/qbar/alpha/beta/Vt 를 계산해
golden 컬럼과 비교. FGTable+atmosphere+auxiliary 로직 정합 확인.
"""
import sys
import csv
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_fdm.ref.jsb_atmos import StandardAtmosphere

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")


def main():
    atm = StandardAtmosphere()
    rows = list(csv.DictReader(open(GOLDEN)))
    cols = ["rho", "a", "T", "mach", "qbar", "alpha", "beta", "vt"]
    maxerr = {c: 0.0 for c in cols}
    for row in rows:
        h = float(row["position/h-sl-ft"])
        u = float(row["velocities/u-fps"])
        v = float(row["velocities/v-fps"])
        w = float(row["velocities/w-fps"])
        at = atm.calculate(h)
        Vt = math.sqrt(u * u + v * v + w * w)
        mUW = u * u + w * w
        beta = math.atan2(v, math.sqrt(mUW)) if Vt > 0.001 else 0.0
        alpha = math.atan2(w, u) if (Vt > 0.001 and mUW >= 1e-6) else 0.0
        qbar = 0.5 * at["rho"] * Vt * Vt
        mach = Vt / at["a"]
        got = dict(rho=at["rho"], a=at["a"], T=at["T"], mach=mach,
                   qbar=qbar, alpha=alpha, beta=beta, vt=Vt)
        ref = dict(rho=float(row["atmosphere/rho-slugs_ft3"]),
                   a=float(row["atmosphere/a-fps"]),
                   T=float(row["atmosphere/T-R"]),
                   mach=float(row["velocities/mach"]),
                   qbar=float(row["aero/qbar-psf"]),
                   alpha=float(row["aero/alpha-rad"]),
                   beta=float(row["aero/beta-rad"]),
                   vt=float(row["velocities/vt-fps"]))
        for c in cols:
            maxerr[c] = max(maxerr[c], abs(got[c] - ref[c]))
    print(f"rows={len(rows)}")
    print("=== max abs error (computed vs golden) ===")
    for c in cols:
        print(f"  {c:6s}: {maxerr[c]:.3e}")
    ok = (maxerr["rho"] < 1e-9 and maxerr["a"] < 1e-6 and maxerr["mach"] < 1e-6
          and maxerr["qbar"] < 1e-4 and maxerr["alpha"] < 1e-9 and maxerr["beta"] < 1e-9)
    print("RESULT:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
