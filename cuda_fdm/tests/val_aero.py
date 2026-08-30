# -*- coding: utf-8 -*-
"""검증 3: MassBalance(CG) + Aerodynamics(힘/모멘트). golden 단일행 함수로 검증.
모든 입력(alpha,beta,qbar,Vt,pqr-aero,조종면,gear,fuel)과 출력(forces/moments-aero)이 같은 row.
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_fdm.ref.jsb_aero import F16Aero
from cuda_fdm.ref import jsb_massbalance as MB

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")


def main():
    rows = list(csv.DictReader(open(GOLDEN)))
    aero = F16Aero()
    f = lambda r, c: float(r[c])
    err = {k: 0.0 for k in ["cg_x", "bi2vel", "ci2vel", "fbx", "fby", "fbz", "l", "m", "n"]}
    arg = {k: -1 for k in err}
    for k, r in enumerate(rows):
        mb = MB.compute(f(r, "propulsion/total-fuel-lbs"))
        st = {
            "Vt": f(r, "velocities/vt-fps"),
            "aero/qbar-psf": f(r, "aero/qbar-psf"),
            "aero/alpha-rad": f(r, "aero/alpha-rad"),
            "aero/beta-rad": f(r, "aero/beta-rad"),
            "aero/h_b-mac-ft": f(r, "aero/h_b-mac-ft"),
            "velocities/mach": f(r, "velocities/mach"),
            "velocities/p-aero-rad_sec": f(r, "velocities/p-aero-rad_sec"),
            "velocities/q-aero-rad_sec": f(r, "velocities/q-aero-rad_sec"),
            "velocities/r-aero-rad_sec": f(r, "velocities/r-aero-rad_sec"),
            "fcs/aileron-pos-rad": f(r, "fcs/aileron-pos-rad"),
            "fcs/elevator-pos-rad": f(r, "fcs/elevator-pos-rad"),
            "fcs/rudder-pos-rad": f(r, "fcs/rudder-pos-rad"),
            "fcs/lef-pos-rad": f(r, "fcs/lef-pos-rad"),
            "fcs/flaperon-mix-rad": f(r, "fcs/flaperon-mix-rad"),
            "fcs/speedbrake-pos-rad": f(r, "fcs/speedbrake-pos-rad"),
            "gear/gear-pos-norm": f(r, "gear/gear-pos-norm"),
        }
        forces, moments, dbg = aero.compute(st, mb["RPBody"])
        pairs = {
            "cg_x": (mb["cg"][0], f(r, "inertia/cg-x-in")),
            "bi2vel": (dbg["bi2vel"], f(r, "aero/bi2vel")),
            "ci2vel": (dbg["ci2vel"], f(r, "aero/ci2vel")),
            "fbx": (forces[0], f(r, "forces/fbx-aero-lbs")),
            "fby": (forces[1], f(r, "forces/fby-aero-lbs")),
            "fbz": (forces[2], f(r, "forces/fbz-aero-lbs")),
            "l": (moments[0], f(r, "moments/l-aero-lbsft")),
            "m": (moments[1], f(r, "moments/m-aero-lbsft")),
            "n": (moments[2], f(r, "moments/n-aero-lbsft")),
        }
        for key, (g, ref) in pairs.items():
            e = abs(g - ref)
            if e > err[key]:
                err[key] = e
                arg[key] = k
    print(f"rows={len(rows)}")
    print("=== max abs error (computed vs golden, per-row) ===")
    for key in err:
        print(f"  {key:8s}: {err[key]:.3e}  @row {arg[key]}")


if __name__ == "__main__":
    main()
