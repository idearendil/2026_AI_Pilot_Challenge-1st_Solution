# -*- coding: utf-8 -*-
"""검증 4: Accelerations (+ECI프레임/사원수/측지/J2중력). golden 단일행.
golden 힘/모멘트(aero+prop) + 상태(uvw,pqr,euler,eci,epa)로 udot/pdot 재구성 → golden 비교.
Ti2b는 golden euler+ECI위치에서 재구성(Ti2b=Tl2b·Ti2l)."""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_fdm.ref import jsb_massbalance as MB
from cuda_fdm.ref import jsb_frames as FR
from cuda_fdm.ref.jsb_lin import matmul, matvec, transpose
from cuda_fdm.ref.jsb_accel import accelerations

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")


def main():
    rows = list(csv.DictReader(open(GOLDEN)))
    f = lambda r, c: float(r[c])
    err = {k: 0.0 for k in ["udot", "vdot", "wdot", "pdot", "qdot", "rdot"]}
    arg = dict(err)
    for k in range(1, len(rows)):
        r = rows[k]
        mb = MB.compute(f(r, "propulsion/total-fuel-lbs"))
        # 프레임 재구성
        epa = f(r, "position/epa-rad")
        eci = (f(r, "position/eci-x-ft"), f(r, "position/eci-y-ft"), f(r, "position/eci-z-ft"))
        Ti2ec = FR.Ti2ec_from_epa(epa)
        ecef = matvec(Ti2ec, eci)
        loc = FR.location_derived(ecef)
        Tec2i = transpose(Ti2ec)
        Tl2i = matmul(Tec2i, loc["Tl2ec"])
        Ti2l = transpose(Tl2i)
        qL = FR.euler_to_quat(f(r, "attitude/roll-rad"), f(r, "attitude/pitch-rad"),
                              f(r, "attitude/heading-true-rad"))
        Tl2b = FR.quat_to_T(qL)
        Ti2b = matmul(Tl2b, Ti2l)
        vGrav = matvec(Tec2i, FR.gravity_j2(ecef, loc["mLat"]))
        # 힘/모멘트 = golden aero+prop
        force = (f(r, "forces/fbx-aero-lbs") + f(r, "forces/fbx-prop-lbs"),
                 f(r, "forces/fby-aero-lbs") + f(r, "forces/fby-prop-lbs"),
                 f(r, "forces/fbz-aero-lbs") + f(r, "forces/fbz-prop-lbs"))
        moment = (f(r, "moments/l-aero-lbsft") + f(r, "moments/l-prop-lbsft"),
                  f(r, "moments/m-aero-lbsft") + f(r, "moments/m-prop-lbsft"),
                  f(r, "moments/n-aero-lbsft") + f(r, "moments/n-prop-lbsft"))
        vUVW = (f(r, "velocities/u-fps"), f(r, "velocities/v-fps"), f(r, "velocities/w-fps"))
        vPQR = (f(r, "velocities/p-rad_sec"), f(r, "velocities/q-rad_sec"), f(r, "velocities/r-rad_sec"))
        out = accelerations(force, moment, mb["mass_slug"], mb["J"], mb["Jinv"],
                            vUVW, vPQR, Ti2b, eci, vGrav)
        pairs = {
            "udot": (out["vUVWdot"][0], f(r, "accelerations/udot-ft_sec2")),
            "vdot": (out["vUVWdot"][1], f(r, "accelerations/vdot-ft_sec2")),
            "wdot": (out["vUVWdot"][2], f(r, "accelerations/wdot-ft_sec2")),
            "pdot": (out["vPQRdot"][0], f(r, "accelerations/pdot-rad_sec2")),
            "qdot": (out["vPQRdot"][1], f(r, "accelerations/qdot-rad_sec2")),
            "rdot": (out["vPQRdot"][2], f(r, "accelerations/rdot-rad_sec2")),
        }
        for key, (g, ref) in pairs.items():
            e = abs(g - ref)
            if e > err[key]:
                err[key] = e
                arg[key] = k
    print(f"rows={len(rows)}  (accel 재구성 vs golden)")
    for key in err:
        print(f"  {key:5s}: {err[key]:.3e}  @row {arg[key]}")


if __name__ == "__main__":
    main()
