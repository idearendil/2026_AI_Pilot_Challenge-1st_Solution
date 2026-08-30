# -*- coding: utf-8 -*-
"""검증 5(결정판): 전체 FDM 루프. golden row0 시드 → 6초 스텝 → golden 전 궤적 비교."""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_fdm.ref.jsb_fdm import FDM

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")


def main():
    rows = list(csv.DictReader(open(GOLDEN)))
    f = lambda r, c: float(r[c])
    r0 = rows[0]
    fdm = FDM()
    prev_aux = dict(
        alpha_rad=f(r0, "aero/alpha-rad"), mach=f(r0, "velocities/mach"),
        vc_kts=1000.0, vg_fps=f(r0, "velocities/vg-fps"),
        n_pilot_y=f(r0, "accelerations/n-pilot-y-norm"),
        n_pilot_z=f(r0, "accelerations/n-pilot-z-norm"),
        p_aero=f(r0, "velocities/p-aero-rad_sec"),
        q_aero=f(r0, "velocities/q-aero-rad_sec"),
        r_aero=f(r0, "velocities/r-aero-rad_sec"))
    fdm.seed(
        eci_pos=(f(r0, "position/eci-x-ft"), f(r0, "position/eci-y-ft"), f(r0, "position/eci-z-ft")),
        eci_vel=(f(r0, "velocities/eci-x-fps"), f(r0, "velocities/eci-y-fps"), f(r0, "velocities/eci-z-fps")),
        euler=(f(r0, "attitude/roll-rad"), f(r0, "attitude/pitch-rad"), f(r0, "attitude/heading-true-rad")),
        vPQR=(f(r0, "velocities/p-rad_sec"), f(r0, "velocities/q-rad_sec"), f(r0, "velocities/r-rad_sec")),
        epa=f(r0, "position/epa-rad"), fuel=f(r0, "propulsion/total-fuel-lbs"),
        fuelflow_pph=f(r0, "propulsion/engine/fuel-flow-rate-pps") * 3600.0, prev_aux=prev_aux,
        ic_uvwdot=(f(r0, "accelerations/udot-ft_sec2"), f(r0, "accelerations/vdot-ft_sec2"),
                   f(r0, "accelerations/wdot-ft_sec2")),
        ic_pqrdot=(f(r0, "accelerations/pdot-rad_sec2"), f(r0, "accelerations/qdot-rad_sec2"),
                   f(r0, "accelerations/rdot-rad_sec2")))

    def action_at(k):
        t = k / 60.0
        for te, a in [(1.5, [0, 0, 0, 0.8]), (3.0, [0, 0.3, 0, 0.8]),
                      (4.5, [0.3, 0, 0, 0.8]), (6.0, [0, 0, 0.3, 0.8])]:
            if t < te:
                return a
        return [0, 0, 0.3, 0.8]

    import math
    def wrap(d):
        return (d + math.pi) % (2 * math.pi) - math.pi

    metrics = {"pos_ft": 0.0, "att_deg": 0.0, "vel_fps": 0.0, "alpha_deg": 0.0}
    argm = {k: -1 for k in metrics}
    for k in range(len(rows) - 1):
        a = action_at(k)
        out = fdm.step(a[0], a[1], a[2], a[3])
        g = rows[k + 1]
        # 위치: ECI 오차
        dp = math.sqrt(sum((out["eci_pos"][i] - f(g, ["position/eci-x-ft", "position/eci-y-ft", "position/eci-z-ft"][i])) ** 2 for i in range(3)))
        # 자세 오차 (deg)
        da = max(abs(math.degrees(wrap(out["euler"][0] - f(g, "attitude/roll-rad")))),
                 abs(math.degrees(wrap(out["euler"][1] - f(g, "attitude/pitch-rad")))),
                 abs(math.degrees(wrap(out["euler"][2] - f(g, "attitude/heading-true-rad")))))
        # 속도 오차
        dv = max(abs(out["vUVW"][i] - f(g, ["velocities/u-fps", "velocities/v-fps", "velocities/w-fps"][i])) for i in range(3))
        dal = abs(math.degrees(out["alpha"] - f(g, "aero/alpha-rad")))
        for key, val in [("pos_ft", dp), ("att_deg", da), ("vel_fps", dv), ("alpha_deg", dal)]:
            if val > metrics[key]:
                metrics[key] = val
                argm[key] = k + 1
    print(f"rows={len(rows)}  전체 궤적 (내 FDM vs golden v1.0.0)")
    print("=== max 오차 ===")
    for key in metrics:
        print(f"  {key:10s}: {metrics[key]:.4e}  @row {argm[key]}")


if __name__ == "__main__":
    main()
