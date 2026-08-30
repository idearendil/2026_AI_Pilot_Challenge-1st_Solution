# -*- coding: utf-8 -*-
"""golden row0 로 FDM.seed 후 내부 FdmState(101 double)을 seed.bin 으로 내보냄.
+ actions.bin (N x 4), + golden_ref.bin (N x 10: eci_pos3,euler3,vUVW3,alpha) 를 만든다.
C(host/GPU) 가 이걸 읽어 step 을 돌리고 out 을 golden_ref 와 비교."""
import sys, csv, struct, math
from pathlib import Path

HERE = Path(__file__).resolve()
RELEASE = HERE.parents[2]
sys.path.insert(0, str(RELEASE))
from cuda_fdm.ref.jsb_fdm import FDM

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")
OUTDIR = HERE.parent / "_bin"
OUTDIR.mkdir(exist_ok=True)


def seed_fdm(rows):
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
    return fdm


def flatten_state(fdm):
    """FdmState 필드 순서(fdm.cuh)와 정확히 일치하는 101 double 리스트."""
    v = []
    v += list(fdm.eci_pos)                 # 3
    v += list(fdm.eci_vel)                 # 3
    v += list(fdm.q)                       # 4
    v += list(fdm.vPQRi)                   # 3
    v += [fdm.epa]                         # 1
    v += [fdm.fuel_total]                  # 1
    v += list(fdm.in_vUVWidot)             # 3
    v += list(fdm.in_vPQRidot)             # 3
    v += list(fdm.vQtrndot)                # 4
    for k in range(3): v += list(fdm.deq_q[k])       # 12
    for k in range(3): v += list(fdm.deq_pqri[k])    # 9
    for k in range(3): v += list(fdm.deq_ivel[k])    # 9
    for k in range(3): v += list(fdm.deq_uvwidot[k]) # 9
    v += list(fdm.prev_vBodyAccel)         # 3
    v += list(fdm.prev_vPQRidot)           # 3
    v += list(fdm.prev_vPQRi)              # 3
    pa = fdm.prev_aux
    v += [pa["alpha_rad"], pa["mach"], pa["vc_kts"], pa["vg_fps"],
          pa["n_pilot_y"], pa["n_pilot_z"], pa["p_aero"], pa["q_aero"], pa["r_aero"]]  # 9
    # EngState: N1,N2,N2norm,FuelFlow_pph
    e = fdm.engine
    v += [e.N1, e.N2, e.N2norm, e.FuelFlow_pph]  # 4
    # FcsState: 6 kinemat outputs + 3 PID(ip,ip2,iout)
    fc = fdm.fcs
    v += [fc.k_tef.output, fc.k_aileron.output, fc.k_elevator.output,
          fc.k_rudder.output, fc.k_gear.output, fc.k_lef.output]
    for pid in (fc.pid_roll, fc.pid_gload, fc.pid_yaw):
        v += [pid.input_prev, pid.input_prev2, pid.i_out_total]  # 15
    assert len(v) == 101, len(v)
    return v


def action_at(k):
    t = k / 60.0
    for te, a in [(1.5, [0, 0, 0, 0.8]), (3.0, [0, 0.3, 0, 0.8]),
                  (4.5, [0.3, 0, 0, 0.8]), (6.0, [0, 0, 0.3, 0.8])]:
        if t < te:
            return a
    return [0, 0, 0.3, 0.8]


def main():
    rows = list(csv.DictReader(open(GOLDEN)))
    f = lambda r, c: float(r[c])
    fdm = seed_fdm(rows)
    state = flatten_state(fdm)
    N = len(rows) - 1
    (OUTDIR / "seed.bin").write_bytes(struct.pack("<101d", *state))
    with open(OUTDIR / "actions.bin", "wb") as fp:
        for k in range(N):
            fp.write(struct.pack("<4d", *[float(x) for x in action_at(k)]))
    with open(OUTDIR / "golden_ref.bin", "wb") as fp:
        for k in range(1, N + 1):
            g = rows[k]
            row = [f(g, "position/eci-x-ft"), f(g, "position/eci-y-ft"), f(g, "position/eci-z-ft"),
                   f(g, "attitude/roll-rad"), f(g, "attitude/pitch-rad"), f(g, "attitude/heading-true-rad"),
                   f(g, "velocities/u-fps"), f(g, "velocities/v-fps"), f(g, "velocities/w-fps"),
                   f(g, "aero/alpha-rad")]
            fp.write(struct.pack("<10d", *row))
    (OUTDIR / "meta.txt").write_text(str(N))
    print(f"seed.bin(101d), actions.bin({N}x4), golden_ref.bin({N}x10) written. N={N}")


if __name__ == "__main__":
    main()
