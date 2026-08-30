# -*- coding: utf-8 -*-
"""검증 2: F16 FCS. golden row k 의 (전프레임)Aux/Accel 값 + act 를 입력으로
FCS를 순차 실행, 조종면 출력을 golden row k+1 과 비교.
FCS는 상태보유(kinematic/PID) → frame 0부터 순차.
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_fdm.ref.jsb_fcs import F16FCS

GOLDEN = Path(r"C:\Users\idear\AppData\Local\Temp\claude"
              r"\D--AIP-LIB-claude-DogFightEnv-Release"
              r"\458f798e-bffd-41e0-9090-695280eb3f01\scratchpad\golden_trace.csv")

# 출력키 -> golden 컬럼
CMP = {
    "aileron_pos_rad": "fcs/aileron-pos-rad",
    "elevator_pos_rad": "fcs/elevator-pos-rad",
    "rudder_pos_rad": "fcs/rudder-pos-rad",
    "flaperon_mix_rad": "fcs/flaperon-mix-rad",
    "left_aileron_pos_norm": "fcs/left-aileron-pos-norm",
    "elevator_pos_norm": "fcs/elevator-pos-norm",
    "rudder_pos_norm": "fcs/rudder-pos-norm",
    "roll_rate_command": "fcs/roll-rate-command",
    "pitch_scheduler": "fcs/pitch-scheduler",
    "yaw_scheduler": "fcs/yaw-scheduler",
    "elevator_scheduler": "fcs/elevator-scheduler",
    "g_load_pid": "fcs/g-load-pid",
    "roll_rate_pid": "fcs/roll-rate-pid",
    "yaw_load_pid": "fcs/yaw-load-pid",
    "aileron_speed_compensated": "fcs/aileron-speed-compensated",
    "left_aileron_pos_rad": "fcs/left-aileron-pos-rad",
    "right_aileron_pos_rad": "fcs/right-aileron-pos-rad",
    "dht_left_pos_rad": "fcs/dht-left-pos-rad",
    "dht_right_pos_rad": "fcs/dht-right-pos-rad",
    "lef_pos_rad": "fcs/lef-pos-rad",
    "gear_pos_norm": "gear/gear-pos-norm",
}


def main():
    rows = list(csv.DictReader(open(GOLDEN)))
    N = len(rows)
    fcs = F16FCS()
    maxerr = {k: 0.0 for k in CMP}
    argmax = {k: -1 for k in CMP}
    for k in range(N - 1):
        r = rows[k]
        aux = dict(
            alpha_rad=float(r["aero/alpha-rad"]),
            mach=float(r["velocities/mach"]),
            vc_kts=float(r["velocities/vc-kts"]),
            vg_fps=float(r["velocities/vg-fps"]),
            n_pilot_y=float(r["accelerations/n-pilot-y-norm"]),
            n_pilot_z=float(r["accelerations/n-pilot-z-norm"]),
            p_aero=float(r["velocities/p-aero-rad_sec"]),
            q_aero=float(r["velocities/q-aero-rad_sec"]),
            r_aero=float(r["velocities/r-aero-rad_sec"]),
            # attitude 는 Propagate(idx0)가 FCS(idx5)보다 먼저 → 현재프레임(row k+1) 값
            pitch_rad=float(rows[k + 1]["attitude/pitch-rad"]),
            roll_rad=float(rows[k + 1]["attitude/roll-rad"]),
        )
        cmd = dict(aileron=float(r["act_roll"]), elevator=float(r["act_pitch"]),
                   rudder=float(r["act_rudder"]), throttle=float(r["act_throttle"]),
                   pitch_trim=0.0, yaw_trim=0.0, gear=0.0)
        out = fcs.step(cmd, aux, gear_pos_force=0.30)
        nr = rows[k + 1]
        for key, col in CMP.items():
            e = abs(out[key] - float(nr[col]))
            if e > maxerr[key]:
                maxerr[key] = e
                argmax[key] = k + 1
    print(f"rows={N}  (비교: out(k) vs golden row k+1)")
    print("=== max abs error ===")
    worst = 0.0
    for key, col in CMP.items():
        print(f"  {key:26s}: {maxerr[key]:.3e}  @row {argmax[key]}")
        worst = max(worst, maxerr[key])
    print(f"WORST = {worst:.3e}")
    print("RESULT:", "PASS" if worst < 1e-6 else "CHECK")


if __name__ == "__main__":
    main()
