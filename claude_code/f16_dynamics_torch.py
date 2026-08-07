# -*- coding: utf-8 -*-
"""reduced_predictor.cpp 의 F-16 reduced 6DoF 동역학을 배치 vectorized torch 로 포팅.

목적: 대회 추론 시 GPU 에서 여러 rollout 을 배치로 굴려 model-based 계획(neural MPC)에
쓰기 위함. C++ 원본(Release_MPC_team_share/native/reduced_predictor)의 reset/step 을
가능한 한 충실히 옮겼다(1-frame FCS latency, midpoint 적분, 공력 40항, FCS PID/actuator/
engine spool 포함). autograd 불필요(forward-only) — no_grad 로 쓰면 된다.

정밀도: 기본 float32(GPU 실용). parity 검증은 float64 로도 가능(DLL double 과 ~1e-6 일치 목표).

계약(모두 배치 B):
  m = TorchReducedF16(header_path, device, dtype, batch=B)
  m.reset(public_states)          # (B,17): [N,E,D(m), roll,pitch,yaw(deg), u,v,w(m/s),
                                  #          p,q,r(rad/s), sim_time, last_roll/pitch/rudder/throttle]
  for _ in range(H): m.step(controls)   # controls (B,4): roll,pitch,rudder,throttle
  out = m.public_state()          # (B,17) 동일 레이아웃
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

_DEF_HEADER = (Path(__file__).resolve().parents[1]
               / "Release_MPC_team_share" / "native" / "reduced_predictor"
               / "generated_f16_data.h")

# ── 물리 상수 (reduced_predictor.cpp) ──────────────────────────────────────
_PI = 3.14159265358979323846
_DEG2RAD = _PI / 180.0
_RAD2DEG = 180.0 / _PI
_FT2M = 0.3048
_M2FT = 1.0 / _FT2M
_LBF2N = 4.4482216152605
_LBFT2NM = 1.3558179483314
_PA2PSF = 0.02088543423315
_G = 9.80665
_DT = 1.0 / 60.0


def _load_data(header_path):
    """generated_f16_data.h 에서 모든 constexpr 배열/스칼라를 파싱."""
    text = Path(header_path).read_text(encoding="utf-8", errors="replace")
    arrays, scalars = {}, {}
    for name, body in re.findall(r"inline constexpr double (\w+)\[\]\s*=\s*\{([^}]*)\}", text):
        arrays[name] = [float(x) for x in body.replace("\n", " ").split(",") if x.strip()]
    for name, val in re.findall(r"inline constexpr double (\w+)\s*=\s*([-\d.eE+]+)\s*;", text):
        scalars[name] = float(val)
    return arrays, scalars


class TorchReducedF16:
    def __init__(self, header_path=None, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        arrays, scalars = _load_data(header_path or _DEF_HEADER)
        self.S = scalars
        # 테이블 텐서화(상수). 이름 그대로 보관.
        self.T = {k: torch.tensor(v, device=self.device, dtype=dtype) for k, v in arrays.items()}
        # FCS 내부 schedule1 (inline 배열) 상수.
        t = lambda v: torch.tensor(v, device=self.device, dtype=dtype)
        self._mach_pts, self._mach_gain = t([0.0, 1.0]), t([1.0, 0.15])
        self._alpha_pts = t([-0.5236, -0.5, 0.0, 0.5, 0.5236])
        self._alpha_gain = t([0.0, 0.11, 1.0, 0.11, 0.0])
        self._spd_pts, self._yaw_gain = t([80.0, 100.0, 150.0]), t([0.0, 15.0, 100.0])
        # 상수 텐서(hot-path 에서 torch.tensor 재생성 금지 — CUDA graph 캡처 중 host→device 복사 불가).
        self._conj_sign = t([1.0, -1.0, -1.0, -1.0])
        self._arm = t([scalars["aero_arm_x_m"], scalars["aero_arm_y_m"], scalars["aero_arm_z_m"]])

    # ── 배치 보간 ──────────────────────────────────────────────────────────
    def _interp1(self, name, q):
        x = self.T[name + "_rows"]; y = self.T[name + "_values"]; n = x.numel()
        hi = torch.clamp(torch.searchsorted(x, q, right=True), 1, n - 1)
        lo = hi - 1
        xl = x[lo]; xh = x[hi]
        tt = torch.clamp((q - xl) / (xh - xl), 0.0, 1.0)
        return y[lo] + tt * (y[hi] - y[lo])

    def _interp2(self, name, qr, qc):
        rows = self.T[name + "_rows"]; cols = self.T[name + "_cols"]
        nr = rows.numel(); nc = cols.numel()
        vals = self.T[name + "_values"].reshape(nr, nc)
        r1 = torch.clamp(torch.searchsorted(rows, qr, right=False) - 1, 0, nr - 1)
        c1 = torch.clamp(torch.searchsorted(cols, qc, right=False) - 1, 0, nc - 1)
        r2 = torch.clamp(r1 + 1, max=nr - 1); c2 = torch.clamp(c1 + 1, max=nc - 1)
        drr = rows[r2] - rows[r1]; dcc = cols[c2] - cols[c1]
        tr = torch.where(r1 == r2, torch.zeros_like(qr),
                         torch.clamp((qr - rows[r1]) / torch.where(drr == 0, torch.ones_like(drr), drr), 0.0, 1.0))
        tc = torch.where(c1 == c2, torch.zeros_like(qc),
                         torch.clamp((qc - cols[c1]) / torch.where(dcc == 0, torch.ones_like(dcc), dcc), 0.0, 1.0))
        v11 = vals[r1, c1]; v12 = vals[r1, c2]; v21 = vals[r2, c1]; v22 = vals[r2, c2]
        a = v11 + tc * (v12 - v11); b = v21 + tc * (v22 - v21)
        return a + tr * (b - a)

    # ── 쿼터니언/벡터 헬퍼 (배치) ─────────────────────────────────────────
    @staticmethod
    def _qmul(a, b):
        aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw], dim=-1)

    @staticmethod
    def _qunit(q):
        n = torch.linalg.norm(q, dim=-1, keepdim=True)
        ident = torch.zeros_like(q); ident[..., 0] = 1.0
        return torch.where((n < 1e-12) | ~torch.isfinite(n), ident, q / n)

    def _euler_to_quat(self, roll, pitch, yaw):
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        q = torch.stack([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                         cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr], dim=-1)
        return self._qunit(q)

    @staticmethod
    def _quat_to_euler(q):
        w, x, y, z = q.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw

    @staticmethod
    def _rot_b2n(q, v):
        w, x, y, z = q.unbind(-1); vx, vy, vz = v.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return torch.stack([
            (1 - 2 * (yy + zz)) * vx + 2 * (xy - wz) * vy + 2 * (xz + wy) * vz,
            2 * (xy + wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz,
            2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz], dim=-1)

    def _rot_n2b(self, q, v):
        return self._rot_b2n(q * self._conj_sign, v)

    # ── atmosphere / flight condition ────────────────────────────────────
    def _atmos(self, alt_m):
        h = torch.clamp(alt_m, -1000.0, 30000.0)
        lo = h <= 11000.0
        T = torch.where(lo, 288.15 - 0.0065 * h, torch.full_like(h, 216.65))
        p = torch.where(lo, 101325.0 * (torch.clamp(T, min=1e-3) / 288.15) ** 5.2558797,
                        22632.06 * torch.exp(-_G * (h - 11000.0) / (287.05287 * 216.65)))
        dens = p / (287.05287 * T)
        return dens, dens / 1.225, torch.sqrt(1.4 * 287.05287 * T)

    def _flight_cond(self, pos, quat, vel):
        vx, vy, vz = vel.unbind(-1)
        speed = torch.clamp(torch.linalg.norm(vel, dim=-1), min=1.0)
        alpha = torch.atan2(vz, torch.clamp(vx, min=1e-6))
        beta = torch.asin(torch.clamp(vy / speed, -1.0, 1.0))
        alt = -pos[..., 2]
        dens, dratio, sound = self._atmos(alt)
        mach = speed / sound
        cas = speed * 1.94384449244 * torch.sqrt(dratio)
        nv = self._rot_b2n(quat, vel)
        gs_fps = torch.hypot(nv[..., 0], nv[..., 1]) * _M2FT
        return dict(speed=speed, alpha=alpha, beta=beta, alt=alt, dens=dens,
                    mach=mach, cas=cas, gs_fps=gs_fps)

    # ── FCS (functional: fcs dict + control → new fcs dict + surfaces) ────
    def _update_fcs(self, fcs, control, cond, dt):
        f = dict(fcs)  # copy
        roll = torch.clamp(control[..., 0], -1.0, 1.0)
        pitch = torch.clamp(control[..., 1], -1.0, 1.0)
        rudder = torch.clamp(control[..., 2], -1.0, 1.0)
        throttle = torch.clamp(control[..., 3], 0.0, 1.0)
        ox, oy, oz = self._omega.unbind(-1)
        er, ep, ey = self._quat_to_euler(self._quat)

        roll_err = roll - 0.31821 * ox
        f["roll_i"] = torch.clamp(f["roll_i"] + (1.5 * roll_err - 0.5 * f["prev_roll_e"]) * dt, -20.0, 20.0)
        roll_d = (roll_err - f["prev_roll_e"]) / dt
        roll_pid = 3.0 * roll_err + 0.00050 * f["roll_i"] - 0.00125 * roll_d
        roll_cmd = torch.clamp(roll_pid + roll, -1.0, 1.0)
        f["prev_roll_e"] = roll_err
        f["ail_n"] = self._move(f["ail_n"], roll_cmd, (2.0 / 0.3) * dt)
        speed_comp = f["ail_n"] * self._interp1_arr(self._mach_pts, self._mach_gain, cond["mach"])

        lim_pitch = torch.clamp(pitch, -1.0, 0.44)
        elev_sched = lim_pitch * self._interp1_arr(self._alpha_pts, self._alpha_gain, cond["alpha"])
        corrected_g = f["prev_nz"] - torch.cos(ep) * torch.cos(er)
        pitch_err = elev_sched + 6.2 * oy - 0.020 * corrected_g
        f["pitch_i"] = torch.clamp(f["pitch_i"] + (1.5 * pitch_err - 0.5 * f["prev_pitch_e"]) * dt, -20.0, 20.0)
        pitch_pid = torch.clamp(0.3000 * pitch_err + 0.0250 * f["pitch_i"], -1.0, 1.0)
        pitch_tgt = torch.clamp(elev_sched + 1.0472 * cond["alpha"] + pitch_pid, -1.0, 1.0)
        f["prev_pitch_e"] = pitch_err
        f["elev_n"] = self._move(f["elev_n"], pitch_tgt, (2.0 / 0.3) * dt)

        yaw_err = (rudder + self._interp1_arr(self._spd_pts, self._yaw_gain, cond["gs_fps"]) * oz
                   + 0.25 * f["prev_ny"])
        f["yaw_i"] = torch.clamp(f["yaw_i"] + (1.5 * yaw_err - 0.5 * f["prev_yaw_e"]) * dt, -20.0, 20.0)
        yaw_d = (yaw_err - f["prev_yaw_e"]) / dt
        yaw_pid = torch.clamp(0.105500 * yaw_err + 0.000010 * f["yaw_i"] + 0.00005 * yaw_d, -1.0, 1.0)
        yaw_tgt = torch.clamp(rudder + yaw_pid, -1.0, 1.0)
        f["prev_yaw_e"] = yaw_err
        f["rud_n"] = self._move(yaw_pid, yaw_tgt, (2.0 / 0.4) * dt)  # 원본: PID 출력에서 시작

        tef = torch.where(cond["mach"] > 0.9, torch.full_like(cond["mach"], -0.0349),
                          torch.where(cond["cas"] < 250.0, torch.full_like(cond["mach"], 0.349),
                                      torch.zeros_like(cond["mach"])))
        f["tef_n"] = self._move(f["tef_n"], torch.clamp(tef * 2.864789, -1.0, 1.0), (2.0 / 3.0) * dt)
        a = cond["alpha"]
        lef = torch.where(cond["mach"] > 0.9, torch.full_like(a, -0.0349),
                          torch.where(a > 0.2618, torch.full_like(a, 0.436),
                                      torch.where(a > 0.0873, torch.full_like(a, 0.262), torch.zeros_like(a))))
        f["lef_n"] = self._move(f["lef_n"], torch.clamp(lef * 2.293578, -1.0, 1.0), (2.0 / 3.0) * dt)

        thr_pos = 2.0 * throttle
        core_tgt = torch.clamp(thr_pos, max=1.0)
        spool_base = 90.0 / (0.4 + 3.0)
        n = torch.clamp(f["eng_n2"] + 0.1, max=1.0)
        denom = 1.0 + 3.0 * (1.0 - n) ** 3 + (1.0 - cond["dens"] / 1.225)
        spool_fac = torch.where(core_tgt >= f["eng_n2"], torch.ones_like(core_tgt), torch.full_like(core_tgt, 3.0))
        spool_rate = spool_fac * spool_base / torch.clamp(denom, min=0.25) / 47.0
        f["eng_n2"] = self._move(f["eng_n2"], core_tgt, spool_rate * dt)
        alt_ft = cond["alt"] * _M2FT
        idle = self.S["mil_thrust_lbf"] * self._interp2("engine_IdleThrust", cond["mach"], alt_ft)
        mil = (self.S["mil_thrust_lbf"] - idle) * self._interp2("engine_MilThrust", cond["mach"], alt_ft)
        thrust = idle + mil * f["eng_n2"] * f["eng_n2"]
        augment = torch.clamp(thr_pos - 1.0, 0.0, 1.0)
        maximum = self.S["max_thrust_lbf"] * self._interp2("engine_AugThrust", cond["mach"], alt_ft)
        thrust = torch.where(augment > 0.0, thrust + augment * (maximum - thrust), thrust)

        left_fl = torch.clamp(-f["tef_n"] - speed_comp, -1.0, 1.0)
        right_fl = torch.clamp(f["tef_n"] - speed_comp, -1.0, 1.0)
        out = dict(
            aileron_rad=0.375 * roll_cmd, elevator_rad=0.436 * f["elev_n"],
            rudder_rad=0.524 * f["rud_n"], lef_rad=lef,
            flaperon_mix_rad=1.4324 * (left_fl + right_fl),
            speedbrake_rad=torch.zeros_like(thrust), thrust_n=thrust * _LBF2N)
        return f, out

    @staticmethod
    def _move(value, target, max_delta):
        return value + torch.clamp(target - value, -max_delta, max_delta)

    @staticmethod
    def _interp1_arr(x, y, q):
        n = x.numel()
        hi = torch.clamp(torch.searchsorted(x, q, right=True), 1, n - 1)
        lo = hi - 1
        tt = torch.clamp((q - x[lo]) / (x[hi] - x[lo]), 0.0, 1.0)
        return y[lo] + tt * (y[hi] - y[lo])

    # ── 공력 (evaluateAero, 40항 그대로) ─────────────────────────────────
    def _aero(self, cond, omega, fcs_out, speed_fps):
        qbar = 0.5 * cond["dens"] * cond["speed"] ** 2 * _PA2PSF
        area = self.S["wing_area_sqft"]; span = self.S["wing_span_ft"]; chord = self.S["mean_chord_ft"]
        a = cond["alpha"]; b = cond["beta"]; mach = cond["mach"]
        ox, oy, oz = omega.unbind(-1)
        lef = fcs_out["lef_rad"]; flap = fcs_out["flaperon_mix_rad"]; sb = fcs_out["speedbrake_rad"]
        ail = fcs_out["aileron_rad"]; rud = fcs_out["rudder_rad"]; elev = fcs_out["elevator_rad"]
        c2v = chord / (2.0 * speed_fps); s2v = span / (2.0 * speed_fps)
        ge = self._interp1("ground_effect", torch.clamp(cond["alt"] * _M2FT / chord, min=0.0))
        qA = qbar * area
        i1 = self._interp1; i2 = self._interp2
        drag = (qA * i2("aero_table_0_aero_coefficient_CDDh", a, elev)
                + qA * i1("aero_table_1_aero_coefficient_CDmach", mach)
                + qA * lef * i1("aero_table_2_aero_coefficient_CDDlef", a)
                + qA * flap * 0.08
                + qA * 0.0 * 0.027
                + qA * sb * i1("aero_table_5_aero_coefficient_CDDsb", a)
                + qA * oy * c2v * i1("aero_table_6_aero_coefficient_CDq", a)
                + qA * oy * c2v * lef * i1("aero_table_7_aero_coefficient_CDq_Dlef", a))
        side = (qA * b * -1.146
                + qA * b * i1("aero_table_9_aero_coefficient_CYb_M", mach)
                + qA * ail * -0.0226
                + qA * rud * 0.086
                + qA * s2v * ox * i1("aero_table_12_aero_coefficient_CYp", a)
                + qA * s2v * oz * i1("aero_table_13_aero_coefficient_CYr", a))
        lift = (qA * ge * i2("aero_table_14_aero_coefficient_CLDh", a, elev)
                + qA * lef * ge * i1("aero_table_15_aero_coefficient_CLDlef", a)
                + qA * flap * ge * 0.35
                + qA * ge * sb * i1("aero_table_17_aero_coefficient_CLDsb", a)
                + qA * oy * ge * c2v * i1("aero_table_18_aero_coefficient_CLq", a)
                + qA * oy * c2v * sb * i1("aero_table_19_aero_coefficient_CLq_Dsb", a))
        qAb = qA * span
        roll = (qAb * i2("aero_table_20_aero_coefficient_Clb", a, b)
                + qAb * b * i1("aero_table_21_aero_coefficient_Clb_M", mach)
                + qAb * s2v * ox * i1("aero_table_22_aero_coefficient_Clp", a)
                + qAb * s2v * oz * i1("aero_table_23_aero_coefficient_Clr", a)
                + qAb * ail * i2("aero_table_24_aero_coefficient_Clda", a, b)
                + qAb * a * ail * i1("aero_table_25_aero_coefficient_Clda_M", mach)
                + qAb * a * rud * i1("aero_table_26_aero_coefficient_Cldr_M", mach)
                + qAb * rud * i2("aero_table_27_aero_coefficient_Cldr", a, b))
        qAc = qA * chord
        pitch = (qAc * i2("aero_table_28_aero_coefficient_CmDh", a, elev)
                 + qAc * a * i1("aero_table_29_aero_coefficient_Cma_M", mach)
                 + qAc * sb * i1("aero_table_30_aero_coefficient_CmDsb", a)
                 + qAc * c2v * oy * i1("aero_table_31_aero_coefficient_Cmq", a))
        yaw = (qAb * i2("aero_table_32_aero_coefficient_Cnb", a, b)
               + qAb * b * i1("aero_table_33_aero_coefficient_Cnb_M", mach)
               + qAb * s2v * ox * i1("aero_table_34_aero_coefficient_Cnp", a)
               + qAb * s2v * oz * i1("aero_table_35_aero_coefficient_Cnr", a)
               + qAb * ail * i1("aero_table_36_aero_coefficient_Cnda_M", mach)
               + qAb * ail * i2("aero_table_37_aero_coefficient_Cnda", a, b)
               + qAb * rud * i2("aero_table_38_aero_coefficient_Cndr", a, b)
               + qAb * a * rud * i1("aero_table_39_aero_coefficient_Cndr_M", mach))
        return drag, side, lift, roll, pitch, yaw

    def _dynamics(self, pos, quat, vel, omega, fcs_out):
        cond = self._flight_cond(pos, quat, vel)
        speed_fps = torch.clamp(cond["speed"] * _M2FT, min=3.0)
        drag, side, lift, roll_m, pitch_m, yaw_m = self._aero(cond, omega, fcs_out, speed_fps)
        ca, sa = torch.cos(cond["alpha"]), torch.sin(cond["alpha"])
        cb, sb = torch.cos(cond["beta"]), torch.sin(cond["beta"])
        drag = drag * _LBF2N; side = side * _LBF2N; lift = lift * _LBF2N
        fx = -drag * ca * cb - side * ca * sb + lift * sa
        fy = -drag * sb + side * cb
        fz = -drag * sa * cb - side * sa * sb - lift * ca
        aero_force = torch.stack([fx, fy, fz], dim=-1)
        total_force = aero_force + torch.stack(
            [fcs_out["thrust_n"], torch.zeros_like(fx), torch.zeros_like(fx)], dim=-1)
        moment = torch.stack([roll_m * _LBFT2NM, pitch_m * _LBFT2NM, yaw_m * _LBFT2NM], dim=-1)
        moment = moment + torch.linalg.cross(self._arm.expand_as(aero_force), aero_force)
        grav_n = torch.zeros_like(vel); grav_n[..., 2] = _G
        grav_b = self._rot_n2b(quat, grav_n)
        mass = self.S["mass_kg"]
        vel_dot = total_force / mass + grav_b - torch.linalg.cross(omega, vel)
        omega_dot = self._inertia_solve(moment - torch.linalg.cross(omega, self._inertia_mul(omega)))
        pos_dot = self._rot_b2n(quat, vel)
        wq = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
        att_dot = self._qmul(quat, wq) * 0.5
        return dict(pos_dot=pos_dot, vel_dot=vel_dot, omega_dot=omega_dot,
                    att_dot=att_dot, aero_force=aero_force)

    def _inertia_mul(self, omega):
        ox, oy, oz = omega.unbind(-1); S = self.S
        return torch.stack([
            S["ixx_kgm2"] * ox + S["ixy_kgm2"] * oy + S["ixz_kgm2"] * oz,
            S["ixy_kgm2"] * ox + S["iyy_kgm2"] * oy + S["iyz_kgm2"] * oz,
            S["ixz_kgm2"] * ox + S["iyz_kgm2"] * oy + S["izz_kgm2"] * oz], dim=-1)

    def _inertia_solve(self, rhs):
        S = self.S
        a, b, c = S["ixx_kgm2"], S["ixy_kgm2"], S["ixz_kgm2"]
        d, e, f = S["iyy_kgm2"], S["iyz_kgm2"], S["izz_kgm2"]
        det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
        rx, ry, rz = rhs.unbind(-1)
        return torch.stack([
            ((d * f - e * e) * rx + (c * e - b * f) * ry + (b * e - c * d) * rz) / det,
            ((c * e - b * f) * rx + (a * f - c * c) * ry + (b * c - a * e) * rz) / det,
            ((b * e - c * d) * rx + (b * c - a * e) * ry + (a * d - b * b) * rz) / det], dim=-1)

    # ── reset / step / public_state ──────────────────────────────────────
    def _new_fcs(self, B, last_ctrl):
        z = torch.zeros(B, device=self.device, dtype=self.dtype)
        return dict(
            roll_i=z.clone(), pitch_i=z.clone(), yaw_i=z.clone(),
            prev_roll_e=z.clone(), prev_pitch_e=z.clone(), prev_yaw_e=z.clone(),
            ail_n=torch.clamp(last_ctrl[..., 0], -1.0, 1.0),
            elev_n=torch.clamp(last_ctrl[..., 1], -1.0, 1.0),
            rud_n=torch.clamp(last_ctrl[..., 2], -1.0, 1.0),
            tef_n=z.clone(), lef_n=z.clone(),
            eng_n2=torch.clamp(2.0 * torch.clamp(last_ctrl[..., 3], 0.0, 1.0), max=1.0),
            prev_nz=torch.ones(B, device=self.device, dtype=self.dtype), prev_ny=z.clone())

    def reset(self, public):
        p = torch.as_tensor(public, device=self.device, dtype=self.dtype)
        self._pos = p[..., 0:3].clone()
        self._quat = self._euler_to_quat(p[..., 3] * _DEG2RAD, p[..., 4] * _DEG2RAD, p[..., 5] * _DEG2RAD)
        self._vel = p[..., 6:9].clone()
        self._omega = p[..., 9:12].clone()
        self._sim_time = p[..., 12].clone()
        self._last_ctrl = p[..., 13:17].clone()
        self._fcs = self._new_fcs(p.shape[0], self._last_ctrl)
        # last_fcs_output: updateFcs 를 초기 fcs 복사본에 1회(원본 reset 과 동일, 상태는 안 advance).
        cond = self._flight_cond(self._pos, self._quat, self._vel)
        _, self._last_fcs = self._update_fcs(self._fcs, self._last_ctrl, cond, _DT)

    def step(self, control):
        c = torch.as_tensor(control, device=self.device, dtype=self.dtype)
        cond = self._flight_cond(self._pos, self._quat, self._vel)
        new_fcs, next_fcs = self._update_fcs(self._fcs, c, cond, _DT)
        self._fcs = new_fcs
        active = self._last_fcs
        first = self._dynamics(self._pos, self._quat, self._vel, self._omega, active)
        mp = self._pos + first["pos_dot"] * (0.5 * _DT)
        mv = self._vel + first["vel_dot"] * (0.5 * _DT)
        mo = self._omega + first["omega_dot"] * (0.5 * _DT)
        mq = self._qunit(self._quat + first["att_dot"] * (0.5 * _DT))
        middle = self._dynamics(mp, mq, mv, mo, active)
        self._pos = self._pos + middle["pos_dot"] * _DT
        self._vel = self._vel + middle["vel_dot"] * _DT
        self._omega = self._omega + middle["omega_dot"] * _DT
        self._quat = self._qunit(self._quat + middle["att_dot"] * _DT)
        self._sim_time = self._sim_time + _DT
        self._last_ctrl = torch.stack([
            torch.clamp(c[..., 0], -1.0, 1.0), torch.clamp(c[..., 1], -1.0, 1.0),
            torch.clamp(c[..., 2], -1.0, 1.0), torch.clamp(c[..., 3], 0.0, 1.0)], dim=-1)
        af = middle["aero_force"]
        self._fcs["prev_nz"] = -af[..., 2] / (self.S["mass_kg"] * _G)
        self._fcs["prev_ny"] = af[..., 1] / (self.S["mass_kg"] * _G)
        self._last_fcs = next_fcs

    def public_state(self):
        roll, pitch, yaw = self._quat_to_euler(self._quat)
        return torch.cat([
            self._pos, torch.stack([roll * _RAD2DEG, pitch * _RAD2DEG, yaw * _RAD2DEG], dim=-1),
            self._vel, self._omega, self._sim_time.unsqueeze(-1), self._last_ctrl], dim=-1)


__all__ = ["TorchReducedF16"]
