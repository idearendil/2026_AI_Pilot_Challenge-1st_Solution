# -*- coding: utf-8 -*-
"""통합 F16 FDM 단일기 단일스텝 (JSBSim v1.0.0 모델순서 재현).
Propagate(ECI적분)→Atmosphere→FCS→MassBalance→Auxiliary→Propulsion→Aero→Accelerations.
double. 이 루프의 로직이 곧 CUDA 커널 스펙.
"""
import math
from collections import deque
from . import jsb_frames as FR
from . import jsb_massbalance as MB
from .jsb_atmos import StandardAtmosphere
from .jsb_fcs import F16FCS
from .jsb_propulsion import F16Turbine
from .jsb_aero import F16Aero
from .jsb_auxiliary import auxiliary
from .jsb_accel import accelerations
from .jsb_lin import matmul, matvec, transpose, cross, vsub, vscale

DT = 1.0 / 60.0
SLGRAVITY = FR.GM / (FR.SEMI_MAJOR * FR.SEMI_MAJOR)   # gAccelReference
SEA_LEVEL_RADIUS = FR.SEMI_MAJOR                       # ref radius


def _integ_rect(integrand, val, deq, dt):
    deq.appendleft(val)
    deq.pop()
    return tuple(integrand[i] + dt * deq[0][i] for i in range(3))


def _integ_ab2(integrand, val, deq, dt):
    deq.appendleft(val)
    deq.pop()
    return tuple(integrand[i] + dt * (1.5 * deq[0][i] - 0.5 * deq[1][i]) for i in range(3))


def _integ_ab3(integrand, val, deq, dt):
    deq.appendleft(val)
    deq.pop()
    return tuple(integrand[i] + (dt / 12.0) * (23.0 * deq[0][i] - 16.0 * deq[1][i] + 5.0 * deq[2][i]) for i in range(3))


def _integ_quat_rect(q, val, deq, dt):
    deq.appendleft(val)
    deq.pop()
    qn = tuple(q[i] + dt * deq[0][i] for i in range(4))
    return FR.quat_normalize(qn)


class FDM:
    def __init__(self):
        self.atmos = StandardAtmosphere()
        self.aero = F16Aero()
        self.fcs = F16FCS()
        self.engine = F16Turbine()
        self.fuel_total = 0.0
        # ECI state
        self.eci_pos = (0.0, 0.0, 0.0)
        self.eci_vel = (0.0, 0.0, 0.0)
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.vPQRi = (0.0, 0.0, 0.0)
        self.epa = 0.0
        # carried derivatives
        self.in_vUVWidot = (0.0, 0.0, 0.0)
        self.in_vPQRidot = (0.0, 0.0, 0.0)
        self.vQtrndot = (0.0, 0.0, 0.0, 0.0)
        # deques
        self.deq_q = None
        self.deq_pqri = None
        self.deq_ivel = None
        self.deq_uvwidot = None
        # prev-frame auxiliary/accel (for FCS & n-pilot)
        self.prev_aux = None
        self.prev_vBodyAccel = (0.0, 0.0, 0.0)
        self.prev_vPQRidot = (0.0, 0.0, 0.0)
        self.prev_vPQRi = (0.0, 0.0, 0.0)

    # ---------- 힘경로 (Atmosphere..Accelerations), 적분 제외 ----------
    def _force_path(self, cmd, kin, fcs, engine, fuel, prev_aux, prev_ba, prev_pqridot, prev_pqri, advance_fuel=True):
        """kin: dict(vUVW,vPQR,vVel_ned,ecef,eci_pos,Ti2b,Tec2i,loc,euler).
        반환 dict(vUVWidot,vPQRidot,vBodyAccel,vPQRi,aux,fcs_out,eng,forces,moments,fuel_burn)."""
        alt_asl = kin["loc"]["radius"] - SEA_LEVEL_RADIUS
        at = self.atmos.calculate(alt_asl)
        mb = MB.compute(fuel)
        # FCS: prev aux + 현재프레임 attitude
        fcs_in = dict(prev_aux)
        fcs_in["pitch_rad"] = kin["euler"][1]
        fcs_in["roll_rad"] = kin["euler"][0]
        fcs_out = fcs.step(cmd, fcs_in, gear_pos_force=0.30)
        aux = auxiliary(kin["vUVW"], kin["vPQR"], kin["vVel_ned"], at["rho"], at["a"], at["P"],
                        mb["cg"], prev_ba, prev_pqridot, prev_pqri, SLGRAVITY)
        thr_pos = 2.0 * cmd["throttle"]
        eng = engine.step(thr_pos, aux["mach"], at["densalt"], at["T"], at["sigma"], DT, mb["cg"])
        st = {
            "Vt": aux["Vt"], "aero/qbar-psf": aux["qbar"],
            "aero/alpha-rad": aux["alpha"], "aero/beta-rad": aux["beta"],
            "aero/h_b-mac-ft": 765.0, "velocities/mach": aux["mach"],
            "velocities/p-aero-rad_sec": aux["p_aero"],
            "velocities/q-aero-rad_sec": aux["q_aero"],
            "velocities/r-aero-rad_sec": aux["r_aero"],
            "fcs/aileron-pos-rad": fcs_out["aileron_pos_rad"],
            "fcs/elevator-pos-rad": fcs_out["elevator_pos_rad"],
            "fcs/rudder-pos-rad": fcs_out["rudder_pos_rad"],
            "fcs/lef-pos-rad": fcs_out["lef_pos_rad"],
            "fcs/flaperon-mix-rad": fcs_out["flaperon_mix_rad"],
            "fcs/speedbrake-pos-rad": fcs_out["speedbrake_pos_rad"],
            "gear/gear-pos-norm": fcs_out["gear_pos_norm"],
        }
        fa, ma, _ = self.aero.compute(st, mb["RPBody"])
        force = (fa[0] + eng["forces"][0], fa[1] + eng["forces"][1], fa[2] + eng["forces"][2])
        moment = (ma[0] + eng["moments"][0], ma[1] + eng["moments"][1], ma[2] + eng["moments"][2])
        vGrav = matvec(kin["Tec2i"], FR.gravity_j2(kin["ecef"], kin["loc"]["mLat"]))
        acc = accelerations(force, moment, mb["mass_slug"], mb["J"], mb["Jinv"],
                            kin["vUVW"], kin["vPQR"], kin["Ti2b"], kin["eci_pos"], vGrav)
        acc["aux"] = aux
        acc["fcs_out"] = fcs_out
        acc["eng"] = eng
        acc["at"] = at
        acc["mb"] = mb
        acc["alt_asl"] = alt_asl
        return acc

    def _kinematics_from_state(self):
        """현재 ECI state → 파생 운동학 (frames, vUVW, vPQR, euler)."""
        Ti2ec = FR.Ti2ec_from_epa(self.epa)
        Tec2i = transpose(Ti2ec)
        ecef = matvec(Ti2ec, self.eci_pos)
        loc = FR.location_derived(ecef)
        Tl2i = matmul(Tec2i, loc["Tl2ec"])
        Ti2l = transpose(Tl2i)
        Ti2b = FR.quat_to_T(self.q)
        Tb2i = transpose(Ti2b)
        Tl2b = matmul(Ti2b, Tl2i)   # UpdateBodyMatrices: Tl2b = Ti2b · Tl2i
        Tb2l = transpose(Tl2b)
        omega = FR.OMEGA
        vUVW = matvec(Ti2b, vsub(self.eci_vel, cross(omega, self.eci_pos)))
        Ti2b_omega = matvec(Ti2b, omega)
        vPQR = vsub(self.vPQRi, Ti2b_omega)
        vVel_ned = matvec(Tb2l, vUVW)
        euler = FR.mat_to_euler(Tl2b)
        return dict(vUVW=vUVW, vPQR=vPQR, vVel_ned=vVel_ned, ecef=ecef, eci_pos=self.eci_pos,
                    Ti2b=Ti2b, Tec2i=Tec2i, loc=loc, euler=euler)

    def seed(self, eci_pos, eci_vel, euler, vPQR, epa, fuel, fuelflow_pph, prev_aux,
             ic_uvwdot=None, ic_pqrdot=None):
        """golden row0 로 시드. IC 파생값으로 deque 초기화.
        ic_uvwdot/ic_pqrdot: golden row0 body 미분(udot..,pdot..). 주면 그것으로 deque 시드
        (JSBSim run_ic 는 IC-eval 과도힘으로 미분을 계산해 deque에 넣으므로, 일관된 힘으로
        재계산하면 첫스텝이 어긋남 → golden IC 미분을 재구성해 정확히 맞춤). None이면 힘경로로 근사."""
        self.eci_pos = tuple(eci_pos)
        self.eci_vel = tuple(eci_vel)
        self.epa = epa
        self.fuel_total = fuel
        self.engine = F16Turbine()
        self.engine.FuelFlow_pph = fuelflow_pph
        self.prev_aux = dict(prev_aux)
        # qAttitudeECI 재구성: Ti2b = Tl2b·Ti2l
        Ti2ec = FR.Ti2ec_from_epa(epa)
        Tec2i = transpose(Ti2ec)
        ecef = matvec(Ti2ec, self.eci_pos)
        loc = FR.location_derived(ecef)
        Tl2i = matmul(Tec2i, loc["Tl2ec"])
        Ti2l = transpose(Tl2i)
        qL = FR.euler_to_quat(euler[0], euler[1], euler[2])
        Tl2b = FR.quat_to_T(qL)
        Ti2b = matmul(Tl2b, Ti2l)
        self.q = FR.mat_to_quat(Ti2b)
        omega = FR.OMEGA
        self.vPQRi = tuple(vPQR[i] + matvec(Ti2b, omega)[i] for i in range(3))
        # IC 파생 (throwaway FCS/engine 로 힘경로 1회)
        kin = self._kinematics_from_state()
        tmp_fcs = F16FCS()
        tmp_eng = F16Turbine()
        tmp_eng.FuelFlow_pph = fuelflow_pph
        acc = self._force_path(
            dict(aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.8,
                 pitch_trim=0.0, yaw_trim=0.0, gear=0.0),
            kin, tmp_fcs, tmp_eng, fuel, self.prev_aux,
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), self.vPQRi)
        self.in_vUVWidot = acc["vUVWidot"]
        self.in_vPQRidot = acc["vPQRidot"]
        self.vQtrndot = FR.quat_qdot(self.q, self.vPQRi)
        self.prev_vBodyAccel = acc["vBodyAccel"]
        self.prev_vPQRidot = acc["vPQRidot"]
        self.prev_vPQRi = acc["vPQRi"]
        # golden row0 body 미분이 주어지면 그것으로 IC 미분 재구성 (run_ic 과도힘 재현)
        if ic_uvwdot is not None:
            from .jsb_lin import vadd, vsub, vscale
            Ti2b = kin["Ti2b"]
            Tb2i = transpose(Ti2b)
            Ti2b_omega = matvec(Ti2b, omega)
            vUVW = kin["vUVW"]
            vPQR = kin["vPQR"]
            vGrav = matvec(kin["Tec2i"], FR.gravity_j2(kin["ecef"], kin["loc"]["mLat"]))
            centri = cross(omega, cross(omega, self.eci_pos))
            vBody = vadd(ic_uvwdot, cross(vadd(vPQR, vscale(Ti2b_omega, 2.0)), vUVW))
            vBody = vadd(vBody, matvec(Ti2b, centri))
            vBody = vsub(vBody, matvec(Ti2b, vGrav))
            self.in_vUVWidot = vadd(matvec(Tb2i, vBody), vGrav)
            self.in_vPQRidot = vadd(ic_pqrdot, cross(self.vPQRi, Ti2b_omega))
            self.prev_vBodyAccel = vBody
            self.prev_vPQRidot = self.in_vPQRidot
            self.prev_vPQRi = self.vPQRi
        # deque 초기화 (5x IC)
        self.deq_q = deque([self.vQtrndot] * 5, maxlen=5)
        self.deq_pqri = deque([self.in_vPQRidot] * 5, maxlen=5)
        self.deq_ivel = deque([self.eci_vel] * 5, maxlen=5)
        self.deq_uvwidot = deque([self.in_vUVWidot] * 5, maxlen=5)
        # fresh FCS/engine 로 스텝 시작
        self.fcs = F16FCS()
        self.engine = F16Turbine()
        self.engine.FuelFlow_pph = fuelflow_pph

    def step(self, aileron, elevator, rudder, throttle):
        cmd = dict(aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle,
                   pitch_trim=0.0, yaw_trim=0.0, gear=0.0)
        # --- Propagate: 적분 (순서 준수) ---
        self.q = _integ_quat_rect(self.q, self.vQtrndot, self.deq_q, DT)
        self.vPQRi = _integ_rect(self.vPQRi, self.in_vPQRidot, self.deq_pqri, DT)
        self.eci_pos = _integ_ab3(self.eci_pos, self.eci_vel, self.deq_ivel, DT)
        self.eci_vel = _integ_ab2(self.eci_vel, self.in_vUVWidot, self.deq_uvwidot, DT)
        self.epa += FR.ROTATION_RATE * DT
        # 프레임/운동학
        kin = self._kinematics_from_state()
        # vQtrndot (다음 스텝용)
        self.vQtrndot = FR.quat_qdot(self.q, self.vPQRi)
        # --- 힘경로 ---
        acc = self._force_path(cmd, kin, self.fcs, self.engine, self.fuel_total,
                               self.prev_aux, self.prev_vBodyAccel,
                               self.prev_vPQRidot, self.prev_vPQRi)
        self.fuel_total -= acc["eng"]["fuel_burn"]
        # 미분 갱신
        self.in_vUVWidot = acc["vUVWidot"]
        self.in_vPQRidot = acc["vPQRidot"]
        self.prev_vBodyAccel = acc["vBodyAccel"]
        self.prev_vPQRidot = acc["vPQRidot"]
        self.prev_vPQRi = acc["vPQRi"]
        aux = acc["aux"]
        self.prev_aux = dict(alpha_rad=aux["alpha"], mach=aux["mach"], vc_kts=aux["vc_kts"],
                             vg_fps=aux["vg_fps"], n_pilot_y=aux["n_pilot_y"],
                             n_pilot_z=aux["n_pilot_z"], p_aero=aux["p_aero"],
                             q_aero=aux["q_aero"], r_aero=aux["r_aero"])
        return dict(eci_pos=self.eci_pos, eci_vel=self.eci_vel, euler=kin["euler"],
                    vUVW=kin["vUVW"], vPQR=kin["vPQR"], alpha=aux["alpha"], beta=aux["beta"],
                    Vt=aux["Vt"], mach=aux["mach"], qbar=aux["qbar"], alt_asl=acc["alt_asl"],
                    loc=kin["loc"], thrust=acc["eng"]["thrust"], fuel=self.fuel_total)
