# -*- coding: utf-8 -*-
"""FGAuxiliary double 복제 (alpha/beta/qbar/mach/p-q-r-aero/vground/n-pilot).
n-pilot 은 전프레임 Accelerations(vBodyAccel,vPQRidot,vPQRi) 사용.
vc-kts 는 스위치 임계(250/20/10/5)만 가르므로, 고속영역에선 큰 값(>250)으로 근사.
정밀 필요시 VcalibratedFromMach(PitotTotalPressure/MachFromImpactPressure) 포팅.
"""
import math
from .jsb_lin import cross, vadd, matvec
from .jsb_massbalance import structural_to_body
from . import jsb_const as C

EYEPOINT = (-336.2, 0.0, 29.5)   # metrics EYEPOINT (in)
# StdDaySLsoundspeed (FGAtmosphere.cpp), FPSTOKTS
STD_SL_SOUNDSPEED = math.sqrt(C.SHRatio * C.Reng * C.StdDaySLtemperature)


def pitot_total_pressure(mach, p):
    """FGJSBBase::PitotTotalPressure."""
    if mach < 0:
        return p
    if mach < 1:
        return p * (1 + 0.2 * mach * mach) ** 3.5
    return p * 166.92158009316827 * mach ** 7.0 / (7 * mach * mach - 1) ** 2.5


def mach_from_impact_pressure(qc, p):
    """FGJSBBase::MachFromImpactPressure."""
    A = qc / p + 1
    M = math.sqrt(5.0 * (A ** (1. / 3.5) - 1))
    if M > 1.0:
        for _ in range(10):
            M = 0.8812848543473311 * math.sqrt(A * (1 - 1.0 / (7.0 * M * M)) ** 2.5)
    return M


def vcalibrated_from_mach(mach, p):
    """FGJSBBase::VcalibratedFromMach → ft/s."""
    qc = pitot_total_pressure(mach, p) - p
    return STD_SL_SOUNDSPEED * mach_from_impact_pressure(qc, C.StdDaySLpressure)


def auxiliary(vUVW, vPQR, vVel_ned, rho, a, P, cg,
              prev_vBodyAccel, prev_vPQRidot, prev_vPQRi, SLGravity):
    u, v, w = vUVW
    AeroU2 = u * u
    AeroV2 = v * v
    AeroW2 = w * w
    mUW = AeroU2 + AeroW2
    Vt2 = mUW + AeroV2
    Vt = math.sqrt(Vt2)
    alpha = beta = 0.0
    if Vt > 0.001:
        beta = math.atan2(v, math.sqrt(mUW))
        if mUW >= 1e-6:
            alpha = math.atan2(w, u)
    qbar = 0.5 * rho * Vt2
    mach = Vt / a
    Vground = math.sqrt(vVel_ned[0] * vVel_ned[0] + vVel_ned[1] * vVel_ned[1])
    # n-pilot (전프레임 accel)
    ToEyePt = structural_to_body(EYEPOINT, cg)
    vPilotAccel = vadd(prev_vBodyAccel, cross(prev_vPQRidot, ToEyePt))
    vPilotAccel = vadd(vPilotAccel, cross(prev_vPQRi, cross(prev_vPQRi, ToEyePt)))
    n_pilot_y = vPilotAccel[1] / SLGravity
    n_pilot_z = vPilotAccel[2] / SLGravity
    # vc-kts (VcalibratedFromMach). abs(mach)>0 일때만, 아니면 0.
    if abs(mach) > 0.0:
        vcas = vcalibrated_from_mach(mach, P)      # ft/s
    else:
        vcas = 0.0
    vc_kts = vcas * C.FPSTOKTS
    return dict(alpha=alpha, beta=beta, Vt=Vt, qbar=qbar, mach=mach,
                p_aero=vPQR[0], q_aero=vPQR[1], r_aero=vPQR[2],
                vg_fps=Vground, n_pilot_y=n_pilot_y, n_pilot_z=n_pilot_z, vc_kts=vc_kts)
