# -*- coding: utf-8 -*-
"""FGAccelerations (gravType=WGS84) double 복제.
입력: 힘/모멘트(body), Mass, J, Jinv, vUVW, vPQR, Ti2b, eci_pos, vGravAccel.
반환 vUVWdot(body), vPQRdot(body), vUVWidot(ECI), vPQRidot(body).
"""
from .jsb_lin import matvec, cross, vadd, vsub, vscale, transpose
from .jsb_frames import OMEGA


def accelerations(force, moment, mass, J, Jinv, vUVW, vPQR, Ti2b, eci_pos, vGravAccel):
    Tb2i = transpose(Ti2b)
    Ti2b_omega = matvec(Ti2b, OMEGA)
    # 각가속
    vPQRi = vadd(vPQR, Ti2b_omega)
    JvPQRi = matvec(J, vPQRi)
    vPQRidot = matvec(Jinv, vsub(moment, cross(vPQRi, JvPQRi)))
    vPQRdot = vsub(vPQRidot, cross(vPQRi, Ti2b_omega))
    # 병진가속
    vBodyAccel = vscale(force, 1.0 / mass)
    # -(vPQR + 2*Ti2b*omega) x vUVW
    term = cross(vadd(vPQR, vscale(Ti2b_omega, 2.0)), vUVW)
    vUVWdot = vsub(vBodyAccel, term)
    # - Ti2b*(omega x (omega x eci_pos))  (원심)
    centri = cross(OMEGA, cross(OMEGA, eci_pos))
    vUVWdot = vsub(vUVWdot, matvec(Ti2b, centri))
    # + Ti2b * vGravAccel
    vUVWdot = vadd(vUVWdot, matvec(Ti2b, vGravAccel))
    # 관성프레임 미분(적분용): vUVWidot = Tb2i*vBodyAccel + vGravAccel
    vUVWidot = vadd(matvec(Tb2i, vBodyAccel), vGravAccel)
    return dict(vUVWdot=vUVWdot, vPQRdot=vPQRdot, vUVWidot=vUVWidot,
                vPQRidot=vPQRidot, vBodyAccel=vBodyAccel, vPQRi=vPQRi)
