# -*- coding: utf-8 -*-
"""F-16 질량/관성/CG (FGMassBalance) double 복제. 원본 f16.xml mass_balance + FGMassBalance.cpp.
연료는 내부탱크 2개 등분배유 가정(총연료/2 each). 외부탱크 0."""
from .jsb_const import LBTOSLUG, INCHTOFT

# f16.xml mass_balance / metrics
EMPTY_WT = 17400.0
BASE_CG = (-193.0, 0.0, -5.1)          # in (structural)
PILOT_WT = 230.0
PILOT_LOC = (-336.2, 0.0, 0.0)
# baseJ (v1.0.0 규약: FGMassBalance ReadInertiaMatrix -> (bixx,-bixy,bixz / -bixy,biyy,-biyz / bixz,-biyz,bizz))
BIXX, BIYY, BIZZ, BIXY, BIXZ, BIYZ = 9496.0, 55814.0, 63100.0, 0.0, -982.0, 0.0
BASEJ = [[BIXX, -BIXY, BIXZ],
         [-BIXY, BIYY, -BIYZ],
         [BIXZ, -BIYZ, BIZZ]]
TANK0_LOC = (-174.4, 65.0, 5.0)
TANK1_LOC = (-174.4, -65.0, 5.0)
AERORP = (-189.5, 0.0, 3.9)            # metrics AERORP (in)


def _pm_inertia(mass_sl, r, cg):
    """GetPointmassInertia(mass_sl, r) with StructuralToBody(r) about current cg."""
    vx = INCHTOFT * (cg[0] - r[0])
    vy = INCHTOFT * (r[1] - cg[1])
    vz = INCHTOFT * (cg[2] - r[2])
    sx, sy, sz = mass_sl * vx, mass_sl * vy, mass_sl * vz
    xx = sx * vx
    yy = sy * vy
    zz = sz * vz
    xy = -sx * vy
    xz = -sx * vz
    yz = -sy * vz
    return [[yy + zz, xy, xz],
            [xy, xx + zz, yz],
            [xz, yz, xx + yy]]


def _madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]


def structural_to_body(r, cg):
    return (INCHTOFT * (cg[0] - r[0]),
            INCHTOFT * (r[1] - cg[1]),
            INCHTOFT * (cg[2] - r[2]))


def compute(fuel_total_lbs):
    """반환 dict: weight, mass_slug, cg(in), Ixx..Iyz, Jinv(3x3), RPBody(ft)."""
    t0 = t1 = fuel_total_lbs / 2.0
    tanks_wt = t0 + t1
    weight = EMPTY_WT + tanks_wt + PILOT_WT
    # moments (structural in * lbs)
    def mom(w, loc):
        return (w * loc[0], w * loc[1], w * loc[2])
    m_empty = mom(EMPTY_WT, BASE_CG)
    m_pilot = mom(PILOT_WT, PILOT_LOC)
    m_t0 = mom(t0, TANK0_LOC)
    m_t1 = mom(t1, TANK1_LOC)
    cg = tuple((m_empty[i] + m_pilot[i] + m_t0[i] + m_t1[i]) / weight for i in range(3))
    # inertia
    mJ = [row[:] for row in BASEJ]
    mJ = _madd(mJ, _pm_inertia(LBTOSLUG * EMPTY_WT, BASE_CG, cg))
    mJ = _madd(mJ, _pm_inertia(LBTOSLUG * PILOT_WT, PILOT_LOC, cg))
    mJ = _madd(mJ, _pm_inertia(LBTOSLUG * t0, TANK0_LOC, cg))
    mJ = _madd(mJ, _pm_inertia(LBTOSLUG * t1, TANK1_LOC, cg))
    Ixx = mJ[0][0]; Iyy = mJ[1][1]; Izz = mJ[2][2]
    Ixy = -mJ[0][1]; Ixz = -mJ[0][2]; Iyz = -mJ[1][2]
    # inverse (Stevens & Lewis)
    k1 = (Iyy * Izz - Iyz * Iyz)
    k2 = (Iyz * Ixz + Ixy * Izz)
    k3 = (Ixy * Iyz + Iyy * Ixz)
    denom = 1.0 / (Ixx * k1 - Ixy * k2 - Ixz * k3)
    k1 *= denom; k2 *= denom; k3 *= denom
    k4 = (Izz * Ixx - Ixz * Ixz) * denom
    k5 = (Ixy * Ixz + Iyz * Ixx) * denom
    k6 = (Ixx * Iyy - Ixy * Ixy) * denom
    Jinv = [[k1, k2, k3], [k2, k4, k5], [k3, k5, k6]]
    J = [[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]]
    RPBody = structural_to_body(AERORP, cg)
    return dict(weight=weight, mass_slug=LBTOSLUG * weight, cg=cg,
                Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz,
                J=J, Jinv=Jinv, RPBody=RPBody)
