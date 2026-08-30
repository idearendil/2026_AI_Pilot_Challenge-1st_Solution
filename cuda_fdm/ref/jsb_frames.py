# -*- coding: utf-8 -*-
"""ECI/ECEF/Local/Body 프레임 변환 + WGS84 측지 + J2 중력 + 사원수
(FGLocation/FGInertial/FGQuaternion/FGMatrix33). double, JSBSim 연산순서/규약 준수.
행렬은 row-major list[3][3]; data[k]=m[k//3][k%3] (JSBSim 1-indexed (i,j)=data[(i-1)*3+(j-1)])."""
import math
from .jsb_lin import transpose

# WGS84 (FGInertial.cpp)
SEMI_MAJOR = 20925646.32546
SEMI_MINOR = 20855486.5951
GM = 14.0764417572E15
J2 = 1.08262982E-03
ROTATION_RATE = 0.00007292115
OMEGA = (0.0, 0.0, ROTATION_RATE)
EC = SEMI_MINOR / SEMI_MAJOR
EC2 = EC * EC
E2 = 1.0 - EC2
CCONST = SEMI_MAJOR * E2


def sign(x):
    return 1.0 if x >= 0.0 else -1.0


def Ti2ec_from_epa(epa):
    ce, se = math.cos(epa), math.sin(epa)
    return [[ce, se, 0.0], [-se, ce, 0.0], [0.0, 0.0, 1.0]]


def location_derived(ecef):
    x, y, z = ecef
    radius = math.sqrt(x * x + y * y + z * z)
    rxy = math.sqrt(x * x + y * y)
    if rxy == 0.0:
        sinLon, cosLon = 0.0, 1.0
    else:
        sinLon, cosLon = y / rxy, x / rxy
    if radius == 0.0:
        sinLat, cosLat = 0.0, 1.0
    else:
        sinLat, cosLat = z / radius, rxy / radius
    mLon = 0.0 if (x == 0.0 and y == 0.0) else math.atan2(y, x)
    mLat = 0.0 if (rxy == 0.0 and z == 0.0) else math.atan2(z, rxy)
    Tec2l = [[-cosLon * sinLat, -sinLon * sinLat, cosLat],
             [-sinLon, cosLon, 0.0],
             [-cosLon * cosLat, -sinLon * cosLat, -sinLat]]
    Tl2ec = transpose(Tec2l)
    a = SEMI_MAJOR
    s0 = abs(z)
    zc = EC * s0
    c0 = EC * rxy
    c02 = c0 * c0
    s02 = s0 * s0
    a02 = c02 + s02
    a0 = math.sqrt(a02)
    a03 = a02 * a0
    s1 = zc * a03 + CCONST * s02 * s0
    c1 = rxy * a03 - CCONST * c02 * c0
    cs0c0 = CCONST * c0 * s0
    b0 = 1.5 * cs0c0 * ((rxy * s0 - zc * c0) * a0 - cs0c0)
    s1 = s1 * a03 - b0 * s0
    cc = EC * (c1 * a03 - b0 * c0)
    mGeodLat = sign(z) * math.atan(s1 / cc)
    s12 = s1 * s1
    cc2 = cc * cc
    GeodeticAltitude = (rxy * cc + s0 * s1 - a * math.sqrt(EC2 * s12 + cc2)) / math.sqrt(s12 + cc2)
    return dict(mLat=mLat, mLon=mLon, mGeodLat=mGeodLat, GeodeticAltitude=GeodeticAltitude,
                radius=radius, Tec2l=Tec2l, Tl2ec=Tl2ec)


def gravity_j2(ecef, latitude):
    x, y, z = ecef
    r = math.sqrt(x * x + y * y + z * z)
    sinLat = math.sin(latitude)
    adivr = SEMI_MAJOR / r
    preCommon = 1.5 * J2 * adivr * adivr
    xy = 1.0 - 5.0 * (sinLat * sinLat)
    zz = 3.0 - 5.0 * (sinLat * sinLat)
    GMOverr2 = GM / (r * r)
    return (-GMOverr2 * ((1.0 + (preCommon * xy)) * x / r),
            -GMOverr2 * ((1.0 + (preCommon * xy)) * y / r),
            -GMOverr2 * ((1.0 + (preCommon * zz)) * z / r))


# ---------- 사원수 (FGQuaternion) ----------
def quat_to_T(q):
    q0, q1, q2, q3 = q
    q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    q0q1, q0q2, q0q3 = q0 * q1, q0 * q2, q0 * q3
    q1q2, q1q3, q2q3 = q1 * q2, q1 * q3, q2 * q3
    return [
        [q0q0 + q1q1 - q2q2 - q3q3, 2.0 * (q1q2 + q0q3), 2.0 * (q1q3 - q0q2)],
        [2.0 * (q1q2 - q0q3), q0q0 - q1q1 + q2q2 - q3q3, 2.0 * (q2q3 + q0q1)],
        [2.0 * (q1q3 + q0q2), 2.0 * (q2q3 - q0q1), q0q0 - q1q1 - q2q2 + q3q3],
    ]


def euler_to_quat(phi, tht, psi):
    thtd2, psid2, phid2 = 0.5 * tht, 0.5 * psi, 0.5 * phi
    St, Sp, Sf = math.sin(thtd2), math.sin(psid2), math.sin(phid2)
    Ct, Cp, Cf = math.cos(thtd2), math.cos(psid2), math.cos(phid2)
    CfCt, CfSt, SfSt, SfCt = Cf * Ct, Cf * St, Sf * St, Sf * Ct
    q = (CfCt * Cp + SfSt * Sp,
         SfCt * Cp - CfSt * Sp,
         CfSt * Cp + SfCt * Sp,
         CfCt * Sp - SfSt * Cp)
    return quat_normalize(q)


def mat_to_quat(m):
    # FGMatrix33 data[]는 column-major: data[k]=m(i,j), i-1=k%3, j-1=k//3
    d = [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]]
    tempQ = [1.0 + d[0] + d[4] + d[8], 1.0 + d[0] - d[4] - d[8],
             1.0 - d[0] + d[4] - d[8], 1.0 - d[0] - d[4] + d[8]]
    idx = 0
    for i in range(1, 4):
        if tempQ[i] > tempQ[idx]:
            idx = i
    Q = [0.0, 0.0, 0.0, 0.0]
    if idx == 0:
        Q[0] = 0.5 * math.sqrt(tempQ[0])
        Q[1] = 0.25 * (d[7] - d[5]) / Q[0]
        Q[2] = 0.25 * (d[2] - d[6]) / Q[0]
        Q[3] = 0.25 * (d[3] - d[1]) / Q[0]
    elif idx == 1:
        Q[1] = 0.5 * math.sqrt(tempQ[1])
        Q[0] = 0.25 * (d[7] - d[5]) / Q[1]
        Q[2] = 0.25 * (d[3] + d[1]) / Q[1]
        Q[3] = 0.25 * (d[2] + d[6]) / Q[1]
    elif idx == 2:
        Q[2] = 0.5 * math.sqrt(tempQ[2])
        Q[0] = 0.25 * (d[2] - d[6]) / Q[2]
        Q[1] = 0.25 * (d[3] + d[1]) / Q[2]
        Q[3] = 0.25 * (d[7] + d[5]) / Q[2]
    else:
        Q[3] = 0.5 * math.sqrt(tempQ[3])
        Q[0] = 0.25 * (d[3] - d[1]) / Q[3]
        Q[1] = 0.25 * (d[6] + d[2]) / Q[3]
        Q[2] = 0.25 * (d[7] + d[5]) / Q[3]
    return tuple(Q)


def mat_to_euler(m):
    """FGMatrix33.GetEuler → (phi,theta,psi). data[] column-major:
    data[6]=m(1,3)=m[0][2], data[7]=m(2,3)=m[1][2], data[8]=m(3,3)=m[2][2],
    data[5]=m(3,2)=m[2][1], data[4]=m(2,2)=m[1][1], data[3]=m(1,2)=m[0][1], data[0]=m(1,1)=m[0][0]."""
    d6 = m[0][2]
    gimbal = False
    if d6 <= -1.0:
        theta = 0.5 * math.pi
        gimbal = True
    elif 1.0 <= d6:
        theta = -0.5 * math.pi
        gimbal = True
    else:
        theta = math.asin(-d6)
    if gimbal:
        phi = math.atan2(-m[2][1], m[1][1])
        psi = 0.0
    else:
        phi = math.atan2(m[1][2], m[2][2])
        psi = math.atan2(m[0][1], m[0][0])
        if psi < 0.0:
            psi += 2 * math.pi
    return (phi, theta, psi)


def quat_mult(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def quat_qdot(q, pqr):
    p, qq, r = pqr
    return (-0.5 * (q[1] * p + q[2] * qq + q[3] * r),
            0.5 * (q[0] * p - q[3] * qq + q[2] * r),
            0.5 * (q[3] * p + q[0] * qq - q[1] * r),
            0.5 * (-q[2] * p + q[1] * qq + q[0] * r))


def quat_normalize(q):
    mag = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if mag == 0.0:
        return q
    inv = 1.0 / mag
    return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


def quat_add_scaled(q, qdot, dt):
    return (q[0] + dt * qdot[0], q[1] + dt * qdot[1],
            q[2] + dt * qdot[2], q[3] + dt * qdot[3])
