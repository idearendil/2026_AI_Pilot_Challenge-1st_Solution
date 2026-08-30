# -*- coding: utf-8 -*-
"""IC (대회 f16_init.xml 파라미터) → 초기 FdmState seed args.
lat/lon/alt/vt/gamma/phi/psi/theta/alpha/beta → (eci_pos,eci_vel,euler,vPQR,epa).
검증된 ref 프레임(jsb_frames)과 동일 규약으로 구성 → ref FDM.seed 에 투입.

reset 은 에피소드당 1회이므로 CPU(ref) 로 계산 후 GPU 업로드(성능 무관).
"""
import math
from cuda_fdm.ref import jsb_frames as FR
from cuda_fdm.ref import jsb_const as C
from cuda_fdm.ref.jsb_lin import matvec, matmul, transpose, cross
from cuda_fdm.ref.jsb_atmos import StandardAtmosphere

_atmos = StandardAtmosphere()
D2R = math.pi / 180.0


def geodetic_to_ecef(lat_gd, lon, alt_asl):
    """WGS84 geodetic 방향(lat_gd,lon) + 반경조건 |ecef| = SEMI_MAJOR + alt_asl 로 ECEF(ft).
    JSBSim v1.0.0 이 golden 에서 쓰는 규약: altitudeASL = |eci| − a, <latitude>=geodetic.
    geodetic 법선 위 점 X=(N+t)cosφcosλ, Y=(N+t)cosφsinλ, Z=(N(1-e²)+t)sinφ 에서
    |ecef|=a+alt 되는 t 를 풀어 사용(golden eci_pos 를 9e-9 ft 재현)."""
    a = FR.SEMI_MAJOR
    e2 = FR.E2
    sphi, cphi = math.sin(lat_gd), math.cos(lat_gd)
    slam, clam = math.sin(lon), math.cos(lon)
    N = a / math.sqrt(1.0 - e2 * sphi * sphi)
    A = cphi * cphi
    B = sphi * sphi
    P = N
    Q = N * (1.0 - e2)
    b = A * P + B * Q
    cst = A * P * P + B * Q * Q
    R = a + alt_asl
    t = -b + math.sqrt(b * b - cst + R * R)
    x = (N + t) * cphi * clam
    y = (N + t) * cphi * slam
    z = (N * (1.0 - e2) + t) * sphi
    return (x, y, z)


def geocentric_to_ecef(lat_gc, lon, radius):
    clat, slat = math.cos(lat_gc), math.sin(lat_gc)
    clon, slon = math.cos(lon), math.sin(lon)
    return (radius * clat * clon, radius * clat * slon, radius * slat)


def ic_to_seedargs(lat_deg, lon_deg, alt_ft, vt_fps,
                   gamma_deg=0.0, phi_deg=0.0, psi_deg=0.0, theta_deg=0.0,
                   alpha_deg=0.0, beta_deg=0.0, lat_type="geodetic", epa=0.0):
    """반환 dict(eci_pos,eci_vel,euler,vPQR,epa). ref FDM.seed 에 넣을 수 있게."""
    lat = lat_deg * D2R
    lon = lon_deg * D2R
    phi, psi = phi_deg * D2R, psi_deg * D2R
    theta = theta_deg * D2R
    alpha, beta = alpha_deg * D2R, beta_deg * D2R

    if lat_type == "geodetic":
        ecef = geodetic_to_ecef(lat, lon, alt_ft)
    else:
        ecef = geocentric_to_ecef(lat, lon, FR.SEMI_MAJOR + alt_ft)

    Ti2ec = FR.Ti2ec_from_epa(epa)
    Tec2i = transpose(Ti2ec)
    eci_pos = matvec(Tec2i, ecef)   # epa=0 이면 = ecef

    # 자세: euler(phi,theta,psi) → Ti2b (ref seed 와 동일 체인)
    loc = FR.location_derived(ecef)
    Tl2i = matmul(Tec2i, loc["Tl2ec"])
    Ti2l = transpose(Tl2i)
    qL = FR.euler_to_quat(phi, theta, psi)
    Tl2b = FR.quat_to_T(qL)
    Ti2b = matmul(Tl2b, Ti2l)
    Tb2i = transpose(Ti2b)

    # body velocity from vt,alpha,beta (FGIC wind->body)
    u = vt_fps * math.cos(alpha) * math.cos(beta)
    v = vt_fps * math.sin(beta)
    w = vt_fps * math.sin(alpha) * math.cos(beta)
    vUVW = (u, v, w)
    # inertial velocity: vUVW = Ti2b·(eci_vel - omega×eci_pos)
    #   => eci_vel = Tb2i·vUVW + omega×eci_pos
    omega = FR.OMEGA
    eci_vel = tuple(matvec(Tb2i, vUVW)[i] + cross(omega, eci_pos)[i] for i in range(3))

    euler = (phi, theta, psi)
    vPQR = (0.0, 0.0, 0.0)
    return dict(eci_pos=eci_pos, eci_vel=eci_vel, euler=euler, vPQR=vPQR, epa=epa,
                vUVW=vUVW, alt_ft=alt_ft, vt=vt_fps, alpha=alpha, beta=beta)


def _flatten_state(fdm):
    """seed 된 ref FDM → fdm.cuh FdmState 순서의 101 double (export_seed 와 동일)."""
    v = []
    v += list(fdm.eci_pos); v += list(fdm.eci_vel); v += list(fdm.q); v += list(fdm.vPQRi)
    v += [fdm.epa, fdm.fuel_total]
    v += list(fdm.in_vUVWidot); v += list(fdm.in_vPQRidot); v += list(fdm.vQtrndot)
    for k in range(3): v += list(fdm.deq_q[k])
    for k in range(3): v += list(fdm.deq_pqri[k])
    for k in range(3): v += list(fdm.deq_ivel[k])
    for k in range(3): v += list(fdm.deq_uvwidot[k])
    v += list(fdm.prev_vBodyAccel); v += list(fdm.prev_vPQRidot); v += list(fdm.prev_vPQRi)
    pa = fdm.prev_aux
    v += [pa["alpha_rad"], pa["mach"], pa["vc_kts"], pa["vg_fps"],
          pa["n_pilot_y"], pa["n_pilot_z"], pa["p_aero"], pa["q_aero"], pa["r_aero"]]
    e = fdm.engine
    v += [e.N1, e.N2, e.N2norm, e.FuelFlow_pph]
    fc = fdm.fcs
    v += [fc.k_tef.output, fc.k_aileron.output, fc.k_elevator.output,
          fc.k_rudder.output, fc.k_gear.output, fc.k_lef.output]
    for pid in (fc.pid_roll, fc.pid_gload, fc.pid_yaw):
        v += [pid.input_prev, pid.input_prev2, pid.i_out_total]
    assert len(v) == 101, len(v)
    return v


def build_seed_vector(lat_deg, lon_deg, alt_ft, vt_fps,
                      gamma_deg=0.0, phi_deg=0.0, psi_deg=0.0, theta_deg=0.0,
                      alpha_deg=0.0, beta_deg=0.0, lat_type="geodetic",
                      fuel_lbs=6000.0, throttle=0.8):
    """IC → 101 double FdmState seed 벡터 (GPU 업로드용). CPU(ref) 로 계산.
    prev_aux 는 IC-일관 근사(alpha/mach/vc 정확, n_pilot 는 1프레임 근사)."""
    from cuda_fdm.ref.jsb_fdm import FDM
    from cuda_fdm.ref.jsb_auxiliary import vcalibrated_from_mach
    sa = ic_to_seedargs(lat_deg, lon_deg, alt_ft, vt_fps, gamma_deg, phi_deg, psi_deg,
                        theta_deg, alpha_deg, beta_deg, lat_type)
    at = _atmos.calculate(alt_ft)   # alt_asl = |eci|-a = alt_ft
    mach = vt_fps / at["a"]
    vcas = vcalibrated_from_mach(mach, at["P"]) if abs(mach) > 0 else 0.0
    prev_aux = dict(alpha_rad=sa["alpha"], mach=mach, vc_kts=vcas * C.FPSTOKTS,
                    vg_fps=vt_fps, n_pilot_y=0.0, n_pilot_z=-1.0,
                    p_aero=0.0, q_aero=0.0, r_aero=0.0)
    # IdleFF/Idle 에서 시작하는 게 아니라 순항추력 근사: fuelflow 는 seek 로 수렴하므로 0 시작 OK
    fdm = FDM()
    fdm.seed(eci_pos=sa["eci_pos"], eci_vel=sa["eci_vel"], euler=sa["euler"],
             vPQR=sa["vPQR"], epa=sa["epa"], fuel=fuel_lbs,
             fuelflow_pph=0.0, prev_aux=prev_aux)   # ic_uvwdot 없음 → throwaway 힘경로 시드
    return _flatten_state(fdm)
