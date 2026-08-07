# -*- coding: utf-8 -*-
"""claude_code.my_observation.build_observation(claude164r, 184-dim)의 배치 torch 포팅.

neural-MPC planner 의 rollout 에서 예측 state 로부터 actor/critic obs 를 GPU 배치로
빌드하기 위함. graph-capturable(정적 shape·branch-free). numpy build_observation 과 parity.

입력: own,tgt = (B, >=12) [N,E,D, roll,pit,yaw(deg), u,v,w(m/s), p,q,r(무관)], 그리고
reconstructor 상태 텐서 dict: t_sec,hp_own,hp_tgt,fuel_own,fuel_tgt,dmg_dealt,dmg_taken (B,),
own_pqr,tgt_pqr (B,3), act_hist (B,20).  출력: obs (B,184).
"""
from __future__ import annotations
import math
import torch

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi
_FT2M = 0.3048
_G = 9.80665
# 상수(my_observation 과 동일)
MAX_SPEED = 600.0; MAX_RANGE_M = 2500.0; MAX_CLOSURE = 1000.0; VS_SCALE = 100.0
AOA_SC = 30.0; SS_SC = 15.0; MIN_ALT = 300.0; ALT_DANGER = 300.0; MAX_ALT = 15000.0
EADV_SC = 5000.0; PQR_SC = 4.0; PURSUIT_ATA = 30.0; PURSUIT_RNG = 3000.0
EPISODE_MAX_T = 200.0; REL_VEL_MIN = -600.0; REL_VEL_MAX = 600.0
MIN_DMG_FT = 500.0
T2_START, T3_START = 100.0, 150.0
T1_CONE, T2_CONE, T3_CONE = 1.0, 2.0, 3.0
T1_MAXFT, T2_MAXFT, T3_MAXFT = 3000.0, 3500.0, 4000.0


def _norm(v, lo, hi):
    mid = (hi + lo) / 2.0; half = (hi - lo) / 2.0
    return (torch.clamp(v, lo, hi) - mid) / half


def _ned_to_body(euler_deg):
    """(B,3) roll,pitch,yaw deg → (B,3,3) R = Tx@Ty@Tz (my_observation/GeoMathUtil 동일)."""
    r = euler_deg[:, 0] * _D2R; p = euler_deg[:, 1] * _D2R; y = euler_deg[:, 2] * _D2R
    z = torch.zeros_like(r); o = torch.ones_like(r)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    Tx = torch.stack([o, z, z, z, cr, sr, z, -sr, cr], -1).reshape(-1, 3, 3)
    Ty = torch.stack([cp, z, -sp, z, o, z, sp, z, cp], -1).reshape(-1, 3, 3)
    Tz = torch.stack([cy, sy, z, -sy, cy, z, z, z, o], -1).reshape(-1, 3, 3)
    return torch.bmm(Tx, torch.bmm(Ty, Tz))


def _mv(R, v):   # (B,3,3)@(B,3)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)


def _dir_frame(x_ned):
    """(B,3) 방향 → (B,3,3) R (행=frame축). x=dir, z=down-proj, y=cross. 수직시 north fallback."""
    nx = torch.linalg.norm(x_ned, dim=-1, keepdim=True)
    x = x_ned / torch.clamp(nx, min=1e-8)
    down = torch.zeros_like(x); down[:, 2] = 1.0
    z = down - (x[:, 2:3]) * x                       # down·x = x_z
    zn = torch.linalg.norm(z, dim=-1, keepdim=True)
    north = torch.zeros_like(x); north[:, 0] = 1.0
    z_fb = north - (x[:, 0:1]) * x
    z = torch.where(zn < 1e-6, z_fb, z)
    z = z / torch.clamp(torch.linalg.norm(z, dim=-1, keepdim=True), min=1e-9)
    y = torch.linalg.cross(z, x); y = y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=1e-9)
    z = torch.linalg.cross(x, y)
    return torch.stack([x, y, z], dim=1)             # rows = frame axes


def _sincos(deg):
    r = deg * _D2R
    return r.sin(), r.cos()


def build_obs(own, tgt, rec):
    B = own.shape[0]; dtype = own.dtype
    own_pos = own[:, 0:3]; tgt_pos = tgt[:, 0:3]
    delta = tgt_pos - own_pos
    dist = torch.linalg.norm(delta, dim=-1)
    los_unit = delta / torch.clamp(dist.unsqueeze(-1), min=1e-6)

    Rob_own = _ned_to_body(own[:, 3:6]); Rob_tgt = _ned_to_body(tgt[:, 3:6])
    Rbn_own = Rob_own.transpose(1, 2); Rbn_tgt = Rob_tgt.transpose(1, 2)

    own_vb = own[:, 6:9]; tgt_vb = tgt[:, 6:9]
    own_vn = _mv(Rbn_own, own_vb); tgt_vn = _mv(Rbn_tgt, tgt_vb)
    rel_vn = tgt_vn - own_vn
    own_spd = torch.linalg.norm(own_vb, dim=-1); tgt_spd = torch.linalg.norm(tgt_vb, dim=-1)
    own_alt = -own_pos[:, 2]; tgt_alt = -tgt_pos[:, 2]

    # ── geometry (GeoMathUtil 포팅) ──
    pb_own = _mv(Rob_own, los_unit)                                  # LOS in own body
    ata = torch.arccos(torch.clamp(pb_own[:, 0], -1.0, 1.0)) * _R2D
    pb_tgt_en = _mv(Rob_tgt, -los_unit)                             # own dir in tgt body
    enemy_ata = torch.arccos(torch.clamp(pb_tgt_en[:, 0], -1.0, 1.0)) * _R2D
    # aspect angle: Tz_pi @ ned_to_body(tgt) @ (own-tgt unit)
    p_at = _mv(Rob_tgt, -los_unit)
    p_at = torch.stack([-p_at[:, 0], -p_at[:, 1], p_at[:, 2]], -1)   # Tz_pi=diag(-1,-1,1)
    aa_mag = torch.arccos(torch.clamp(p_at[:, 0], -1.0, 1.0)) * _R2D
    sign = torch.where(p_at[:, 1] < -0.10, -torch.ones_like(aa_mag),
                       torch.where((p_at[:, 1] > -0.01) & (p_at[:, 1] < 0.01),
                                   torch.sign(p_at[:, 2]), torch.ones_like(aa_mag)))
    aa = sign * aa_mag
    dis_b = _mv(Rob_own, los_unit)
    az = torch.arctan2(dis_b[:, 1], dis_b[:, 0]) * _R2D
    el = -torch.arcsin(torch.clamp(dis_b[:, 2], -1.0, 1.0)) * _R2D

    # ── scalars ──
    u, v, w = own_vb[:, 0], own_vb[:, 1], own_vb[:, 2]
    aoa = torch.where(own_spd < 1.0, torch.zeros_like(u), torch.arctan2(w, u) * _R2D)
    ss = torch.where(own_spd < 1.0, torch.zeros_like(u),
                     torch.arctan2(v, torch.sqrt(u * u + w * w)) * _R2D)
    vs = -own_vn[:, 2]
    own_eh = own_alt + own_spd ** 2 / (2 * _G); tgt_eh = tgt_alt + tgt_spd ** 2 / (2 * _G)
    eadv = own_eh - tgt_eh
    closure = ((own_vn - tgt_vn) * los_unit).sum(-1)

    t = rec["t_sec"]
    active_cone = torch.where(t >= T3_START, torch.full_like(t, T3_CONE),
                              torch.where(t >= T2_START, torch.full_like(t, T2_CONE), torch.full_like(t, T1_CONE)))
    active_maxft = torch.where(t >= T3_START, torch.full_like(t, T3_MAXFT),
                               torch.where(t >= T2_START, torch.full_like(t, T2_MAXFT), torch.full_like(t, T1_MAXFT)))
    aim_sharp = 2.0 * torch.exp(-((ata / 3.0) ** 2)) - 1.0
    aim_marg = torch.tanh((active_cone - ata.abs()) / torch.clamp(active_cone, min=1e-6))
    en_sharp = 2.0 * torch.exp(-((enemy_ata / 3.0) ** 2)) - 1.0
    en_marg = torch.tanh((active_cone - enemy_ata.abs()) / torch.clamp(active_cone, min=1e-6))
    min_dmg_m = MIN_DMG_FT * _FT2M; active_max_m = active_maxft * _FT2M
    span = torch.clamp(active_max_m - min_dmg_m, min=1e-6)
    rm_near = torch.tanh((dist - min_dmg_m) / span)
    rm_far = torch.tanh((active_max_m - dist) / span)

    orn = _sincos(own[:, 3]); opt = _sincos(own[:, 4]); oyw = _sincos(own[:, 5])
    trn = _sincos(tgt[:, 3]); tpt = _sincos(tgt[:, 4]); tyw = _sincos(tgt[:, 5])
    ovb = _bank(Rbn_own, own_vn); tvb = _bank(Rbn_tgt, tgt_vn)
    dd = rec["dmg_dealt"]; dt_ = rec["dmg_taken"]
    p_ata = torch.clamp(1.0 - ata.abs() / PURSUIT_ATA, min=0.0)
    p_rng = torch.clamp(1.0 - dist / PURSUIT_RNG, min=0.0)
    pursuit = 2.0 * (p_ata * p_rng) - 1.0
    sa, ca = _sincos(ata); saa, caa = _sincos(aa); sz, cz = _sincos(az); se, ce = _sincos(el)

    scal = torch.stack([
        _norm(own_spd, 0, MAX_SPEED), _norm(tgt_spd, 0, MAX_SPEED),
        torch.tanh(aoa / AOA_SC), torch.tanh(ss / SS_SC),
        torch.tanh((own_alt - MIN_ALT) / ALT_DANGER), _norm(vs, -VS_SCALE, VS_SCALE),
        _norm(rec["hp_own"], 0, 1), _norm(rec["hp_tgt"], 0, 1), rec["hp_own"] - rec["hp_tgt"],
        eadv / (eadv.abs() + EADV_SC), _norm(dist, 0, MAX_RANGE_M), _norm(closure, -MAX_CLOSURE, MAX_CLOSURE),
        sa, ca, saa, caa, sz, cz, se, ce,
        aim_sharp, aim_marg, en_sharp, en_marg, rm_near, rm_far, _norm(t, 0, EPISODE_MAX_T),
        orn[0], orn[1], opt[0], opt[1], oyw[0], oyw[1],
        trn[0], trn[1], tpt[0], tpt[1], tyw[0], tyw[1],
        ovb[0], ovb[1], tvb[0], tvb[1],
        _norm(rec["fuel_own"], 0, 1), _norm(rec["fuel_tgt"], 0, 1),
        torch.clamp(2.0 * dd - 1.0, -1.0, 1.0), torch.clamp(2.0 * dt_ - 1.0, -1.0, 1.0),
        pursuit, _norm(own_alt, 0, MAX_ALT), _norm(tgt_alt, 0, MAX_ALT),
    ], dim=-1)   # (B,50)

    # ── vector block (114) ──
    own_wn = _mv(Rbn_own, rec["own_pqr"]); tgt_wn = _mv(Rbn_tgt, rec["tgt_pqr"])
    eye = torch.eye(3, dtype=dtype, device=own.device).unsqueeze(0).expand(B, 3, 3)
    frames = {"world": eye, "mybody": Rob_own, "oppbody": Rob_tgt,
              "myvel": _dir_frame(own_vn), "oppvel": _dir_frame(tgt_vn), "los": _dir_frame(delta)}
    grav = torch.zeros_like(own_pos); grav[:, 2] = 1.0
    vecs = {"gravity": grav, "los": los_unit, "own_vel": own_vn, "tgt_vel": tgt_vn,
            "rel_vel": rel_vn, "own_omega": own_wn, "tgt_omega": tgt_wn}
    SPECS = [("gravity", "unit", {"world"}), ("los", "unit", {"los"}),
             ("own_vel", "vel", {"myvel"}), ("tgt_vel", "vel", {"oppvel"}),
             ("rel_vel", "vel", set()), ("own_omega", "omega", set()), ("tgt_omega", "omega", set())]
    FR = ["world", "mybody", "oppbody", "myvel", "oppvel", "los"]
    vfeat = []
    for key, kind, skip in SPECS:
        for fr in FR:
            if fr in skip:
                continue
            comp = _mv(frames[fr], vecs[key])
            if kind == "vel":
                comp = _norm(comp, REL_VEL_MIN, REL_VEL_MAX)
            elif kind == "omega":
                comp = torch.tanh(comp / PQR_SC)
            vfeat.append(comp)
    vblk = torch.cat(vfeat, dim=-1)   # (B,114)

    obs = torch.cat([scal, vblk, rec["act_hist"]], dim=-1)   # (B,184)
    return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)


def _bank(Rbn, dir_n):
    """_bank_about_dir 포팅: μ=atan2(yv_z,yv_y), yv=_dir_frame(dir)@body_y_ned. (sin,cos)."""
    R = _dir_frame(dir_n)
    body_y = Rbn[:, :, 1]
    yv = _mv(R, body_y)
    mu = torch.arctan2(yv[:, 2], yv[:, 1])
    return mu.sin(), mu.cos()


# 벡터블록 38개 (key, kind) 레이아웃(build_obs 와 동일 순서). frame 은 아래 dict 로.
_SPECS_F = [("gravity", "unit", {"world"}), ("los", "unit", {"los"}),
            ("own_vel", "vel", {"myvel"}), ("tgt_vel", "vel", {"oppvel"}),
            ("rel_vel", "vel", set()), ("own_omega", "omega", set()), ("tgt_omega", "omega", set())]
_FR_F = ["world", "mybody", "oppbody", "myvel", "oppvel", "los"]
_LAYOUT_F = [(key, kind, fr) for key, kind, _sk in _SPECS_F for fr in _FR_F if fr not in _sk]
# kind 경계: unit 0-9, vel 10-25, omega 26-37
_N_UNIT = sum(1 for _, k, _ in _LAYOUT_F if k == "unit")      # 10
_N_VEL = sum(1 for _, k, _ in _LAYOUT_F if k == "vel")        # 16

# 스칼라 배치 normalize 의 [lo,hi] (순서: own_spd,tgt_spd,vs,hp_own,hp_tgt,dist,closure,t_sec,
# fuel_own,fuel_tgt,own_alt,tgt_alt). device/dtype 별 캐시(hot-path torch.tensor 재생성 금지).
_NORM_LO_RAW = [0., 0., -VS_SCALE, 0., 0., 0., -MAX_CLOSURE, 0., 0., 0., 0., 0.]
_NORM_HI_RAW = [MAX_SPEED, MAX_SPEED, VS_SCALE, 1., 1., MAX_RANGE_M, MAX_CLOSURE,
                EPISODE_MAX_T, 1., 1., MAX_ALT, MAX_ALT]
_norm_cache = {}


def _norm_bounds(dev, dtype):
    key = (dev, dtype)
    if key not in _norm_cache:
        lo = torch.tensor(_NORM_LO_RAW, device=dev, dtype=dtype)
        hi = torch.tensor(_NORM_HI_RAW, device=dev, dtype=dtype)
        _norm_cache[key] = (lo, hi, (hi + lo) / 2, (hi - lo) / 2)
    return _norm_cache[key]


def build_obs_fast(own, tgt, rec):
    """build_obs 와 동일 결과의 융합판(matmul/dir_frame/ned_to_body/sincos/scalar 배치화)."""
    B = own.shape[0]; dtype = own.dtype; dev = own.device
    _NORM_LO, _NORM_HI, _NORM_MID, _NORM_HALF = _norm_bounds(dev, dtype)
    own_pos = own[:, 0:3]; tgt_pos = tgt[:, 0:3]
    delta = tgt_pos - own_pos
    dist = torch.linalg.norm(delta, dim=-1)
    los_unit = delta / torch.clamp(dist.unsqueeze(-1), min=1e-6)

    # ③ ned_to_body 배치(own+tgt)
    Rob = _ned_to_body(torch.cat([own[:, 3:6], tgt[:, 3:6]], 0))
    Rob_own, Rob_tgt = Rob[:B], Rob[B:]
    Rbn_own = Rob_own.transpose(1, 2); Rbn_tgt = Rob_tgt.transpose(1, 2)
    own_vb = own[:, 6:9]; tgt_vb = tgt[:, 6:9]
    own_vn = _mv(Rbn_own, own_vb); tgt_vn = _mv(Rbn_tgt, tgt_vb); rel_vn = tgt_vn - own_vn
    own_spd = torch.linalg.norm(own_vb, dim=-1); tgt_spd = torch.linalg.norm(tgt_vb, dim=-1)
    own_alt = -own_pos[:, 2]; tgt_alt = -tgt_pos[:, 2]

    # ② dir_frame 배치(myvel,oppvel,los)
    DF = _dir_frame(torch.cat([own_vn, tgt_vn, delta], 0))
    dir_myvel, dir_oppvel, dir_los = DF[:B], DF[B:2 * B], DF[2 * B:]

    # geometry: _mv 2회로 dedup (mv_own=Rob_own@los, mv_tgt=Rob_tgt@-los)
    mv_own = _mv(Rob_own, los_unit); mv_tgt = _mv(Rob_tgt, -los_unit)
    ata = torch.arccos(torch.clamp(mv_own[:, 0], -1.0, 1.0)) * _R2D
    enemy_ata = torch.arccos(torch.clamp(mv_tgt[:, 0], -1.0, 1.0)) * _R2D
    p1 = -mv_tgt[:, 1]; p2 = mv_tgt[:, 2]
    aa_mag = torch.arccos(torch.clamp(-mv_tgt[:, 0], -1.0, 1.0)) * _R2D
    sign = torch.where(p1 < -0.10, -torch.ones_like(aa_mag),
                       torch.where((p1 > -0.01) & (p1 < 0.01), torch.sign(p2), torch.ones_like(aa_mag)))
    aa = sign * aa_mag
    az = torch.arctan2(mv_own[:, 1], mv_own[:, 0]) * _R2D
    el = -torch.arcsin(torch.clamp(mv_own[:, 2], -1.0, 1.0)) * _R2D

    # ④ 각도 10개 sin/cos 배치: own rpy, tgt rpy, ata, aa, az, el
    ang = torch.stack([own[:, 3], own[:, 4], own[:, 5], tgt[:, 3], tgt[:, 4], tgt[:, 5],
                       ata, aa, az, el], dim=-1) * _D2R
    S = ang.sin(); C = ang.cos()

    u, v, w = own_vb[:, 0], own_vb[:, 1], own_vb[:, 2]
    aoa = torch.where(own_spd < 1.0, torch.zeros_like(u), torch.arctan2(w, u) * _R2D)
    ss = torch.where(own_spd < 1.0, torch.zeros_like(u), torch.arctan2(v, torch.sqrt(u * u + w * w)) * _R2D)
    vs = -own_vn[:, 2]
    own_eh = own_alt + own_spd ** 2 / (2 * _G); tgt_eh = tgt_alt + tgt_spd ** 2 / (2 * _G)
    eadv = own_eh - tgt_eh
    closure = ((own_vn - tgt_vn) * los_unit).sum(-1)
    t = rec["t_sec"]
    active_cone = torch.where(t >= T3_START, torch.full_like(t, T3_CONE),
                              torch.where(t >= T2_START, torch.full_like(t, T2_CONE), torch.full_like(t, T1_CONE)))
    active_maxft = torch.where(t >= T3_START, torch.full_like(t, T3_MAXFT),
                               torch.where(t >= T2_START, torch.full_like(t, T2_MAXFT), torch.full_like(t, T1_MAXFT)))
    cone_c = torch.clamp(active_cone, min=1e-6)
    min_dmg_m = MIN_DMG_FT * _FT2M; active_max_m = active_maxft * _FT2M
    span = torch.clamp(active_max_m - min_dmg_m, min=1e-6)

    # ④b 스칼라 배치화: exp 2개 / tanh 7개 / normalize 12개를 각각 한 번에.
    ex = torch.exp(torch.stack([-((ata / 3.0) ** 2), -((enemy_ata / 3.0) ** 2)], -1))  # (B,2)
    aim_sharp = 2.0 * ex[:, 0] - 1.0; en_sharp = 2.0 * ex[:, 1] - 1.0
    th = torch.tanh(torch.stack([
        aoa / AOA_SC, ss / SS_SC, (own_alt - MIN_ALT) / ALT_DANGER,
        (active_cone - ata.abs()) / cone_c, (active_cone - enemy_ata.abs()) / cone_c,
        (dist - min_dmg_m) / span, (active_max_m - dist) / span], -1))     # (B,7)
    nv = torch.stack([own_spd, tgt_spd, vs, rec["hp_own"], rec["hp_tgt"], dist, closure, t,
                      rec["fuel_own"], rec["fuel_tgt"], own_alt, tgt_alt], -1)          # (B,12)
    nvn = (torch.minimum(torch.maximum(nv, _NORM_LO), _NORM_HI) - _NORM_MID) / _NORM_HALF
    # bank: 이미 만든 dir_myvel/dir_oppvel 재사용
    yv_o = _mv(dir_myvel, Rbn_own[:, :, 1]); yv_t = _mv(dir_oppvel, Rbn_tgt[:, :, 1])
    mu = torch.stack([torch.arctan2(yv_o[:, 2], yv_o[:, 1]), torch.arctan2(yv_t[:, 2], yv_t[:, 1])], -1)
    muS = mu.sin(); muC = mu.cos()
    dd = rec["dmg_dealt"]; dtk = rec["dmg_taken"]
    p_ata = torch.clamp(1.0 - ata.abs() / PURSUIT_ATA, min=0.0)
    p_rng = torch.clamp(1.0 - dist / PURSUIT_RNG, min=0.0)
    pursuit = 2.0 * (p_ata * p_rng) - 1.0

    scal = torch.stack([
        nvn[:, 0], nvn[:, 1],                                # own_spd, tgt_spd
        th[:, 0], th[:, 1], th[:, 2], nvn[:, 2],             # aoa, ss, alt_margin, vs
        nvn[:, 3], nvn[:, 4], rec["hp_own"] - rec["hp_tgt"], # hp_own, hp_tgt, hp_diff
        eadv / (eadv.abs() + EADV_SC), nvn[:, 5], nvn[:, 6], # eadv, dist, closure
        S[:, 6], C[:, 6], S[:, 7], C[:, 7], S[:, 8], C[:, 8], S[:, 9], C[:, 9],   # ata,aa,az,el
        aim_sharp, th[:, 3], en_sharp, th[:, 4], th[:, 5], th[:, 6], nvn[:, 7],   # +t_sec
        S[:, 0], C[:, 0], S[:, 1], C[:, 1], S[:, 2], C[:, 2],                     # own rpy
        S[:, 3], C[:, 3], S[:, 4], C[:, 4], S[:, 5], C[:, 5],                     # tgt rpy
        muS[:, 0], muC[:, 0], muS[:, 1], muC[:, 1],
        nvn[:, 8], nvn[:, 9],                                # fuel own/tgt
        torch.clamp(2.0 * dd - 1.0, -1.0, 1.0), torch.clamp(2.0 * dtk - 1.0, -1.0, 1.0),
        pursuit, nvn[:, 10], nvn[:, 11],                     # pursuit, own_alt, tgt_alt
    ], dim=-1)   # (B,50)

    # ① 벡터블록: 38개 (frame@vec) 를 한 번의 배치 bmm 으로
    own_wn = _mv(Rbn_own, rec["own_pqr"]); tgt_wn = _mv(Rbn_tgt, rec["tgt_pqr"])
    eye = torch.eye(3, dtype=dtype, device=dev).unsqueeze(0).expand(B, 3, 3)
    frames = {"world": eye, "mybody": Rob_own, "oppbody": Rob_tgt,
              "myvel": dir_myvel, "oppvel": dir_oppvel, "los": dir_los}
    grav = torch.zeros_like(own_pos); grav[:, 2] = 1.0
    vecs = {"gravity": grav, "los": los_unit, "own_vel": own_vn, "tgt_vel": tgt_vn,
            "rel_vel": rel_vn, "own_omega": own_wn, "tgt_omega": tgt_wn}
    Fs = torch.stack([frames[fr] for _, _, fr in _LAYOUT_F], 0).reshape(38 * B, 3, 3)
    Vs = torch.stack([vecs[key] for key, _, _ in _LAYOUT_F], 0).reshape(38 * B, 3)
    comp = torch.bmm(Fs, Vs.unsqueeze(-1)).squeeze(-1).reshape(38, B, 3)
    comp = torch.cat([comp[:_N_UNIT],
                      torch.clamp(comp[_N_UNIT:_N_UNIT + _N_VEL], REL_VEL_MIN, REL_VEL_MAX) / REL_VEL_MAX,
                      torch.tanh(comp[_N_UNIT + _N_VEL:] / PQR_SC)], 0)
    vblk = comp.permute(1, 0, 2).reshape(B, 114)

    obs = torch.cat([scal, vblk, rec["act_hist"]], dim=-1)
    return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)


__all__ = ["build_obs", "build_obs_fast"]
