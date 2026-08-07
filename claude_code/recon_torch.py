# -*- coding: utf-8 -*-
"""StateReconstructor 진화 로직의 배치 torch 포팅(neural-MPC rollout 용).

damage_rate(3-tier), HP/fuel 적분, SO(3)-log 기반 p/q/r 추정, action 이력을 매 0.1초
스텝마다 갱신한다. planner 가 예측 state 마다 obs 를 만들 수 있도록 rec dict 를 진화시킨다.
graph-capturable(branch-free, torch.where).
"""
from __future__ import annotations
import math
import torch
from claude_code.obs_torch import _ned_to_body, _mv

_M2FT = 1.0 / 0.3048
_R2D = 180.0 / math.pi
DT = 0.1
FUEL_BURN_PER_SEC = 8.0e-5
FUEL_REF_SPEED = 300.0
MIN_FT = 500.0
T2_START, T3_START = 100.0, 150.0
T1_CONE, T2_CONE, T3_CONE = 1.0, 2.0, 3.0
T1_MAX, T2_MAX, T3_MAX = 3000.0, 3500.0, 4000.0


def damage_rate_t(r_ft, theta_deg, t):
    a = theta_deg.abs()
    z = torch.zeros_like(r_ft)
    tier1 = (r_ft >= MIN_FT) & (r_ft <= T1_MAX) & (a < T1_CONE)
    tier2 = (t >= T2_START) & (r_ft >= MIN_FT) & (r_ft <= T2_MAX) & (a < T2_CONE)
    tier3 = (t >= T3_START) & (r_ft >= MIN_FT) & (r_ft <= T3_MAX) & (a < T3_CONE)
    v1 = 1.0 * (T1_MAX - r_ft) / (T1_MAX - MIN_FT)
    v2 = 0.3 * (T2_MAX - r_ft) / (T2_MAX - MIN_FT)
    v3 = 0.1 * (T3_MAX - r_ft) / (T3_MAX - MIN_FT)
    return torch.where(tier1, v1, torch.where(tier2, v2, torch.where(tier3, v3, z)))


def _log_so3(R):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_t = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.arccos(cos_t)
    denom = 2.0 * torch.sin(theta)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    small = (theta < 1e-6) | (denom.abs() < 1e-8)
    scale = torch.where(small, torch.zeros_like(theta), theta / torch.where(small, torch.ones_like(denom), denom))
    return axis * scale.unsqueeze(-1)


def estimate_pqr_t(prev_euler, curr_euler, dt):
    Rp = _ned_to_body(prev_euler); Rc = _ned_to_body(curr_euler)
    r_delta = torch.bmm(Rp, Rc.transpose(1, 2))   # ned_to_body(prev) @ body_to_ned(curr)
    return _log_so3(r_delta) / max(dt, 1e-8)


def new_recon(B, device, dtype):
    z = torch.zeros(B, device=device, dtype=dtype)
    return dict(hp_own=torch.ones(B, device=device, dtype=dtype),
                hp_tgt=torch.ones(B, device=device, dtype=dtype),
                fuel_own=torch.ones(B, device=device, dtype=dtype),
                fuel_tgt=torch.ones(B, device=device, dtype=dtype),
                t_sec=z.clone(), dmg_dealt=z.clone(), dmg_taken=z.clone(),
                own_pqr=torch.zeros(B, 3, device=device, dtype=dtype),
                tgt_pqr=torch.zeros(B, 3, device=device, dtype=dtype),
                prev_own_att=None, prev_tgt_att=None,
                act_hist=torch.zeros(B, 20, device=device, dtype=dtype))


def advance_recon(rec, own_next, tgt_next, dt=DT):
    """예측된 다음 state(own_next,tgt_next) 로 rec 을 진화(반환=새 rec).
    호출 순서: env 와 동일하게 step 후 advance. build_obs 는 이 rec 을 쓴다."""
    delta = tgt_next[:, 0:3] - own_next[:, 0:3]
    dist_m = torch.linalg.norm(delta, dim=-1)
    r_ft = dist_m * _M2FT
    los = delta / torch.clamp(dist_m.unsqueeze(-1), min=1e-6)
    Rob_own = _ned_to_body(own_next[:, 3:6]); Rob_tgt = _ned_to_body(tgt_next[:, 3:6])
    ata_own = torch.arccos(torch.clamp(_mv(Rob_own, los)[:, 0], -1.0, 1.0)) * _R2D
    ata_tgt = torch.arccos(torch.clamp(_mv(Rob_tgt, -los)[:, 0], -1.0, 1.0)) * _R2D
    t = rec["t_sec"]
    dd = damage_rate_t(r_ft, ata_own, t)
    dtk = damage_rate_t(r_ft, ata_tgt, t)
    out = dict(rec)
    out["hp_tgt"] = torch.clamp(rec["hp_tgt"] - dd * dt, min=0.0)
    out["hp_own"] = torch.clamp(rec["hp_own"] - dtk * dt, min=0.0)
    own_spd = torch.linalg.norm(own_next[:, 6:9], dim=-1)
    tgt_spd = torch.linalg.norm(tgt_next[:, 6:9], dim=-1)
    out["fuel_own"] = torch.clamp(rec["fuel_own"] - FUEL_BURN_PER_SEC * (own_spd / FUEL_REF_SPEED) * dt, min=0.0)
    out["fuel_tgt"] = torch.clamp(rec["fuel_tgt"] - FUEL_BURN_PER_SEC * (tgt_spd / FUEL_REF_SPEED) * dt, min=0.0)
    out["dmg_dealt"] = dd; out["dmg_taken"] = dtk
    ce_own = own_next[:, 3:6]; ce_tgt = tgt_next[:, 3:6]
    if rec["prev_own_att"] is not None:
        out["own_pqr"] = estimate_pqr_t(rec["prev_own_att"], ce_own, dt)
        out["tgt_pqr"] = estimate_pqr_t(rec["prev_tgt_att"], ce_tgt, dt)
    else:
        out["own_pqr"] = torch.zeros_like(rec["own_pqr"]); out["tgt_pqr"] = torch.zeros_like(rec["tgt_pqr"])
    out["prev_own_att"] = ce_own; out["prev_tgt_att"] = ce_tgt
    out["t_sec"] = t + dt
    return out


def push_action(rec, action):
    """action 이력 갱신(row0=최신). act_hist (B,20)=[t-1(4),t-2,...,t-5]."""
    out = dict(rec)
    out["act_hist"] = torch.cat([action, rec["act_hist"][:, :16]], dim=-1)
    return out


__all__ = ["damage_rate_t", "estimate_pqr_t", "advance_recon", "push_action", "new_recon"]
