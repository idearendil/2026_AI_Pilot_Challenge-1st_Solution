# -*- coding: utf-8 -*-
"""claude164r 관측 + my_reward 보상의 GPU 벡터화 (torch, fp64).

CPU 참조(claude_code/my_observation.py, my_reward.py, GeoMathUtil.GeometryInfo)와
**동일 규약**을 재현한다. 레이아웃/상수는 참조 모듈에서 직접 import 해 정합을 강제한다
(참조가 바뀌면 검증 테스트가 어긋난다).

핵심 설계
---------
- 모든 상태를 **기체 단위(nac=2*nenv)** 로 관리한다. 기체 a 의 상대는 partner=a^1.
  거리·시간은 env 안에서 두 기체가 공유한다.
- StateReconstructor 를 배치 텐서로 유지한다(hp/fuel/t_sec/자세 history/pqr/action history
  /shaping prev_x). advance() 는 RL step 당 한 번, sim.step **후** 새 상태로 적분한다.
- 관측은 관점 기체 a(own=a, target=a^1)에서 (nac,OBS_SIZE)로 한 번에 만든다.
- 보상도 관점 기체 a 에서 (nac,)로 만든다(self-play: 두 기체 모두 보상).

state9 입력: (nac,9) = N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s].
"""
import ctypes as _CT
import math
import sys
from pathlib import Path

import torch

_RELEASE = Path(__file__).resolve().parents[1]
if str(_RELEASE) not in sys.path:
    sys.path.insert(0, str(_RELEASE))
if str(_RELEASE / "claude_code") not in sys.path:
    sys.path.insert(0, str(_RELEASE / "claude_code"))
if str(_RELEASE / "cuda_fdm" / "tests") not in sys.path:
    sys.path.insert(0, str(_RELEASE / "cuda_fdm" / "tests"))
_GEN = _RELEASE / "cuda_fdm" / "gen"

# CPU 참조 상수/레이아웃 (정합 강제)
import my_observation as R          # claude_code/my_observation.py
import my_reward as RW              # claude_code/my_reward.py

OBS_SIZE = R.OBSERVATION_SIZE       # 184 (=50 scalar + 114 vector + 20 action-history)
VEC_LAYOUT = R._VEC_LAYOUT          # [(key, kind, frame), ...] 38개
FRAME_NAMES = R.FRAME_NAMES
ACT_LEN = R.ACTION_HISTORY_LEN
ACT_DIM = R.ACTION_DIM

D2R = 3.141592653589793 / 180.0
R2D = 180.0 / 3.141592653589793
_FT2M = R.FEET_TO_METER             # 0.3048
_M2FT = R.METER_TO_FEET             # 3.28084 (obs 규약)
_G = R.G


# ══════════════════════════════════════════════════════════════════════════════
# 배치 기하 (GeometryInfo 벡터화). 모두 (M,9) own/tgt → (M,) 또는 (M,3).
# ══════════════════════════════════════════════════════════════════════════════
def ned_to_body(rpy_deg):
    """(M,3) roll/pitch/yaw[deg] → R_ned_to_body (M,3,3) = Tx@Ty@Tz.
    GeometryInfo/_ned_to_body_matrix 와 동일한 부호 규약."""
    r = rpy_deg[:, 0] * D2R
    p = rpy_deg[:, 1] * D2R
    y = rpy_deg[:, 2] * D2R
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    z = torch.zeros_like(r); o = torch.ones_like(r)
    Tx = torch.stack([torch.stack([o, z, z], 1),
                      torch.stack([z, cr, sr], 1),
                      torch.stack([z, -sr, cr], 1)], 1)
    Ty = torch.stack([torch.stack([cp, z, -sp], 1),
                      torch.stack([z, o, z], 1),
                      torch.stack([sp, z, cp], 1)], 1)
    Tz = torch.stack([torch.stack([cy, sy, z], 1),
                      torch.stack([-sy, cy, z], 1),
                      torch.stack([z, z, o], 1)], 1)
    return torch.bmm(Tx, torch.bmm(Ty, Tz))


def _mv(R3, v):
    """batched (M,3,3)·(M,3) → (M,3)."""
    return torch.einsum('mij,mj->mi', R3, v)


def _unit(v, eps=0.0):
    """(M,3) 정규화. norm==0 이면 원본(참조 규약: 0 이면 그대로)."""
    n = torch.linalg.norm(v, dim=1, keepdim=True)
    return torch.where(n > 0, v / n.clamp_min(1e-300), v)


def distance_m(own9, tgt9):
    return torch.linalg.norm(tgt9[:, :3] - own9[:, :3], dim=1)


def ata_deg(own9, tgt9):
    """3D antenna train angle(own→tgt), 0~180 (부호 없음). _get_antenna_train_angle proj=False."""
    pu = _unit(tgt9[:, :3] - own9[:, :3])
    Rnb = ned_to_body(own9[:, 3:6])
    pt = _mv(Rnb, pu)
    return torch.arccos(pt[:, 0].clamp(-1.0, 1.0)) * R2D


def aspect_deg(own9, tgt9):
    """3D aspect angle, 부호 있음. _get_aspect_angle proj=False."""
    Rt = ned_to_body(tgt9[:, 3:6])
    pu = _unit(own9[:, :3] - tgt9[:, :3])
    b = _mv(Rt, pu)                     # Tx@Ty@Tz @ p_unit
    pt0 = -b[:, 0]; pt1 = -b[:, 1]; pt2 = b[:, 2]   # Tz_pi = diag(-1,-1,1)
    sign = torch.where(pt1 > 0, torch.ones_like(pt1),
                       torch.where(pt1 < 0, -torch.ones_like(pt1),
                                   torch.where(pt2 >= 0, torch.ones_like(pt1),
                                               -torch.ones_like(pt1))))
    return sign * torch.arccos(pt0.clamp(-1.0, 1.0)) * R2D


def los_az_el_deg(own9, tgt9):
    """LOS azimuth(-180~180), elevation(-90~90) in own body. _get_los_angle."""
    du = _unit(tgt9[:, :3] - own9[:, :3])
    Rnb = ned_to_body(own9[:, 3:6])
    db = _mv(Rnb, du)
    az = torch.atan2(db[:, 1], db[:, 0]) * R2D
    el = -torch.arcsin(db[:, 2].clamp(-1.0, 1.0)) * R2D
    return az, el


# ══════════════════════════════════════════════════════════════════════════════
# 방향-정렬 좌표계 / 뱅크각 / SO(3) log (my_observation 벡터화)
# ══════════════════════════════════════════════════════════════════════════════
def dir_frame(x_ned):
    """(M,3) x축 방향 → R_ned_to_frame (M,3,3), rows=[x,y,z]. roll 은 중력 고정.
    x∥중력이면 north→east fallback. |x|<1e-8 이면 항등. _dir_frame 규약."""
    M = x_ned.shape[0]
    dev, dt = x_ned.device, x_ned.dtype
    eye = torch.eye(3, device=dev, dtype=dt).expand(M, 3, 3)
    nx = torch.linalg.norm(x_ned, dim=1, keepdim=True)
    safe = (nx.squeeze(1) >= 1e-8)
    xn = x_ned / nx.clamp_min(1e-300)

    def proj_out(ref):   # ref (3,) → z = ref - (ref·xn)xn  (M,3)
        r = ref.view(1, 3)
        d = (xn * r).sum(1, keepdim=True)
        return r - d * xn

    down = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt)
    north = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt)
    east = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt)
    z_d = proj_out(down);  n_d = torch.linalg.norm(z_d, dim=1)
    z_n = proj_out(north); n_n = torch.linalg.norm(z_n, dim=1)
    z_e = proj_out(east)
    use_d = (n_d >= 1e-6)
    use_n = (~use_d) & (n_n >= 1e-6)
    z = torch.where(use_d.unsqueeze(1), z_d,
                    torch.where(use_n.unsqueeze(1), z_n, z_e))
    z = z / torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(1e-300)
    y = torch.linalg.cross(z, xn, dim=1)
    y = y / torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-300)
    z = torch.linalg.cross(xn, y, dim=1)
    Rf = torch.stack([xn, y, z], 1)                 # rows
    return torch.where(safe.view(M, 1, 1), Rf, eye)


def bank_sincos(Rb2n, dir_ned):
    """방향벡터 축 뱅크각 μ 의 (sin,cos). _bank_about_dir 벡터화.
    Rb2n=(M,3,3) body→ned. body y축(ned)=Rb2n[:,:,1]."""
    Rf = dir_frame(dir_ned)
    body_y_ned = Rb2n[:, :, 1]
    yv = _mv(Rf, body_y_ned)
    mu = torch.atan2(yv[:, 2], yv[:, 1])
    return torch.sin(mu), torch.cos(mu)


def log_so3(Rm):
    """(M,3,3) 회전행렬 → axis*angle (M,3). _log_so3 벡터화(θ<1e-8 → 0)."""
    tr = Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2]
    cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_t)
    ax = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2],
                      Rm[:, 0, 2] - Rm[:, 2, 0],
                      Rm[:, 1, 0] - Rm[:, 0, 1]], 1)
    denom = 2.0 * torch.sin(theta)
    ok = (theta >= 1e-8) & (denom.abs() >= 1e-8)
    # denom 이 음수일 수 있으니 안전 나눗셈(0 근처는 1 로 대체 후 where 로 무효화).
    safe_denom = torch.where(denom.abs() < 1e-300, torch.ones_like(denom), denom)
    scale = torch.where(ok, theta / safe_denom, torch.zeros_like(theta))
    return ax * scale.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# 스칼라 helper
# ══════════════════════════════════════════════════════════════════════════════
def normalize_t(x, lo, hi):
    """observation.normalize 벡터화: clip 후 [-1,1] 선형."""
    if hi <= lo:
        return torch.zeros_like(x)
    mid = (hi + lo) * 0.5
    half = (hi - lo) * 0.5
    return (x.clamp(lo, hi) - mid) / half


def _sincos_deg(a_deg):
    r = a_deg * D2R
    return torch.sin(r), torch.cos(r)


def damage_rate_t(r_ft, ata_deg_abs, t_sec):
    """cone-damage rate 벡터화. my_observation.damage_rate 와 동일(tier1>2>3 우선)."""
    a = ata_deg_abs.abs()
    z = torch.zeros_like(r_ft)
    m1 = (r_ft >= R.MIN_DAMAGE_RANGE_FT) & (r_ft <= R.TIER1_MAX_RANGE_FT) & (a < R.TIER1_CONE_DEG)
    v1 = 1.0 * (R.TIER1_MAX_RANGE_FT - r_ft) / (R.TIER1_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    m2 = (t_sec >= R.TIER2_START_SEC) & (r_ft >= R.MIN_DAMAGE_RANGE_FT) & \
         (r_ft <= R.TIER2_MAX_RANGE_FT) & (a < R.TIER2_CONE_DEG)
    v2 = 0.3 * (R.TIER2_MAX_RANGE_FT - r_ft) / (R.TIER2_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    m3 = (t_sec >= R.TIER3_START_SEC) & (r_ft >= R.MIN_DAMAGE_RANGE_FT) & \
         (r_ft <= R.TIER3_MAX_RANGE_FT) & (a < R.TIER3_CONE_DEG)
    v3 = 0.1 * (R.TIER3_MAX_RANGE_FT - r_ft) / (R.TIER3_MAX_RANGE_FT - R.MIN_DAMAGE_RANGE_FT)
    return torch.where(m1, v1, torch.where(m2, v2, torch.where(m3, v3, z)))


def _active_envelope(t_sec):
    """(cone_deg, max_range_ft) 배치. _active_damage_envelope."""
    cone = torch.where(t_sec >= R.TIER3_START_SEC, torch.full_like(t_sec, R.TIER3_CONE_DEG),
                       torch.where(t_sec >= R.TIER2_START_SEC, torch.full_like(t_sec, R.TIER2_CONE_DEG),
                                   torch.full_like(t_sec, R.TIER1_CONE_DEG)))
    rng = torch.where(t_sec >= R.TIER3_START_SEC, torch.full_like(t_sec, R.TIER3_MAX_RANGE_FT),
                      torch.where(t_sec >= R.TIER2_START_SEC, torch.full_like(t_sec, R.TIER2_MAX_RANGE_FT),
                                  torch.full_like(t_sec, R.TIER1_MAX_RANGE_FT)))
    return cone, rng


# ══════════════════════════════════════════════════════════════════════════════
# 배치 reconstructor + obs/reward
# ══════════════════════════════════════════════════════════════════════════════
class BatchObsReward:
    """nac=2*nenv 기체의 관측/보상 배치 계산기 + 재구성 상태 보유."""

    def __init__(self, nenv, device="cuda", dtype=torch.float64, enable_kernel=True):
        self.nenv = nenv
        self.nac = 2 * nenv
        self.device = device
        self.dtype = dtype
        self.dt = R.DT_PER_STEP
        idx = torch.arange(self.nac, device=device)
        self.partner = idx ^ 1                       # (nac,) 상대 기체 index
        self.env_of = idx // 2                        # (nac,) → env
        self._alloc()
        self.reset_all()
        # ── 융합 NVRTC 커널(advance+reward, build_obs) — step() 핫패스 가속 ──
        self.kernel_ready = False
        if enable_kernel and dtype == torch.float64:
            self._init_kernels()

    def _alloc(self):
        n, ne, dev, dt = self.nac, self.nenv, self.device, self.dtype
        self.hp = torch.ones(n, device=dev, dtype=dt)
        self.fuel = torch.ones(n, device=dev, dtype=dt)
        self.t_sec = torch.zeros(ne, device=dev, dtype=dt)
        self.prev_att = torch.zeros(n, 3, device=dev, dtype=dt)
        self.prev_valid = torch.zeros(n, device=dev, dtype=torch.bool)
        self.pqr = torch.zeros(n, 3, device=dev, dtype=dt)
        self.last_dmg_dealt = torch.zeros(n, device=dev, dtype=dt)   # rate
        self.last_dmg_taken = torch.zeros(n, device=dev, dtype=dt)   # rate
        self.hp_loss = torch.zeros(n, device=dev, dtype=dt)          # 이번 step a 의 HP 손실
        self.act_hist = torch.zeros(n, ACT_LEN, ACT_DIM, device=dev, dtype=dt)
        self.prev_x = torch.zeros(n, device=dev, dtype=dt)           # reward shaping
        self.prev_x_valid = torch.zeros(n, device=dev, dtype=torch.bool)

    # ── 융합 커널 (advance+reward, build_obs) ─────────────────────────────────
    def _init_kernels(self):
        import cuda_rt
        src = (_GEN / "obs_kernel.cu").read_text(encoding="utf-8")
        cap = torch.cuda.get_device_capability()
        arch = f"compute_{cap[0]}{cap[1]}"
        ptx = cuda_rt.compile_ptx(src, arch)
        self._k_adv = cuda_rt.Kernel(ptx, "advance_kernel")
        self._k_obs = cuda_rt.Kernel(ptx, "build_obs_kernel")
        self.block = 128
        self.obs_buf = torch.zeros(self.nac, OBS_SIZE, dtype=torch.float32, device=self.device)
        self.reward_buf = torch.zeros(self.nac, dtype=torch.float64, device=self.device)
        self.term_buf = torch.zeros(self.nenv, dtype=torch.uint8, device=self.device)
        self.trunc_buf = torch.zeros(self.nenv, dtype=torch.uint8, device=self.device)
        # 대회 NED origin ecef 상수(rl_env 와 동일 공식)
        olat = math.radians(37.91455691666666); olon = math.radians(128.18188127777776)
        slat, clat = math.sin(olat), math.cos(olat)
        slon, clon = math.sin(olon), math.cos(olon)
        A, E2 = 6378137.0, 0.0066943799901411
        Nr = A / math.sqrt(1.0 - E2 * slat * slat)
        self._ox, self._oy, self._oz = Nr * clat * clon, Nr * clat * slon, Nr * (1.0 - E2) * slat
        self._osc = (slat, clat, slon, clon)
        self.kernel_ready = True

    def _origin_args(self):
        s, c, so, co = self._osc
        return [_CT.c_double(self._ox), _CT.c_double(self._oy), _CT.c_double(self._oz),
                _CT.c_double(s), _CT.c_double(c), _CT.c_double(so), _CT.c_double(co)]

    def kernel_advance(self, states, actions, cfg=None, min_alt=300.0, max_time=200.0):
        """융합 advance+reward 커널(1 thread/env): action push, hp/연료/pqr/시간 적분,
        종료 판정, 보상 계산을 한 번에. sim.step **후** 호출. 반환 (reward(nac,), term, trunc)."""
        cfg = cfg if cfg is not None else RW.MY_REWARD_CONFIG
        st = states.contiguous(); ac = actions.contiguous()
        own_w = float(cfg.get("own_damage_weight", 0.5))
        p = lambda t: _CT.c_void_p(t.data_ptr())
        args = [p(st), p(ac), p(self.hp), p(self.fuel), p(self.t_sec), p(self.prev_att),
                p(self.prev_valid), p(self.pqr), p(self.last_dmg_dealt), p(self.last_dmg_taken),
                p(self.hp_loss), p(self.act_hist), p(self.prev_x), p(self.prev_x_valid),
                p(self.reward_buf), p(self.term_buf), p(self.trunc_buf), _CT.c_int(self.nenv)]
        args += self._origin_args()
        args += [_CT.c_double(self.dt), _CT.c_double(min_alt), _CT.c_double(max_time),
                 _CT.c_double(own_w), _CT.c_double(float(cfg["damage_scale"])),
                 _CT.c_double(float(cfg["shaping_reward_scale"])),
                 _CT.c_double(float(cfg["win_reward"])), _CT.c_double(float(cfg["loss_reward"])),
                 _CT.c_double(float(cfg["ownship_alt_reward"])),
                 _CT.c_double(float(cfg["target_alt_reward"]))]
        grid = ((self.nenv + self.block - 1) // self.block, 1, 1)
        self._k_adv.launch(grid, (self.block, 1, 1), args)
        return self.reward_buf, self.term_buf, self.trunc_buf

    def kernel_build_obs(self, states):
        """융합 build_obs 커널(1 thread/기체): states+재구성 → obs(nac,184) f32. 순수(비파괴)."""
        st = states.contiguous()
        p = lambda t: _CT.c_void_p(t.data_ptr())
        args = [p(st), p(self.hp), p(self.fuel), p(self.t_sec), p(self.pqr),
                p(self.last_dmg_dealt), p(self.last_dmg_taken), p(self.act_hist),
                p(self.obs_buf), _CT.c_int(self.nac)]
        args += self._origin_args()
        grid = ((self.nac + self.block - 1) // self.block, 1, 1)
        self._k_obs.launch(grid, (self.block, 1, 1), args)
        return self.obs_buf

    # ── 상태 리셋 (env 단위) ──────────────────────────────────────────────────
    def reset_all(self):
        self.hp.fill_(1.0); self.fuel.fill_(1.0); self.t_sec.zero_()
        self.prev_att.zero_(); self.prev_valid.zero_(); self.pqr.zero_()
        self.last_dmg_dealt.zero_(); self.last_dmg_taken.zero_(); self.hp_loss.zero_()
        self.act_hist.zero_(); self.prev_x.zero_(); self.prev_x_valid.zero_()

    def reset_envs(self, env_mask):
        """env_mask (nenv,) bool 인 env 의 재구성 상태 초기화(autoreset/stagger reseed).
        masked_fill_ 로 **in-place·동기화없음**(GPU 스칼라 카운트 불필요) — 커널 경로에서
        텐서 주소가 유지돼 매 step 재할당/동기화 오버헤드를 없앤다."""
        ac = env_mask.repeat_interleave(2)           # (nac,)
        ac1 = ac.unsqueeze(1)
        self.hp.masked_fill_(ac, 1.0)
        self.fuel.masked_fill_(ac, 1.0)
        self.t_sec.masked_fill_(env_mask, 0.0)
        self.prev_att.masked_fill_(ac1, 0.0)
        self.prev_valid.masked_fill_(ac, False)
        self.pqr.masked_fill_(ac1, 0.0)
        self.last_dmg_dealt.masked_fill_(ac, 0.0)
        self.last_dmg_taken.masked_fill_(ac, 0.0)
        self.hp_loss.masked_fill_(ac, 0.0)
        self.act_hist.masked_fill_(ac.view(-1, 1, 1), 0.0)
        self.prev_x.masked_fill_(ac, 0.0)
        self.prev_x_valid.masked_fill_(ac, False)

    # ── stagger 용 상태 snapshot/capture/restore ──────────────────────────────
    _AC_KEYS = ("hp", "fuel", "prev_att", "prev_valid", "pqr", "last_dmg_dealt",
                "last_dmg_taken", "hp_loss", "act_hist", "prev_x", "prev_x_valid")
    _ENV_KEYS = ("t_sec",)

    def clone_state(self):
        buf = {k: getattr(self, k).clone() for k in self._AC_KEYS + self._ENV_KEYS}
        return buf

    def capture_into(self, buf, env_mask):
        """env_mask 인 env 의 현재 상태를 buf 로 복사(그 위상에서 캡처)."""
        ac = env_mask.repeat_interleave(2)
        for k in self._AC_KEYS:
            buf[k][ac] = getattr(self, k)[ac]
        for k in self._ENV_KEYS:
            buf[k][env_mask] = getattr(self, k)[env_mask]

    def restore(self, buf):
        for k in self._AC_KEYS + self._ENV_KEYS:
            getattr(self, k).copy_(buf[k])

    # ── action history push (RL step 당 한 번, 다음 obs 가 최근 action 포함) ────
    def push_actions(self, actions_nac4):
        a = actions_nac4.to(self.dtype)
        self.act_hist = torch.roll(self.act_hist, 1, dims=1)
        self.act_hist[:, 0, :] = a[:, :ACT_DIM]

    # ── advance: sim.step 후 새 상태로 재구성 적분 ────────────────────────────
    def advance(self, states9):
        """states9 (nac,9). hp/fuel/t_sec/pqr/last_dmg 갱신. build_obs/reward 전에 호출."""
        own = states9
        tgt = states9[self.partner]
        r_m = distance_m(own, tgt)
        r_ft = r_m * _M2FT                                   # obs 규약(*3.28084)
        ata_a = ata_deg(own, tgt)                            # a→partner, 0..180
        t_ac = self.t_sec[self.env_of]                       # (nac,)
        rate_deals = damage_rate_t(r_ft, ata_a, t_ac)        # a 가 partner 에 가하는 rate
        rate_taken = rate_deals[self.partner]                # a 가 받는 rate
        hp_before = self.hp
        hp_after = (hp_before - rate_taken * self.dt).clamp_min(0.0)
        self.hp_loss = hp_before - hp_after                  # 이번 step a 의 HP 손실
        self.hp = hp_after
        # 연료
        speed = torch.linalg.norm(own[:, 6:9], dim=1)
        burn = R.FUEL_BURN_PER_SEC * (speed / R.FUEL_REF_SPEED)
        self.fuel = (self.fuel - burn * self.dt).clamp_min(0.0)
        # pqr (SO3 log)
        att = own[:, 3:6]
        Rb2n_prev = ned_to_body(self.prev_att).transpose(1, 2)
        Rb2n_curr = ned_to_body(att).transpose(1, 2)
        r_delta = torch.bmm(Rb2n_prev.transpose(1, 2), Rb2n_curr)
        pqr_new = log_so3(r_delta) / max(self.dt, 1e-8)
        self.pqr = torch.where(self.prev_valid.unsqueeze(1), pqr_new, torch.zeros_like(pqr_new))
        self.prev_att = att.clone()
        self.prev_valid = torch.ones_like(self.prev_valid)
        # last dmg (rate)
        self.last_dmg_dealt = rate_deals
        self.last_dmg_taken = rate_taken
        # 시간
        self.t_sec = self.t_sec + self.dt

    # ── 관측 (nac, OBS_SIZE) ──────────────────────────────────────────────────
    def build_obs(self, states9):
        own = states9
        tgt = states9[self.partner]
        t_ac = self.t_sec[self.env_of]

        own_rpy = own[:, 3:6]; tgt_rpy = tgt[:, 3:6]
        Rnb_o = ned_to_body(own_rpy); Rb2n_o = Rnb_o.transpose(1, 2)
        Rnb_t = ned_to_body(tgt_rpy); Rb2n_t = Rnb_t.transpose(1, 2)
        own_vb = own[:, 6:9]; tgt_vb = tgt[:, 6:9]
        own_vn = _mv(Rb2n_o, own_vb)
        tgt_vn = _mv(Rb2n_t, tgt_vb)
        rel_vn = tgt_vn - own_vn
        own_spd = torch.linalg.norm(own_vb, dim=1)
        tgt_spd = torch.linalg.norm(tgt_vb, dim=1)
        own_alt = -own[:, 2]; tgt_alt = -tgt[:, 2]

        delta = tgt[:, :3] - own[:, :3]
        dist = torch.linalg.norm(delta, dim=1)
        los_u = torch.where(dist.unsqueeze(1) > 1e-6, delta / dist.clamp_min(1e-300).unsqueeze(1),
                            torch.zeros_like(delta))
        closure = torch.where(dist > 1e-6, ((own_vn - tgt_vn) * los_u).sum(1),
                              torch.zeros_like(dist))

        ata = ata_deg(own, tgt)                    # 0..180
        enemy_ata = ata[self.partner]
        aa = aspect_deg(own, tgt)
        az, el = los_az_el_deg(own, tgt)

        # ── AoA / sideslip ──
        u, v, w = own_vb[:, 0], own_vb[:, 1], own_vb[:, 2]
        slow = own_spd < 1.0
        aoa = torch.where(slow, torch.zeros_like(u), torch.atan2(w, u) * R2D)
        sslip = torch.where(slow, torch.zeros_like(u),
                            torch.atan2(v, torch.sqrt(u * u + w * w)) * R2D)
        vspeed = -own_vn[:, 2]

        e_own = own_alt + own_spd ** 2 / (2.0 * _G)
        e_tgt = tgt_alt + tgt_spd ** 2 / (2.0 * _G)
        e_adv = e_own - e_tgt

        s_ata, c_ata = _sincos_deg(ata)
        s_aa, c_aa = _sincos_deg(aa)
        s_az, c_az = _sincos_deg(az)
        s_el, c_el = _sincos_deg(el)

        aim_sharp = 2.0 * torch.exp(-((ata / 3.0) ** 2)) - 1.0
        cone, maxrng_ft = _active_envelope(t_ac)
        aim_margin = torch.tanh((cone - ata.abs()) / cone.clamp_min(1e-6))
        en_aim_sharp = 2.0 * torch.exp(-((enemy_ata / 3.0) ** 2)) - 1.0
        en_aim_margin = torch.tanh((cone - enemy_ata.abs()) / cone.clamp_min(1e-6))

        min_r_m = R.MIN_DAMAGE_RANGE_FT * _FT2M
        max_r_m = maxrng_ft * _FT2M
        span = (max_r_m - min_r_m).clamp_min(1e-6)
        rm_near = torch.tanh((dist - min_r_m) / span)
        rm_far = torch.tanh((max_r_m - dist) / span)

        or_s, or_c = _sincos_deg(own_rpy[:, 0])
        op_s, op_c = _sincos_deg(own_rpy[:, 1])
        oy_s, oy_c = _sincos_deg(own_rpy[:, 2])
        tr_s, tr_c = _sincos_deg(tgt_rpy[:, 0])
        tp_s, tp_c = _sincos_deg(tgt_rpy[:, 1])
        ty_s, ty_c = _sincos_deg(tgt_rpy[:, 2])
        ovb_s, ovb_c = bank_sincos(Rb2n_o, own_vn)
        tvb_s, tvb_c = bank_sincos(Rb2n_t, tgt_vn)

        dmg_dealt = self.last_dmg_dealt
        dmg_taken = self.last_dmg_taken
        pf_ata = (1.0 - ata.abs() / R.PURSUIT_ATA_SCALE_DEG).clamp_min(0.0)
        pf_rng = (1.0 - dist / R.PURSUIT_RANGE_M).clamp_min(0.0)
        pursuit = 2.0 * (pf_ata * pf_rng) - 1.0

        hp_o = self.hp; hp_t = self.hp[self.partner]
        fuel_o = self.fuel; fuel_t = self.fuel[self.partner]

        scalars = [
            normalize_t(own_spd, 0.0, R.MAX_SPEED),
            normalize_t(tgt_spd, 0.0, R.MAX_SPEED),
            torch.tanh(aoa / R.AOA_SCALE_DEG),
            torch.tanh(sslip / R.SIDESLIP_SCALE_DEG),
            torch.tanh((own_alt - R.MIN_ALTITUDE_M) / R.ALTITUDE_DANGER_SCALE_M),
            normalize_t(vspeed, -R.VERTICAL_SPEED_SCALE, R.VERTICAL_SPEED_SCALE),
            normalize_t(hp_o, 0.0, 1.0),
            normalize_t(hp_t, 0.0, 1.0),
            hp_o - hp_t,
            e_adv / (e_adv.abs() + R.ENERGY_ADVANTAGE_SCALE_M),
            normalize_t(dist, 0.0, R.MAX_RANGE_M),
            normalize_t(closure, -R.MAX_CLOSURE_SPEED, R.MAX_CLOSURE_SPEED),
            s_ata, c_ata, s_aa, c_aa, s_az, c_az, s_el, c_el,
            aim_sharp, aim_margin, en_aim_sharp, en_aim_margin,
            rm_near, rm_far,
            normalize_t(t_ac, 0.0, R.EPISODE_MAX_TIME_SEC),
            or_s, or_c, op_s, op_c, oy_s, oy_c,
            tr_s, tr_c, tp_s, tp_c, ty_s, ty_c,
            ovb_s, ovb_c, tvb_s, tvb_c,
            normalize_t(fuel_o, 0.0, 1.0),
            normalize_t(fuel_t, 0.0, 1.0),
            (2.0 * dmg_dealt - 1.0).clamp(-1.0, 1.0),
            (2.0 * dmg_taken - 1.0).clamp(-1.0, 1.0),
            pursuit,
            normalize_t(own_alt, 0.0, R.MAX_ALTITUDE_M),
            normalize_t(tgt_alt, 0.0, R.MAX_ALTITUDE_M),
        ]
        scal = torch.stack(scalars, 1)             # (nac,50)

        # ── frame-expressed 벡터 블록 (VEC_LAYOUT 순서) ──
        own_om_n = _mv(Rb2n_o, self.pqr)
        tgt_om_n = _mv(Rb2n_t, self.pqr[self.partner])
        frames = {
            "world": torch.eye(3, device=self.device, dtype=self.dtype).expand(self.nac, 3, 3),
            "mybody": Rnb_o, "oppbody": Rnb_t,
            "myvel": dir_frame(own_vn), "oppvel": dir_frame(tgt_vn),
            "los": dir_frame(delta),
        }
        vecs = {"gravity": torch.tensor([0.0, 0.0, 1.0], device=self.device,
                                        dtype=self.dtype).expand(self.nac, 3),
                "los": los_u, "own_vel": own_vn, "tgt_vel": tgt_vn, "rel_vel": rel_vn,
                "own_omega": own_om_n, "tgt_omega": tgt_om_n}
        vcols = []
        for key, kind, fr in VEC_LAYOUT:
            comp = _mv(frames[fr], vecs[key])       # (nac,3)
            if kind == "unit":
                vcols.append(comp)
            elif kind == "vel":
                vcols.append(normalize_t(comp, R.REL_VEL_MIN, R.REL_VEL_MAX))
            else:  # omega
                vcols.append(torch.tanh(comp / R.PQR_SCALE_RAD_S))
        vecb = torch.cat(vcols, 1)                  # (nac,114)

        acth = self.act_hist.reshape(self.nac, -1)  # (nac,20)

        obs = torch.cat([scal, vecb, acth], 1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.to(torch.float32)

    # ── 보상 (nac,) ──────────────────────────────────────────────────────────
    def compute_reward(self, states9, terminated_env, cfg=None):
        """my_reward.compute_reward 벡터화(관점 기체 a). terminated_env (nenv,) bool.
        cfg: MY_REWARD_CONFIG(없으면 참조 기본값). 반환 (nac,) reward."""
        cfg = cfg if cfg is not None else RW.MY_REWARD_CONFIG
        own = states9
        tgt = states9[self.partner]
        hp_o = self.hp; hp_t = self.hp[self.partner]
        loss_o = self.hp_loss                        # own_damage
        loss_t = self.hp_loss[self.partner]          # target_damage
        own_w = float(cfg.get("own_damage_weight", 0.5))
        dmg_scale = float(cfg["damage_scale"])
        both_alive = (hp_o > 0.0) & (hp_t > 0.0)
        r_damage = torch.where(both_alive,
                               (loss_t - loss_o * own_w) * dmg_scale,
                               torch.zeros_like(hp_o))

        # shaping (per aircraft, telescoping)
        scale = float(cfg["shaping_reward_scale"])
        dist_ft = distance_m(own, tgt) / _FT2M       # reward 규약(/0.3048)
        a1 = ata_deg(own, tgt).abs()
        a2 = ata_deg(tgt, own).abs()
        own_alt_ft = (-own[:, 2]) / _FT2M
        cur_x = self._shaping_potential(dist_ft, a1, a2, own_alt_ft)
        r_shaping = torch.where(self.prev_x_valid & (scale != 0.0),
                                (cur_x - self.prev_x) * scale, torch.zeros_like(cur_x))
        if scale != 0.0:
            self.prev_x = cur_x
            self.prev_x_valid = torch.ones_like(self.prev_x_valid)

        # terminal (win/loss=0; alt reward). terminated_env 에서만.
        term_ac = terminated_env.repeat_interleave(2)
        own_below = (-own[:, 2]) < R.MIN_ALTITUDE_M
        tgt_below = (-tgt[:, 2]) < R.MIN_ALTITUDE_M
        r_term = torch.zeros_like(hp_o)
        r_term = torch.where(term_ac & own_below,
                             torch.full_like(hp_o, float(cfg["ownship_alt_reward"])), r_term)
        r_term = torch.where(term_ac & (~own_below) & tgt_below,
                             torch.full_like(hp_o, float(cfg["target_alt_reward"])), r_term)
        r_term = r_term + torch.where(term_ac & (hp_t <= 0.0),
                                      torch.full_like(hp_o, float(cfg["win_reward"])), torch.zeros_like(hp_o))
        r_term = r_term + torch.where(term_ac & (hp_o <= 0.0),
                                      torch.full_like(hp_o, float(cfg["loss_reward"])), torch.zeros_like(hp_o))
        return r_damage + r_shaping + r_term

    @staticmethod
    def _shaping_potential(dist_ft, a1, a2, own_alt_ft):
        """my_reward._shaping_potential 벡터화."""
        near = dist_ft <= 500.0
        mid = (~near) & (dist_ft <= 15000.0)
        base_near = dist_ft + 14000.0
        x_near = (base_near * (90.0 - a1) / 90.0 * 2.5
                  - base_near * (90.0 - a2) / 90.0 * 2.5 + 999500.0)
        base_mid = 15000.0 - dist_ft
        x_mid = (base_mid + base_mid * (90.0 - a1) / 90.0 * 2.5
                 - base_mid * (90.0 - a2) / 90.0 * 2.5 + 985000.0)
        x_far = 1000000.0 - dist_ft
        x = torch.where(near, x_near, torch.where(mid, x_mid, x_far))
        in_alt = (own_alt_ft >= RW._ALT_SHAPING_FLOOR_FT) & (own_alt_ft <= RW._ALT_SHAPING_TOP_FT)
        d = own_alt_ft - RW._ALT_SHAPING_TOP_FT
        x = x + torch.where(in_alt, -(d * d) / RW._ALT_SHAPING_DIVISOR, torch.zeros_like(x))
        return x
