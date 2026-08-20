# -*- coding: utf-8 -*-
"""추론 호환 neural-MPC planner (obs-WM 계약).

full-state WM(nmpc_planner.WMWrapperV2) 은 state[2:23](FCS 내부상태 포함)를 입력으로 써서
대결 서버(plane_info_to_state 가 인덱스 0~8 만 채움)에서 못 쓴다. 이 planner 는 C++
reduced_predictor 와 동일한 계약의 obs-WM(`wm_model_obs.pt`)을 굴린다:
  WM 입력 = 관측 state[D,att,u,v,w,p,q,r](2:12) + command window(현재+과거 K-1)
  WM 출력 = 다음 [N,E,D,att,vel,pqr](0:12).  FCS 은닉상태는 command window 가 대체.

obs_torch.build_obs 가 state 인덱스 0~8 + rec.pqr 만 읽으므로(9~22 미사용) actor/critic obs 는
obs-WM 출력(0:11)만으로 충분하다. eager rollout(CUDA-graph 없이) — 제출 레이턴시 여유 충분.

계약: plan(own_state, tgt_state, seed) -> best first command(np4, throttle∈[0,1]).
"""
from __future__ import annotations
import math
import numpy as np
import torch

from claude_code.model import make_actor_critic, make_action_grid
from claude_code.obs_torch import build_obs_fast as build_obs
from claude_code.obs_torch import _ned_to_body as _n2b, _mv as _mvb
from claude_code import recon_torch as RT
from claude_code.my_reward import MY_REWARD_CONFIG as _RC

_M2FT = 1.0 / 0.3048
_R2D = 180.0 / math.pi
_D2R = math.pi / 180.0


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class WMObs:
    """obs-WM: step(state(B,>=12), cmd_window(B,K*4)) -> next state(0:12 갱신). command 공간."""

    def __init__(self, ckpt_path, device, dtype=torch.float32):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.obs_idx = list(ck["obs_idx"]); self.ang_in = list(ck["ang_in"])
        self.out_idx = list(ck["out_idx"]); self.ang_out = list(ck["ang_out"])
        self.k = int(ck["k_act"]); self.adim = int(ck["adim"])
        self.dev = torch.device(device); self.dtype = dtype
        t = lambda a: torch.tensor(a, device=device, dtype=dtype)
        self.xm, self.xs, self.ym, self.ys = t(ck["xm"]), t(ck["xs"]), t(ck["ym"]), t(ck["ys"])
        self.obs_i = torch.tensor(self.obs_idx, device=device, dtype=torch.long)
        self.ang_i = torch.tensor(self.ang_in, device=device, dtype=torch.long)
        keep = [i for i in range(len(self.obs_idx)) if i not in self.ang_in]
        self.keep_i = torch.tensor(keep, device=device, dtype=torch.long)
        self.out_i = torch.tensor(self.out_idx, device=device, dtype=torch.long)
        self.ang_out_state = torch.tensor([self.out_idx[i] for i in self.ang_out],
                                          device=device, dtype=torch.long)
        width = int(ck["width"]); blocks = int(ck["blocks"])
        self.net = _ResWM(int(ck["in_dim"]), int(ck["out_dim"]), width, blocks).to(device).eval()
        self.net.load_state_dict(ck["state"])
        for p in self.net.parameters():
            p.requires_grad_(False)

    def step(self, state, cmd_window):
        base = state.index_select(1, self.obs_i)                 # (B,10)
        ang = base.index_select(1, self.ang_i) * _D2R            # (B,3)
        rest = base.index_select(1, self.keep_i)                 # (B,7)
        feats = torch.cat([rest, torch.sin(ang), torch.cos(ang), cmd_window], dim=-1)
        d = self.net((feats - self.xm) / self.xs) * self.ys + self.ym    # (B,12)
        nxt = state.clone()
        nxt[:, self.out_i] = state.index_select(1, self.out_i) + d
        att = nxt.index_select(1, self.ang_out_state)
        nxt[:, self.ang_out_state] = (att + 180.0) % 360.0 - 180.0
        return nxt


class _ResBlock(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(w, w), torch.nn.SiLU(), torch.nn.Linear(w, w))

    def forward(self, x):
        return x + self.f(x)


class _ResWM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, width, blocks):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(in_dim, width), torch.nn.SiLU())
        self.body = torch.nn.Sequential(*[_ResBlock(width) for _ in range(blocks)])
        self.out = torch.nn.Linear(width, out_dim)

    def forward(self, x):
        return self.out(self.body(self.inp(x)))


_REC_KEYS = ("hp_own", "hp_tgt", "fuel_own", "fuel_tgt", "t_sec", "dmg_dealt", "dmg_taken",
             "own_pqr", "tgt_pqr")


class ObsMPCPlanner:
    def __init__(self, wm, ac_ckpt, device="cuda", K=12, M=8, H=20, dtype=torch.float32,
                 decide_every=1, gamma=0.98, use_nstep=True):
        self.wm = wm; self.device = torch.device(device); self.dtype = dtype
        self.K, self.M, self.H, self.B = K, M, H, K * M
        self.decide_every = int(decide_every)
        self.use_nstep = bool(use_nstep); self.gamma = float(gamma)
        self.r_dmg_scale = float(_RC["damage_scale"]); self.r_shape_scale = float(_RC["shaping_reward_scale"])
        self.min_alt = 300.0
        dv = self.device
        self._r_own_alt = torch.tensor(float(_RC["ownship_alt_reward"]), device=dv, dtype=dtype)
        self._r_tgt_alt = torch.tensor(float(_RC["target_alt_reward"]), device=dv, dtype=dtype)
        self._zero = torch.tensor(0.0, device=dv, dtype=dtype)
        self.kw = self.wm.k * self.wm.adim          # command window 폭(=20)
        ck = torch.load(ac_ckpt, map_location=device, weights_only=False)
        mk = ck["model_kwargs"]; self.num_bins = int(mk["num_bins"])
        self.model = make_actor_critic(**mk).to(device).eval()
        self.model.load_state_dict({k: torch.as_tensor(v) for k, v in ck["state_dict"].items()})
        rms = ck["obs_rms"]
        self.rms_mean = torch.tensor(rms["mean"], device=device, dtype=dtype)
        self.rms_std = torch.sqrt(torch.tensor(rms["var"], device=device, dtype=dtype) + 1e-8)
        self.grid = torch.tensor(make_action_grid(self.num_bins), device=device, dtype=dtype)
        self._graph = None

    # ── primitives ──
    def _norm(self, obs):
        return torch.clamp((obs - self.rms_mean) / self.rms_std, -10.0, 10.0)

    def _logits(self, obs):
        return self.model.actor_logits(self._norm(obs)).view(-1, 4, self.num_bins)

    def _value(self, obs):
        return self.model.get_value(self._norm(obs)).view(-1)

    @staticmethod
    def _to_cmd(a):
        return torch.cat([a[:, :3], (a[:, 3:4] + 1.0) * 0.5], dim=-1)

    def _cmd_window(self, ah):
        """raw action history(B,20, throttle∈[-1,1]) → command window(throttle∈[0,1])."""
        w = ah.clone()
        w[:, 3::self.wm.adim] = (w[:, 3::self.wm.adim] + 1.0) * 0.5
        return w

    def _gidx(self, logits, noise):
        g = -torch.log(-torch.log(noise.clamp(1e-9, 1.0)))
        return (logits + g).argmax(-1)

    def _obs(self, o, t, rec, ah):
        return build_obs(o, t, {k: rec[k] for k in _REC_KEYS} | {"act_hist": ah})

    def _shaping_x(self, own, tgt):
        delta = tgt[:, 0:3] - own[:, 0:3]
        dist_m = torch.linalg.norm(delta, dim=-1)
        dist_ft = dist_m * _M2FT
        los = delta / torch.clamp(dist_m.unsqueeze(-1), min=1e-6)
        a1 = torch.arccos(torch.clamp(_mvb(_n2b(own[:, 3:6]), los)[:, 0], -1.0, 1.0)) * _R2D
        a2 = torch.arccos(torch.clamp(_mvb(_n2b(tgt[:, 3:6]), -los)[:, 0], -1.0, 1.0)) * _R2D
        alt_ft = (-own[:, 2]) * _M2FT
        c1 = (90.0 - a1) / 90.0 * 2.5; c2 = (90.0 - a2) / 90.0 * 2.5
        base_n = dist_ft + 14000.0
        x_near = base_n * c1 - base_n * c2 + 999500.0
        base_m = 15000.0 - dist_ft
        x_mid = base_m + base_m * c1 - base_m * c2 + 985000.0
        x_far = 1000000.0 - dist_ft
        x = torch.where(dist_ft <= 500.0, x_near, torch.where(dist_ft <= 15000.0, x_mid, x_far))
        # my_reward 신형 고도항: 1000~10000ft 에서 -(alt-10000)^2/101.25
        alt_term = -((alt_ft - 10000.0) ** 2) / 101.25
        x = x + torch.where((alt_ft >= 1000.0) & (alt_ft <= 10000.0), alt_term, self._zero)
        return x

    # ── 핵심 rollout ──
    def _rollout(self, own0, tgt0, seed, n_cand, n_self, n_opp):
        B, dev, dt = self.B, self.device, self.dtype
        own = own0.unsqueeze(0).repeat(B, 1); tgt = tgt0.unsqueeze(0).repeat(B, 1)
        exp = lambda x: x.reshape(1).repeat(B) if x.dim() == 0 else x.unsqueeze(0).repeat(B, 1)
        rec = {"hp_own": exp(seed["hp_own"]), "hp_tgt": exp(seed["hp_tgt"]),
               "fuel_own": exp(seed["fuel_own"]), "fuel_tgt": exp(seed["fuel_tgt"]),
               "t_sec": exp(seed["t_sec"]), "dmg_dealt": torch.zeros(B, device=dev, dtype=dt),
               "dmg_taken": torch.zeros(B, device=dev, dtype=dt),
               "own_pqr": exp(seed["prev_own_pqr"]), "tgt_pqr": exp(seed["prev_tgt_pqr"]),
               "prev_own_att": own[:, 3:6].clone(), "prev_tgt_att": tgt[:, 3:6].clone()}
        my_ah = exp(seed["my_act_hist"]); opp_ah = exp(seed["opp_act_hist"])

        rec0 = {k: rec[k][:1] for k in _REC_KEYS}
        lg0 = self._logits(build_obs(own[:1], tgt[:1], rec0 | {"act_hist": my_ah[:1]}))
        cand_idx = self._gidx(lg0.expand(self.K, 4, self.num_bins), n_cand)
        cand_idx = torch.cat([lg0.argmax(-1), cand_idx[1:]], dim=0)
        cand = self.grid[cand_idx]
        first = cand.repeat_interleave(self.M, dim=0)

        cat = torch.cat
        a_me = first; a_op = None
        G = torch.zeros(B, device=dev, dtype=dt); alive = torch.ones(B, device=dev, dtype=dt)
        x_prev = self._shaping_x(own, tgt) if self.use_nstep else None; disc = 1.0
        for h in range(self.H):
            if h % self.decide_every == 0:
                own_b = cat([own, tgt], 0); opp_b = cat([tgt, own], 0)
                rec_b = {"hp_own": cat([rec["hp_own"], rec["hp_tgt"]]), "hp_tgt": cat([rec["hp_tgt"], rec["hp_own"]]),
                         "fuel_own": cat([rec["fuel_own"], rec["fuel_tgt"]]), "fuel_tgt": cat([rec["fuel_tgt"], rec["fuel_own"]]),
                         "t_sec": cat([rec["t_sec"], rec["t_sec"]]), "dmg_dealt": cat([rec["dmg_dealt"], rec["dmg_taken"]]),
                         "dmg_taken": cat([rec["dmg_taken"], rec["dmg_dealt"]]),
                         "own_pqr": cat([rec["own_pqr"], rec["tgt_pqr"]]), "tgt_pqr": cat([rec["tgt_pqr"], rec["own_pqr"]]),
                         "act_hist": cat([my_ah, opp_ah])}
                lg = self._logits(build_obs(own_b, opp_b, rec_b))
                if h != 0:
                    a_me = self.grid[self._gidx(lg[:self.B], n_self[h])]
                a_op = self.grid[self._gidx(lg[self.B:], n_opp[h])]
            my_ah = cat([a_me, my_ah[:, :self.kw - self.wm.adim]], -1)
            opp_ah = cat([a_op, opp_ah[:, :self.kw - self.wm.adim]], -1)
            sn = self.wm.step(cat([own, tgt], 0),
                              cat([self._cmd_window(my_ah), self._cmd_window(opp_ah)], 0))
            own = sn[:self.B]; tgt = sn[self.B:]
            rec = RT.advance_recon(rec, own, tgt, RT.DT)
            if self.use_nstep:
                x_new = self._shaping_x(own, tgt)
                r_shape = (x_new - x_prev) * self.r_shape_scale
                both = ((rec["hp_own"] > 0.0) & (rec["hp_tgt"] > 0.0)).to(dt)
                r_dmg = both * (rec["dmg_dealt"] * RT.DT - 0.5 * rec["dmg_taken"] * RT.DT) * self.r_dmg_scale
                G = G + disc * (r_shape + r_dmg) * alive
                alt_own = -own[:, 2]; alt_tgt = -tgt[:, 2]
                own_dead = (alt_own < self.min_alt) | (rec["hp_own"] <= 0.0)
                tgt_dead = (alt_tgt < self.min_alt) | (rec["hp_tgt"] <= 0.0)
                r_term = torch.where(alt_own < self.min_alt, self._r_own_alt,
                                     torch.where(alt_tgt < self.min_alt, self._r_tgt_alt, self._zero))
                newly = alive * (own_dead | tgt_dead).to(dt)
                G = G + disc * r_term * newly
                alive = alive * (1.0 - (own_dead | tgt_dead).to(dt))
                x_prev = x_new; disc = disc * self.gamma
        leaf_v = self._value(self._obs(own, tgt, rec, my_ah))
        if self.use_nstep:
            val = (G + disc * leaf_v * alive).view(self.K, self.M).mean(1)
        else:
            val = leaf_v.view(self.K, self.M).mean(1)
        best = val.argmax()
        return cand.index_select(0, best.view(1)).squeeze(0)

    @torch.no_grad()
    def plan(self, own_state, tgt_state, seed):
        """eager 경로(참고/검증용; 느림 — 실사용은 plan_fast)."""
        dev, dt = self.device, self.dtype
        own = torch.as_tensor(np.asarray(own_state)[:23], device=dev, dtype=dt)
        tgt = torch.as_tensor(np.asarray(tgt_state)[:23], device=dev, dtype=dt)
        s = {k: torch.as_tensor(np.asarray(seed[k]), device=dev, dtype=dt) for k in
             ("hp_own", "hp_tgt", "fuel_own", "fuel_tgt", "t_sec", "prev_own_pqr", "prev_tgt_pqr",
              "my_act_hist", "opp_act_hist")}
        nc = torch.rand(self.K, 4, self.num_bins, device=dev)
        ns = torch.rand(self.H, self.B, 4, self.num_bins, device=dev)
        no = torch.rand(self.H, self.B, 4, self.num_bins, device=dev)
        a = self._rollout(own, tgt, s, nc, ns, no)
        return self._to_cmd(a.unsqueeze(0)).squeeze(0).cpu().numpy()

    # ── CUDA-graph 캡처 (실시간 경로) ──
    def build_graph(self):
        dev, dt, nb = self.device, self.dtype, self.num_bins
        z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
        self.g_own = z(23); self.g_tgt = z(23)
        self.g_seed = {"hp_own": z(), "hp_tgt": z(), "fuel_own": z(), "fuel_tgt": z(), "t_sec": z(),
                       "prev_own_pqr": z(3), "prev_tgt_pqr": z(3), "my_act_hist": z(20), "opp_act_hist": z(20)}
        self.g_ncand = z(self.K, 4, nb); self.g_nself = z(self.H, self.B, 4, nb); self.g_nopp = z(self.H, self.B, 4, nb)

        def run():
            return self._rollout(self.g_own, self.g_tgt, self.g_seed, self.g_ncand, self.g_nself, self.g_nopp)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.g_out = run()

    @torch.no_grad()
    def plan_fast(self, own_state, tgt_state, seed):
        if self._graph is None:
            self.build_graph()
        self.g_own.copy_(torch.as_tensor(np.asarray(own_state)[:23], device=self.device, dtype=self.dtype))
        self.g_tgt.copy_(torch.as_tensor(np.asarray(tgt_state)[:23], device=self.device, dtype=self.dtype))
        for k in self.g_seed:
            self.g_seed[k].copy_(torch.as_tensor(np.asarray(seed[k]), device=self.device, dtype=self.dtype))
        self.g_ncand.uniform_(); self.g_nself.uniform_(); self.g_nopp.uniform_()
        self._graph.replay()
        return self._to_cmd(self.g_out.detach().unsqueeze(0)).squeeze(0).cpu().numpy()


__all__ = ["WMObs", "ObsMPCPlanner"]
