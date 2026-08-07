# -*- coding: utf-8 -*-
"""neural-MPC planner: 교체 가능한 world model + actor(후보/상대) + critic(leaf).
H스텝 rollout 후 2초 뒤 critic value 가 가장 좋은 first-action 을 고른다.

CUDA-graph 캡처 지원: 샘플링을 Gumbel-max(노이즈=graph 밖 입력)로 바꿔 rollout 전체를
정적 그래프로 캡처한다 → plan_fast() 는 노이즈/상태만 채우고 replay(수 ms). WM 은
wm_step(state23,action4)->next23 인터페이스만 지키면 교체 가능.
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


class WMWrapper:
    def __init__(self, ckpt_path, device, dtype=torch.float32):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.xm = torch.tensor(ck["xm"], device=device, dtype=dtype)
        self.xs = torch.tensor(ck["xs"], device=device, dtype=dtype)
        self.ym = torch.tensor(ck["ym"], device=device, dtype=dtype)
        self.ys = torch.tensor(ck["ys"], device=device, dtype=dtype)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(25, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 23)).to(device).eval()
        self.net.load_state_dict(ck["state"])

    def step(self, state, action):
        feats = torch.cat([state[:, 2:23], action], dim=-1)
        dn = self.net((feats - self.xm) / self.xs)
        return state + dn * self.ys + self.ym


class _ResBlock(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(w, w), torch.nn.SiLU(), torch.nn.Linear(w, w))
    def forward(self, x): return x + self.f(x)


class _ResWM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, width, blocks):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(in_dim, width), torch.nn.SiLU())
        self.body = torch.nn.Sequential(*[_ResBlock(width) for _ in range(blocks)])
        self.out = torch.nn.Linear(width, out_dim)
    def forward(self, x): return self.out(self.body(self.inp(x)))


class WMWrapperV2:
    """v2 WM: 각도 sin/cos 입력 + residual/SiLU + 출력 각도 wrap. graph-capturable."""
    _DEG2RAD = math.pi / 180.0

    def __init__(self, ckpt_path, device, dtype=torch.float32):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.xm = torch.tensor(ck["xm"], device=device, dtype=dtype)
        self.xs = torch.tensor(ck["xs"], device=device, dtype=dtype)
        self.ym = torch.tensor(ck["ym"], device=device, dtype=dtype)
        self.ys = torch.tensor(ck["ys"], device=device, dtype=dtype)
        self.ang_local = list(ck["ang_local"])            # state[2:23] 내 각도 위치 [1,2,3]
        keep = [i for i in range(21) if i not in self.ang_local]
        self.keep_idx = torch.tensor(keep, device=device, dtype=torch.long)
        self.ang_idx = torch.tensor(self.ang_local, device=device, dtype=torch.long)
        self.net = _ResWM(int(ck["in_dim"]), 23, int(ck["width"]), int(ck["blocks"])).to(device).eval()
        self.net.load_state_dict(ck["state"])
        for p in self.net.parameters(): p.requires_grad_(False)

    def step(self, state, action):
        base = state[:, 2:23]
        ang = base.index_select(1, self.ang_idx) * self._DEG2RAD
        rest = base.index_select(1, self.keep_idx)
        feats = torch.cat([rest, torch.sin(ang), torch.cos(ang), action], dim=-1)
        dn = self.net((feats - self.xm) / self.xs)
        nxt = state + dn * self.ys + self.ym
        att = (nxt[:, 3:6] + 180.0) % 360.0 - 180.0        # roll/pit/yaw wrap
        return torch.cat([nxt[:, :3], att, nxt[:, 6:]], dim=-1)


_REC_KEYS = ("hp_own", "hp_tgt", "fuel_own", "fuel_tgt", "t_sec", "dmg_dealt", "dmg_taken",
             "own_pqr", "tgt_pqr")


class NeuralMPCPlanner:
    def __init__(self, wm, ac_ckpt, device="cuda", K=12, M=8, H=20, dtype=torch.float32,
                 decide_every=1, gamma=0.98, use_nstep=True):
        self.wm = wm; self.device = torch.device(device); self.dtype = dtype
        self.K, self.M, self.H, self.B = K, M, H, K * M
        self.decide_every = int(decide_every)   # 결정(obs+actor) 주기(WM 스텝 단위). 2 = 0.2초.
        # ── n-step return 평가: Σγ^t r_t + γ^H V(s_H). use_nstep=False 면 leaf V(s_H)만.
        self.use_nstep = bool(use_nstep); self.gamma = float(gamma)
        self.r_dmg_scale = float(_RC["damage_scale"]); self.r_shape_scale = float(_RC["shaping_reward_scale"])
        self.min_alt = 300.0                      # env min_altitude(m). 이하=고도이탈 종료.
        dv = torch.device(device)
        self._r_own_alt = torch.tensor(float(_RC["ownship_alt_reward"]), device=dv, dtype=dtype)
        self._r_tgt_alt = torch.tensor(float(_RC["target_alt_reward"]), device=dv, dtype=dtype)
        self._zero = torch.tensor(0.0, device=dv, dtype=dtype)
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

    def _gidx(self, logits, noise):
        """Gumbel-max: argmax(logits - log(-log(u))). noise ~ U(0,1) 같은 shape."""
        g = -torch.log(-torch.log(noise.clamp(1e-9, 1.0)))
        return (logits + g).argmax(-1)

    @staticmethod
    def _swap(rec, opp_ah):
        return {"hp_own": rec["hp_tgt"], "hp_tgt": rec["hp_own"], "fuel_own": rec["fuel_tgt"],
                "fuel_tgt": rec["fuel_own"], "t_sec": rec["t_sec"], "dmg_dealt": rec["dmg_taken"],
                "dmg_taken": rec["dmg_dealt"], "own_pqr": rec["tgt_pqr"], "tgt_pqr": rec["own_pqr"],
                "act_hist": opp_ah}

    def _obs(self, o, t, rec, ah):
        return build_obs(o, t, {k: rec[k] for k in _REC_KEYS} | {"act_hist": ah})

    def _shaping_x(self, own, tgt):
        """my_reward._shaping_potential 벡터화판: 거리(ft)/ATA(a1,a2)/고도(ft) → 포텐셜 x (B,)."""
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
        alt_term = (alt_ft - 3000.0) * (3000.0 - alt_ft) / 1000.0
        x = x + torch.where((alt_ft >= 1000.0) & (alt_ft <= 3000.0), alt_term, self._zero)
        return x

    # ── 핵심 rollout (eager/graph 공용). 모든 입력=텐서, 출력=best action(4,) ──
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
        lg0 = self._logits(build_obs(own[:1], tgt[:1], rec0 | {"act_hist": my_ah[:1]}))  # (1,4,nb)
        cand_idx = self._gidx(lg0.expand(self.K, 4, self.num_bins), n_cand)             # (K,4)
        cand_idx = torch.cat([lg0.argmax(-1), cand_idx[1:]], dim=0)                     # cand0=greedy
        cand = self.grid[cand_idx]                                                      # (K,4)
        first = cand.repeat_interleave(self.M, dim=0)                                   # (B,4)

        cat = torch.cat
        a_me = first; a_op = None
        # n-step return 누적기: G=Σγ^t r_t, alive=미종료 마스크, x_prev=직전 shaping potential.
        G = torch.zeros(B, device=dev, dtype=dt); alive = torch.ones(B, device=dev, dtype=dt)
        x_prev = self._shaping_x(own, tgt) if self.use_nstep else None; disc = 1.0
        for h in range(self.H):
            if h % self.decide_every == 0:   # 결정 스텝(매 decide_every WM스텝=0.1s*de). obs+actor.
                own_b = cat([own, tgt], 0); opp_b = cat([tgt, own], 0)
                rec_b = {"hp_own": cat([rec["hp_own"], rec["hp_tgt"]]), "hp_tgt": cat([rec["hp_tgt"], rec["hp_own"]]),
                         "fuel_own": cat([rec["fuel_own"], rec["fuel_tgt"]]), "fuel_tgt": cat([rec["fuel_tgt"], rec["fuel_own"]]),
                         "t_sec": cat([rec["t_sec"], rec["t_sec"]]), "dmg_dealt": cat([rec["dmg_dealt"], rec["dmg_taken"]]),
                         "dmg_taken": cat([rec["dmg_taken"], rec["dmg_dealt"]]),
                         "own_pqr": cat([rec["own_pqr"], rec["tgt_pqr"]]), "tgt_pqr": cat([rec["tgt_pqr"], rec["own_pqr"]]),
                         "act_hist": cat([my_ah, opp_ah])}
                lg = self._logits(build_obs(own_b, opp_b, rec_b))      # (2B,4,nb)
                if h != 0:
                    a_me = self.grid[self._gidx(lg[:self.B], n_self[h])]
                a_op = self.grid[self._gidx(lg[self.B:], n_opp[h])]
            # 결정 스텝이 아니면 직전 action 유지(hold). action 이력은 매 0.1s 갱신.
            my_ah = cat([a_me, my_ah[:, :16]], -1)
            opp_ah = cat([a_op, opp_ah[:, :16]], -1)
            sn = self.wm.step(cat([own, tgt], 0), cat([self._to_cmd(a_me), self._to_cmd(a_op)], 0))
            own = sn[:self.B]; tgt = sn[self.B:]
            rec = RT.advance_recon(rec, own, tgt, RT.DT)
            if self.use_nstep:
                # per-step reward (env my_reward 규약): shaping 차분 + damage(양측 생존 시).
                x_new = self._shaping_x(own, tgt)
                r_shape = (x_new - x_prev) * self.r_shape_scale
                both = ((rec["hp_own"] > 0.0) & (rec["hp_tgt"] > 0.0)).to(dt)
                r_dmg = both * (rec["dmg_dealt"] * RT.DT - 0.5 * rec["dmg_taken"] * RT.DT) * self.r_dmg_scale
                G = G + disc * (r_shape + r_dmg) * alive
                # 종료(고도이탈/격추) 처리: 최초 종료 스텝에만 terminal reward, 이후 bootstrap 제외.
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
            val = (G + disc * leaf_v * alive).view(self.K, self.M).mean(1)   # Σγ^t r_t + γ^H V(s_H)
        else:
            val = leaf_v.view(self.K, self.M).mean(1)
        best = val.argmax()
        return cand.index_select(0, best.view(1)).squeeze(0)   # (4,)

    # ── CUDA-graph 캡처 ──
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
            for _ in range(3): run()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.g_out = run()

    @torch.no_grad()
    def plan_fast(self, own_state, tgt_state, seed):
        if self._graph is None:
            self.build_graph()
        self.g_own.copy_(torch.as_tensor(own_state[:23], device=self.device, dtype=self.dtype))
        self.g_tgt.copy_(torch.as_tensor(tgt_state[:23], device=self.device, dtype=self.dtype))
        for k in self.g_seed:
            self.g_seed[k].copy_(torch.as_tensor(np.asarray(seed[k]), device=self.device, dtype=self.dtype))
        self.g_ncand.uniform_(); self.g_nself.uniform_(); self.g_nopp.uniform_()
        self._graph.replay()
        return self.g_out.detach().cpu().numpy()

    @torch.no_grad()
    def plan(self, own_state, tgt_state, seed):
        """eager 경로(참고/검증용)."""
        dev, dt = self.device, self.dtype
        own = torch.as_tensor(own_state[:23], device=dev, dtype=dt)
        tgt = torch.as_tensor(tgt_state[:23], device=dev, dtype=dt)
        s = {k: torch.as_tensor(np.asarray(seed[k]), device=dev, dtype=dt) for k in
             ("hp_own", "hp_tgt", "fuel_own", "fuel_tgt", "t_sec", "prev_own_pqr", "prev_tgt_pqr",
              "my_act_hist", "opp_act_hist")}
        nc = torch.rand(self.K, 4, self.num_bins, device=dev)
        ns = torch.rand(self.H, self.B, 4, self.num_bins, device=dev)
        no = torch.rand(self.H, self.B, 4, self.num_bins, device=dev)
        return self._rollout(own, tgt, s, nc, ns, no).cpu().numpy()


__all__ = ["WMWrapper", "WMWrapperV2", "NeuralMPCPlanner"]
