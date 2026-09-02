# -*- coding: utf-8 -*-
"""고도 선형 블렌딩(altitude blend) 복합 ActionProvider.

[[altguard_provider]] 의 **하드 스위치** 대신, 상한/하한 고도 사이에서 basic 정책
(actor net)의 action 과 team-share MPC 의 action 을 **고도 기반 선형 가중평균**으로
섞는다.

  - alt ≥ blend_hi_ft(기본 4000ft) : w=0 → **순수 actor**(MPC 미실행)
  - alt = 3000ft                   : w=0.5 → actor·MPC **평균**
  - alt ≤ blend_lo_ft(기본 2000ft) : w=1 → **순수 MPC**
  - 그 사이는 선형:  w = (hi - alt) / (hi - lo)
    → 4000→2000 으로 내려올수록 MPC 가중 w 가 0→1 로 커지고, 올라가면 반대로 작아진다.

출력 action = (1-w)·actor_cmd + w·mpc_cmd  (성분별). actor_cmd 와 mpc_cmd 는 모두
서버 CMD 공간 [roll,pitch,rudder ∈ [-1,1], throttle ∈ [0,1]] 이라 성분별 블렌딩이
그대로 유효하다.

제어 주기 / 관측 일관성은 altguard 와 동일:
  - actor(basic): RL-step 경계(step_ratio substep)마다 1회 결정 → 10Hz, 사이엔 캐시 유지.
  - MPC        : w>0 인 구간에서 매 substep 재호출 → 60Hz.
  - 가중 w     : 매 substep 현재 고도로 재계산 → 부드럽게 변한다.
  - StateReconstructor 는 RL-step 마다 1회 advance + **실제 적용된(블렌딩된) action** 을
    push 해, 순수 actor 로 복귀했을 때 관측(HP·action 이력)이 끊기지 않게 한다.

MPC state 규약(pqr/시간 자체 생성)도 altguard 와 동일하다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

_ROOT = Path(__file__).resolve().parents[1]
_DEF_MPC_ROOT = _ROOT / "Release_MPC_team_share"

FT_TO_M = 0.3048


def _load_teamshare_mpc(mpc_root: Path, config_path: str | None):
    """Release_MPC_team_share 의 MPCActionProvider 를 로드(경로 격리 후 복원)."""
    src = str(Path(mpc_root).resolve() / "src")
    if src not in sys.path:
        sys.path.append(src)              # append: 메인 dogfight/observation import 을 가리지 않게
    from mpc.config import load_config
    from mpc.provider import MPCActionProvider as TeamShareMPC
    cfg = load_config(config_path) if config_path else load_config(
        str(Path(mpc_root).resolve() / "configs" / "mpc.yaml"))
    return TeamShareMPC(Path(mpc_root).resolve(), cfg), cfg


class AltBlendMPCProvider(ActionProvider):
    """basic 정책(10Hz) + team-share MPC(60Hz)를 고도 선형 가중평균으로 섞는 provider."""

    def __init__(self, bundle_dir: str | Path, *, mpc_root: str | Path = _DEF_MPC_ROOT,
                 mpc_config_path: str | None = None, step_ratio: int = 6,
                 device: str = "cpu", stochastic: bool = True,
                 blend_hi_ft: float = 4000.0, blend_lo_ft: float = 2000.0,
                 confidence: float = 0.9):
        from GeoMathUtil import GeometryInfo
        from claude_code.model import (load_bundle, make_obs_normalizer,
                                       discrete_indices_to_continuous,
                                       policy_action_to_command)
        from claude_code import my_observation as MO

        if not (blend_hi_ft > blend_lo_ft):
            raise ValueError(f"blend_hi_ft({blend_hi_ft}) 는 blend_lo_ft({blend_lo_ft}) 보다 커야 합니다")

        self._MO = MO
        self._to_cont = discrete_indices_to_continuous
        self._to_cmd = policy_action_to_command
        self.step = max(1, int(step_ratio))
        self.device = device
        self.stochastic = bool(stochastic)
        self.hi_m = float(blend_hi_ft) * FT_TO_M
        self.lo_m = float(blend_lo_ft) * FT_TO_M
        self.confidence = float(confidence)

        # ── basic 번들 정책 ─────────────────────────────────────────────────
        self.model, self.meta = load_bundle(str(bundle_dir), device=device)
        if self.meta.get("observation_module") != "claude_code.my_observation":
            raise ValueError(
                "AltBlendMPCProvider 의 basic 번들은 claude164r(my_observation) 이어야 "
                f"합니다: {bundle_dir} (module={self.meta.get('observation_module')!r})")
        self._normalize = make_obs_normalizer(self.meta.get("obs_normalization"))
        self._geo = GeometryInfo()
        self._rec = MO.StateReconstructor()

        # ── team-share MPC ─────────────────────────────────────────────────
        self.mpc, self.mpc_cfg = _load_teamshare_mpc(Path(mpc_root), mpc_config_path)
        from mpc.transforms import AngularRateEstimator      # MPC 로드 후 import 가능
        self._own_rates = AngularRateEstimator()
        self._tgt_rates = AngularRateEstimator()

        self._sub = 0                 # 에피소드 시작부터의 substep 카운터(=시간 격자)
        self._actor_cmd = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # 에피소드 첫 관측은 advance 없이 fresh recon(GPU 학습 reset→build 규약).
        self._first_boundary = True

    # ---------------------------------------------------------------- helpers
    def _mpc_weight(self, alt_m: float) -> float:
        """고도(m) → MPC 가중 w∈[0,1]. hi 이상=0, lo 이하=1, 사이 선형."""
        if alt_m >= self.hi_m:
            return 0.0
        if alt_m <= self.lo_m:
            return 1.0
        return float((self.hi_m - alt_m) / (self.hi_m - self.lo_m))

    def _actor_command(self, own: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """claude164r 관측 → basic 정책 샘플링 → command. (reconstructor push 는 하지 않음)"""
        obs = self._MO.build_observation(own, tgt, self._geo, None, reconstructor=self._rec)
        obs = self._normalize(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            act_fn = self.model.act_stochastic if self.stochastic else self.model.act_deterministic
            raw = act_fn(obs_t).squeeze(0).cpu().numpy()
        if hasattr(self.model, "num_bins"):
            raw = self._to_cont(raw, self.model.num_bins)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)[:4]
        return np.asarray(self._to_cmd(raw), dtype=np.float32)

    def _mpc_state(self, s9: np.ndarray, pqr_deg: np.ndarray, t: float) -> np.ndarray:
        st = np.zeros(51, dtype=np.float64)
        st[0:9] = np.asarray(s9, dtype=np.float64)[0:9]
        st[9:12] = np.asarray(pqr_deg, dtype=np.float64)
        st[41] = float(t)
        return st

    def _mpc_command(self, own: np.ndarray, tgt: np.ndarray,
                     own_pqr_deg: np.ndarray, tgt_pqr_deg: np.ndarray,
                     t: float) -> np.ndarray:
        res = self.mpc.compute_action(ActionContext(
            sim=None, opponent_sim=None,
            ownship_state=self._mpc_state(own, own_pqr_deg, t),
            target_state=self._mpc_state(tgt, tgt_pqr_deg, t),
            info={"frame_index": self._sub}))
        return np.asarray(res.action, dtype=np.float32)

    # ----------------------------------------------------------------- API
    def reset(self, context: ActionContext | None = None) -> None:
        self._rec.reset()
        self.mpc.reset(None)
        self._own_rates.reset()
        self._tgt_rates.reset()
        self._sub = 0
        self._actor_cmd = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._first_boundary = True

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        tgt = np.asarray(context.target_state, dtype=np.float64)
        t = self._sub / float(self._MO.SIM_HZ)          # substep sim time (60Hz 격자)
        boundary = (self._sub % self.step == 0)

        # 각속도 추정기는 매 substep(60Hz) 갱신 → MPC 진입 시 pqr 이 항상 최신(대회 방식).
        own_pqr_deg = np.degrees(self._own_rates.update(own[3:6], t))
        tgt_pqr_deg = np.degrees(self._tgt_rates.update(tgt[3:6], t))

        alt_m = -float(own[2])                          # altitude = -D
        w = self._mpc_weight(alt_m)

        if boundary:
            # RL-step 마다 1회: reconstructor 갱신(HP/pqr/시간) + actor 재결정(캐시).
            # 에피소드 첫 boundary 는 advance 스킵(fresh recon, GPU 학습 reset 규약).
            if not self._first_boundary:
                self._rec.advance(own, tgt)
            self._first_boundary = False
            self._actor_cmd = self._actor_command(own, tgt)
        actor_cmd = self._actor_cmd

        if w > 0.0:
            mpc_cmd = self._mpc_command(own, tgt, own_pqr_deg, tgt_pqr_deg, t)   # 60Hz
            out = (1.0 - w) * actor_cmd + w * mpc_cmd
        else:
            out = actor_cmd

        if boundary:
            # 실제 적용된(블렌딩된) action 을 raw 로 되돌려 push → 관측 action 이력 일관.
            applied_raw = np.asarray(out, dtype=np.float64).copy()
            applied_raw[3] = 2.0 * applied_raw[3] - 1.0     # throttle [0,1] → [-1,1]
            self._rec.push_action(applied_raw)

        self._sub += 1
        if w <= 0.0:
            src = "altblend_actor"
        elif w >= 1.0:
            src = "altblend_mpc"
        else:
            src = "altblend_mix"
        return ActionResult(action=np.asarray(out, dtype=np.float32), source=src,
                            confidence=self.confidence,
                            info={"w": w, "alt_m": alt_m, "alt_ft": alt_m / FT_TO_M})

    def close(self) -> None:
        try:
            self.mpc.close()
        except Exception:
            pass


def make_altblend_provider(bundle_dir, *, mpc_root=_DEF_MPC_ROOT, mpc_config_path=None,
                           step_ratio=6, device="cpu", stochastic=True,
                           blend_hi_ft=4000.0, blend_lo_ft=2000.0):
    return AltBlendMPCProvider(
        bundle_dir, mpc_root=mpc_root, mpc_config_path=mpc_config_path,
        step_ratio=step_ratio, device=device, stochastic=stochastic,
        blend_hi_ft=blend_hi_ft, blend_lo_ft=blend_lo_ft)


__all__ = ["AltBlendMPCProvider", "make_altblend_provider"]
