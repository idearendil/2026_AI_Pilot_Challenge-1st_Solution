# -*- coding: utf-8 -*-
"""고도 안전망(altitude guard) 복합 ActionProvider.

평상시엔 **basic 번들**(claude164r MLP 정책)로 조종하고, 아군 기체의 고도가
임계값(기본 3000ft) 이하로 내려가면 **Release_MPC_team_share 의 MPC**(CEM + 네이티브
F-16 predictor)가 대신 조종한다.

제어 주기
--------
env 는 provider 를 매 substep(60Hz) 호출한다. 이 복합 provider 는 매 substep 호출을
받아 내부에서 두 컨트롤러의 제어 주기를 맞춘다:

  - **basic 번들**: 학습·제출과 동일하게 RL-step(=STEP_RATIO substep=0.1s)마다 1회
    결정(stochastic 샘플링)하고 그 사이엔 같은 command 를 유지 → 10Hz.
  - **MPC**: 매 substep 재호출 → 60Hz. (MPC 자체는 policy_hz=10 으로 내부 replan,
    출력은 매 프레임 갱신하는 것이 대회 서버 구동 방식과 동일하다.)

모드 결정은 RL-step 경계(0.1s)마다 고도로 판정해 그 RL-step 동안 유지한다.

관측 일관성
----------
basic 정책이 보는 claude164r 관측은 StateReconstructor(HP/fuel/pqr/시간/직전 action
이력) 를 필요로 한다. MPC 가 조종하는 동안에도 reconstructor 를 **RL-step 마다 1회**
advance 하고 실제 적용된 action(raw)을 push 해, 고도를 회복해 basic 으로 복귀할 때
관측(HP·action 이력)이 끊기지 않게 한다.

MPC state 규약
--------------
MPC provider 는 state[0:3]=NED(pos, D=down+), [3:6]=euler(deg), [6:9]=body vel(u,v,w),
[9:12]=pqr(deg), [41]=sim_time(s) 를 기대한다. env 의 JSBSim state 는 [0:9] 를 그대로
쓰고, pqr 은 자세 유한차분(AngularRateEstimator, 60Hz)으로, 시간은 substep 카운터로
채워 넣는다(대회 MPCCommandPolicy 와 동일 방식).
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
    added = src not in sys.path
    if added:
        sys.path.append(src)          # append: 메인 dogfight/observation import 을 가리지 않게
    try:
        from mpc.config import load_config
        from mpc.provider import MPCActionProvider as TeamShareMPC
    finally:
        pass
    cfg = load_config(config_path) if config_path else load_config(
        str(Path(mpc_root).resolve() / "configs" / "mpc.yaml"))
    return TeamShareMPC(Path(mpc_root).resolve(), cfg), cfg


class AltGuardMPCProvider(ActionProvider):
    """basic 번들(10Hz) + 저고도 시 team-share MPC(60Hz) 복합 provider."""

    def __init__(self, bundle_dir: str | Path, *, mpc_root: str | Path = _DEF_MPC_ROOT,
                 mpc_config_path: str | None = None, step_ratio: int = 6,
                 device: str = "cpu", stochastic: bool = True,
                 guard_altitude_ft: float = 3000.0, confidence: float = 0.9):
        from GeoMathUtil import GeometryInfo
        from claude_code.model import (load_bundle, make_obs_normalizer,
                                       discrete_indices_to_continuous,
                                       policy_action_to_command)
        from claude_code import my_observation as MO

        self._MO = MO
        self._to_cont = discrete_indices_to_continuous
        self._to_cmd = policy_action_to_command
        self.step = max(1, int(step_ratio))
        self.device = device
        self.stochastic = bool(stochastic)
        self.thr_m = float(guard_altitude_ft) * FT_TO_M
        self.confidence = float(confidence)

        # ── basic 번들 정책 ─────────────────────────────────────────────────
        self.model, self.meta = load_bundle(str(bundle_dir), device=device)
        if self.meta.get("observation_module") != "claude_code.my_observation":
            raise ValueError(
                "AltGuardMPCProvider 의 basic 번들은 claude164r(my_observation) 이어야 "
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
        self._mode = "basic"
        self._cached = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # ---------------------------------------------------------------- helpers
    def _basic_command(self, own: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """claude164r 관측 → basic 정책 → command. reconstructor 에 raw action push."""
        obs = self._MO.build_observation(own, tgt, self._geo, None, reconstructor=self._rec)
        obs = self._normalize(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            act_fn = self.model.act_stochastic if self.stochastic else self.model.act_deterministic
            raw = act_fn(obs_t).squeeze(0).cpu().numpy()
        if hasattr(self.model, "num_bins"):
            raw = self._to_cont(raw, self.model.num_bins)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)[:4]
        self._rec.push_action(raw)                     # 다음 관측 action 이력용
        return np.asarray(self._to_cmd(raw), dtype=np.float32)

    def _mpc_state(self, s9: np.ndarray, pqr_deg: np.ndarray, t: float) -> np.ndarray:
        st = np.zeros(51, dtype=np.float64)
        st[0:9] = np.asarray(s9, dtype=np.float64)[0:9]     # NED pos + euler(deg) + body vel
        st[9:12] = np.asarray(pqr_deg, dtype=np.float64)    # deg (provider 가 rad 로 변환)
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
        self._mode = "basic"
        self._cached = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        tgt = np.asarray(context.target_state, dtype=np.float64)
        t = self._sub / float(self._MO.SIM_HZ)          # substep sim time (60Hz 격자)
        boundary = (self._sub % self.step == 0)

        # 각속도 추정기는 매 substep(60Hz) 갱신 → MPC 진입 시 pqr 이 항상 최신(대회 방식).
        own_pqr_deg = np.degrees(self._own_rates.update(own[3:6], t))
        tgt_pqr_deg = np.degrees(self._tgt_rates.update(tgt[3:6], t))

        alt_m = -float(own[2])                          # altitude = -D

        if boundary:
            # RL-step 마다 1회: reconstructor 갱신(HP/pqr/시간) + 모드 판정.
            self._rec.advance(own, tgt)
            self._mode = "mpc" if alt_m <= self.thr_m else "basic"
            if self._mode == "mpc":
                cmd = self._mpc_command(own, tgt, own_pqr_deg, tgt_pqr_deg, t)
                raw = cmd.astype(np.float64).copy(); raw[3] = 2.0 * raw[3] - 1.0
                self._rec.push_action(raw)              # 적용 action 이력 유지(basic 복귀 대비)
                self._cached = cmd
            else:
                self._cached = self._basic_command(own, tgt)
            out = self._cached
        else:
            if self._mode == "mpc":
                out = self._mpc_command(own, tgt, own_pqr_deg, tgt_pqr_deg, t)  # 60Hz
            else:
                out = self._cached                      # basic 은 10Hz 유지

        self._sub += 1
        src = "altguard_mpc" if self._mode == "mpc" else "altguard_basic"
        return ActionResult(action=np.asarray(out, dtype=np.float32), source=src,
                            confidence=self.confidence, info={"mode": self._mode,
                                                              "alt_m": alt_m})

    def close(self) -> None:
        try:
            self.mpc.close()
        except Exception:
            pass


def make_altguard_provider(bundle_dir, *, mpc_root=_DEF_MPC_ROOT, mpc_config_path=None,
                           step_ratio=6, device="cpu", stochastic=True,
                           guard_altitude_ft=3000.0):
    return AltGuardMPCProvider(
        bundle_dir, mpc_root=mpc_root, mpc_config_path=mpc_config_path,
        step_ratio=step_ratio, device=device, stochastic=stochastic,
        guard_altitude_ft=guard_altitude_ft)


__all__ = ["AltGuardMPCProvider", "make_altguard_provider"]
