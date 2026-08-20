"""obs-WM neural-MPC 정책을 원본 ActionProvider 계약으로 감싼다.

MLPActionProvider(단일 정책)와 달리, 매 RL-step 에 `ObsMPCPlanner.plan_fast` 로 1~2초
lookahead rollout 을 굴려 first-action 을 고른다. WM=추론호환 obs-WM(wm_model_obs.pt),
actor/critic=team01/basic 번들(→ac_ckpt). 관측 불가한 pqr 은 StateReconstructor 추정으로,
FCS 은닉상태는 action 이력(command window)으로 대체 — C++ MPC 와 동일 계약.

seed 구성:
  hp/fuel/t_sec        ← StateReconstructor 누적
  prev_own/tgt_pqr     ← reconstructor 자세차분 추정(own_pqr_est/tgt_pqr_est)
  my_act_hist          ← reconstructor.action_history(raw [-1,1]) flatten (20,)
  opp_act_hist         ← 0 (상대 명령 관측 불가; rollout 은 self-play actor 로 상대 생성)
초기 state 의 pqr(9:12)도 reconstructor 추정으로 주입(WM 첫 스텝 입력).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

from claude_code.obs_mpc_planner import WMObs, ObsMPCPlanner


class MPCActionProvider(ActionProvider):
    def __init__(self, wm_ckpt: str | Path, ac_ckpt: str | Path, device: str = "cuda",
                 K: int = 12, M: int = 8, H: int = 20, decide_every: int = 1,
                 gamma: float = 0.98, confidence: float = 0.9, use_fast: bool = True):
        from claude_code.my_observation import (StateReconstructor, reset_reconstructor,
                                                advance_reconstructor, push_action_reconstructor,
                                                get_reconstructor)
        self._reset_recon = reset_reconstructor
        self._advance_recon = advance_reconstructor
        self._push_action = push_action_reconstructor
        self._get_recon = get_reconstructor
        self.device = device
        self.confidence = confidence
        self.use_fast = bool(use_fast)
        wm = WMObs(str(wm_ckpt), device)
        self.planner = ObsMPCPlanner(wm, str(ac_ckpt), device=device, K=K, M=M, H=H,
                                     decide_every=decide_every, gamma=gamma)
        self.H = H

    def reset(self, context: ActionContext | None = None) -> None:
        self._reset_recon()

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64).copy()
        tgt = np.asarray(context.target_state, dtype=np.float64).copy()
        # HP/damage/pqr/action 이력 갱신 (RL-step 당 1회; 관측 빌드 전 규약과 동일)
        self._advance_recon(own, tgt)
        rec = self._get_recon()
        # 관측 불가 pqr(9:12) 을 reconstructor 추정으로 주입 (WM 첫 스텝 입력)
        own[9:12] = np.asarray(rec.own_pqr_est, dtype=np.float64)
        tgt[9:12] = np.asarray(rec.tgt_pqr_est, dtype=np.float64)
        seed = {
            "hp_own": float(rec.hp_own), "hp_tgt": float(rec.hp_tgt),
            "fuel_own": float(rec.fuel_own), "fuel_tgt": float(rec.fuel_tgt),
            "t_sec": float(rec.t_sec),
            "prev_own_pqr": np.asarray(rec.own_pqr_est, np.float64),
            "prev_tgt_pqr": np.asarray(rec.tgt_pqr_est, np.float64),
            "my_act_hist": np.asarray(rec.action_history, np.float64).reshape(-1),
            "opp_act_hist": np.zeros(20, np.float64),
        }
        planfn = self.planner.plan_fast if self.use_fast else self.planner.plan
        command = np.asarray(planfn(own, tgt, seed), dtype=np.float64)   # throttle∈[0,1]
        # reconstructor 에 raw action([-1,1]^4) push (다음 관측/이력용)
        raw = command.copy(); raw[3] = 2.0 * command[3] - 1.0
        self._push_action(raw)
        return ActionResult(action=command.astype(np.float32), source="claude_code_obs_mpc",
                            confidence=self.confidence, info={"H": self.H})

    def close(self) -> None:
        return None


__all__ = ["MPCActionProvider"]
