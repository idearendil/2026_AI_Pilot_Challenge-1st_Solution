# -*- coding: utf-8 -*-
"""신경망 정책을 **매 substep(60Hz)** 로 굴리는 ActionProvider 래퍼.

배경
----
학습/기본 평가에서 정책은 RL-step(=STEP_RATIO substep=0.1s)마다 1회 결정하고 그 action 을
6 substep 동안 유지한다(10Hz 제어). 이 래퍼는 정책을 **매 substep(60Hz) 재결정**시켜 상대의
빠른 기동에 더 민감하게 반응하게 한다(예: 60Hz 로 도는 unreal_bt_client.exe 와 대등하게).

관측 일관성(중요)
-----------------
관측의 대부분(양 기체 위치/자세/속도 = 기하)은 매 substep 신선하게 반영해 60Hz 반응성을 준다.
그러나 학습 때 0.1s 간격으로 갱신되던 값들은 그 간격을 유지한다:
  - reconstructor 의 pqr/HP/fuel/시간 추정: **STEP_RATIO substep(0.1s)마다 1회만 advance**
    (그 사이엔 직전 값을 유지). advance 의 dt 는 학습과 동일한 0.1s.
  - action history(직전 K 개 내 action): 매 substep action 을 60Hz 큐에 쌓되, 관측에는
    **STEP_RATIO 간격으로 subsample** 한 K 개(=0.1s 간격, 학습과 동일)를 넣는다.

즉 "이전 action 6개를 0.1s 간격으로 받는다면, 60Hz 큐에 쌓아두고 6 substep 간격으로 하나씩
골라 넣는" 방식이다. pqr/HP/action-history 는 학습 분포를 유지하고, 결정 빈도만 60Hz 로 올린다.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult


class HighRateProvider(ActionProvider):
    def __init__(self, *, model, obs_rms, build_obs, recon, step_ratio: int, device: str,
                 explore: bool, to_continuous, to_command, uses_action_history: bool,
                 action_hist_len: int = 5, action_dim: int = 4, source: str = "high_rate"):
        self.model = model
        self.obs_rms = obs_rms
        self.build_obs = build_obs                  # (own, opp, recon) -> obs(np)
        self.recon = recon
        self.step = max(1, int(step_ratio))
        self.device = device
        self.explore = bool(explore)
        self.to_continuous = to_continuous          # (raw, num_bins) -> 연속 [-1,1]^4
        self.to_command = to_command                # raw[-1,1]^4 -> sim command(throttle[0,1])
        self.uses_hist = bool(uses_action_history)
        self.K = int(action_hist_len)
        self.adim = int(action_dim)
        self.q: deque = deque(maxlen=self.K * self.step)   # 60Hz raw-action 큐
        self._sub = 0
        self.source = source

    def reset(self, context: ActionContext | None = None) -> None:
        if self.recon is not None:
            self.recon.reset()
        self.q.clear()
        self._sub = 0

    def _subsampled_history(self) -> np.ndarray:
        """60Hz 큐 → 0.1s(STEP_RATIO) 간격 K 개(row0=가장 최근=STEP substep 전). 부족분 0 패딩."""
        H = np.zeros((self.K, self.adim), dtype=np.float64)
        ql = list(self.q)
        n = len(ql)
        for k in range(self.K):
            lag = (k + 1) * self.step
            if lag <= n:
                H[k] = ql[n - lag]
        return H

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        mean = np.asarray(self.obs_rms.mean, dtype=np.float64)
        var = np.asarray(self.obs_rms.var, dtype=np.float64)
        n = (np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + 1e-8)
        return np.clip(n, -10.0, 10.0).astype(np.float32)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        opp = np.asarray(context.target_state, dtype=np.float64)
        # pqr/HP/time 등은 0.1s(STEP_RATIO) 주기로만 advance(학습과 동일 dt·간격).
        if self.recon is not None and self._sub % self.step == 0:
            self.recon.advance(own, opp)
        # action history 는 60Hz 큐를 0.1s 간격으로 subsample 해 매 substep 주입.
        if self.uses_hist and self.recon is not None:
            self.recon.action_history = self._subsampled_history()
        obs = self.build_obs(own, opp, self.recon)
        obs_t = torch.as_tensor(self._normalize(obs), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.explore:
                if hasattr(self.model, "act_stochastic"):
                    raw = self.model.act_stochastic(obs_t)
                else:
                    raw, _, _, _ = self.model.get_action_and_value(obs_t)
                raw = raw.squeeze(0).cpu().numpy()
            else:
                raw = self.model.act_deterministic(obs_t).squeeze(0).cpu().numpy()
        if hasattr(self.model, "num_bins"):
            raw = self.to_continuous(raw, self.model.num_bins)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)[:self.adim]
        self.q.append(raw.copy())          # 매 substep(60Hz) 큐에 push
        self._sub += 1
        cmd = np.asarray(self.to_command(raw), dtype=np.float32)
        return ActionResult(action=cmd, source=self.source, confidence=0.9, info={})

    def close(self) -> None:
        return None


def high_rate_from_bundle(bundle_dir, *, step_ratio, device="cpu", explore=False):
    """claude164r(my_observation) 번들 → 60Hz HighRateProvider. (MLP 정책·exe_clone 등)"""
    from GeoMathUtil import GeometryInfo
    from claude_code.model import (load_bundle, discrete_indices_to_continuous,
                                   policy_action_to_command)
    from claude_code.normalizers import RunningMeanStd
    from claude_code import my_observation as MO

    model, meta = load_bundle(bundle_dir, device=device)
    if meta.get("observation_module") != "claude_code.my_observation":
        raise ValueError(
            f"high-rate 는 claude164r(my_observation) 번들만 지원합니다: {bundle_dir} "
            f"(observation_module={meta.get('observation_module')!r})")
    rms = (RunningMeanStd.from_state_dict(meta["obs_normalization"])
           if meta.get("obs_normalization") else None)
    geo = GeometryInfo()
    recon = MO.StateReconstructor()

    def build(own, opp, rec):
        return MO.build_observation(own, opp, geo, None, reconstructor=rec)

    return HighRateProvider(
        model=model, obs_rms=rms, build_obs=build, recon=recon, step_ratio=step_ratio,
        device=device, explore=explore, to_continuous=discrete_indices_to_continuous,
        to_command=policy_action_to_command, uses_action_history=True,
        action_hist_len=int(MO.ACTION_HISTORY_LEN), action_dim=int(MO.ACTION_DIM),
        source="high_rate_rl")


def high_rate_from_gylee(snapshot, *, step_ratio, device="cpu", explore=False):
    """model2_gylee(claude47r) → 60Hz HighRateProvider. 47D 관측엔 action history 가 없다."""
    from GeoMathUtil import GeometryInfo
    from model2_gylee.loader import load_model
    from model2_gylee import my_observation as GMO
    from model2_gylee.model import discrete_indices_to_continuous, policy_action_to_command

    model, obs_rms, _ = load_model(snapshot, device=device, verify_checksum=False)
    geo = GeometryInfo()
    recon = GMO.StateReconstructor()

    def build(own, opp, rec):
        return GMO.build_observation(own, opp, geo, None, reconstructor=rec)

    return HighRateProvider(
        model=model, obs_rms=obs_rms, build_obs=build, recon=recon, step_ratio=step_ratio,
        device=device, explore=explore, to_continuous=discrete_indices_to_continuous,
        to_command=policy_action_to_command, uses_action_history=False, source="high_rate_gylee")


__all__ = ["HighRateProvider", "high_rate_from_bundle", "high_rate_from_gylee"]
