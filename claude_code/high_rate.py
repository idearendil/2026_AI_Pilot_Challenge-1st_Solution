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
학습 때 0.1s(STEP_RATIO substep) 간격으로 정의되던 값들도 **그 0.1s 간격을 유지**하되, 미분량
(가속도·각속도)은 '현재 시점' 기준으로 신선하게 유지한다:
  - **미분량(가속도·pqr)**: 매 substep **'현재 상태 − 정확히 0.1s(STEP_RATIO substep) 전
    상태' 의 sliding 차분**으로 재계산한다(60Hz 상태 큐를 STEP_RATIO 만큼 되돌아본다).
    → 관측의 신선한 속도/자세와 **끝점이 일치**한다. (예전엔 advance() 에서만 계산돼 경계
    사이 5 substep 동안 '직전 경계 값'에 동결 → 신선한 속도/자세와 불일치했다. 학습(10Hz)은
    항상 '현재로 끝나는 0.1s 차분'을 보므로, 동결 값은 off-distribution 이었다.)
  - **순간량(damage 지시 dmg_dealt/taken)**: 현재 기하(거리·ATA)로 **매 substep 재계산**한다.
    가속도·pqr 과 같은 이유로 '현재 상태' 기준이라야 관측의 신선한 aim 기하와 정합한다.
  - **적분량(HP/연료/시간)**: **0.1s(=STEP_RATIO substep) 경계에서만 dt=0.1s 로 적분**한다
    (학습과 동일한 advance()). 60Hz(dt=1/60) 로 촘촘히 적분해봤으나 승률이 떨어져 되돌렸다.
  - action history(직전 K 개 action): 매 substep action 을 60Hz 큐에 쌓되, 관측에는
    **STEP_RATIO 간격으로 subsample** 한 K 개(=0.1s 간격, 학습과 동일)를 넣는다.

즉 미분량(가속도·pqr)·순간량(damage 지시)·action history 는 매 substep 현재 상태 기준으로 신선하게
만들고(끝점=현재, 0.1s 창은 학습과 동일), HP/연료/시간(적분량)은 0.1s 경계 적분을 유지하며,
결정 빈도를 60Hz 로 올린다. 다른 reconstructor(예: gylee 47D)는 기존 0.1s 경계 advance 를 유지한다.
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
        # 에피소드 첫 관측은 advance 없이 fresh recon(GPU 학습 reset→build 규약).
        self._first_boundary = True
        # ── 60Hz 재구성 (claude164r StateReconstructor 전용) ────────────────────────
        # 재구성이 필요한 feature 를 성격별로 학습 분포에 맞게 만든다:
        #   · 미분량(가속도·pqr): '현재 − 0.1s(STEP substep) 전' sliding 차분(아래 60Hz 상태 큐).
        #   · 순간량(damage 지시): 현재 기하로 매 substep 재계산.
        #   · 적분량(HP/연료/시간): **0.1s(STEP substep) 경계에서만 dt=0.1s 로 적분**(advance()).
        #     — 60Hz(dt=1/60) 적분은 승률이 떨어져 학습(0.1s)과 동일하게 되돌렸다.
        # 다른 reconstructor(예: gylee 47D)는 자체 규약이 있어 기존 0.1s 경계 advance 를 유지한다.
        from claude_code.my_observation import StateReconstructor as _MORec
        self._mo = isinstance(self.recon, _MORec)
        self._rate_q: deque = deque(maxlen=self.step + 1)  # (own_vel_ned, own_att, tgt_vel_ned, tgt_att)
        if self._mo:
            from claude_code.my_observation import (
                StateIndex as _SI, SIM_HZ as _HZ, METER_TO_FEET as _M2FT,
                damage_rate as _dmg, _ned_to_body_matrix as _n2b, _estimate_pqr as _epqr)
            self._ned2body = _n2b
            self._est_pqr = _epqr
            self._dmg_rate = _dmg
            self._rpy = (int(_SI.ROLL), int(_SI.PITCH), int(_SI.YAW))
            self._rate_dt = self.step / float(_HZ)   # 0.1s: 미분량 sliding 창(학습 RL-step 간격)
            self._M2FT = float(_M2FT)                 # damage 지시 계산용(거리 m→ft)

    def reset(self, context: ActionContext | None = None) -> None:
        if self.recon is not None:
            self.recon.reset()
        self.q.clear()
        self._rate_q.clear()
        self._sub = 0
        self._first_boundary = True

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

    def _recompute_dmg(self, own: np.ndarray, opp: np.ndarray) -> None:
        """순간량 damage 지시(dmg_dealt/taken)를 매 substep '현재 기하'로 재계산해
        reconstructor 필드에 반영한다(claude164r 전용). advance() 가 0.1s 경계에서 넣는
        동결 값을 대체해, 관측의 신선한 aim 기하(거리·ATA)와 정합시킨다.

        damage_rate 는 순간 함수(적분 아님)라 매 substep 현재 값으로 두는 게 학습(그 스텝의
        거리·ATA 로 계산)과 같은 의미다. t_sec 은 적분량이라 advance() 가 0.1s 로만 올린다
        (≤0.1s staleness → tier 임계값이 초 단위라 무시 가능). HP/연료는 여기서 건드리지 않는다."""
        rec = self.recon
        geo = rec._geo
        r_ft = geo._get_distance(own, opp) * self._M2FT
        ata_own = geo._get_antenna_train_angle(own, opp, False)   # 내가 상대를 겨눔
        ata_tgt = geo._get_antenna_train_angle(opp, own, False)   # 상대가 나를 겨눔
        rec.last_dmg_dealt = self._dmg_rate(r_ft, ata_own, rec.t_sec)   # 내가 가하는 초당 피해율
        rec.last_dmg_taken = self._dmg_rate(r_ft, ata_tgt, rec.t_sec)   # 내가 받는 초당 피해율

    def _recompute_sliding_rates(self, own: np.ndarray, opp: np.ndarray) -> None:
        """가속도·각속도(pqr)를 매 substep '현재 − 정확히 0.1s(STEP substep) 전' 차분으로
        재계산해 reconstructor 필드에 반영한다(claude164r 전용).

        미분량만 다룬다: 0.1s 이력이 아직 없으면(에피소드 초반 STEP substep 이내) 0 으로 둔다
        (학습 첫 step = prev None → 0 규약과 동일). 값 자체는 학습 advance() 가 경계에서
        계산하는 '현재−0.1s전' 차분과 동일 수식이며, 여기선 매 substep 끝점을 현재로 갱신한다."""
        r, p, y = self._rpy
        rec = self.recon
        own_vel_ned = self._ned2body(own[r], own[p], own[y]).T @ own[6:9]
        tgt_vel_ned = self._ned2body(opp[r], opp[p], opp[y]).T @ opp[6:9]
        own_att = np.array([own[r], own[p], own[y]], dtype=np.float64)
        tgt_att = np.array([opp[r], opp[p], opp[y]], dtype=np.float64)
        self._rate_q.append((own_vel_ned, own_att, tgt_vel_ned, tgt_att))
        if len(self._rate_q) > self.step:      # 정확히 STEP substep(=0.1s) 전 샘플이 존재
            ov0, oa0, tv0, ta0 = self._rate_q[0]
            dt = self._rate_dt
            rec.own_accel_est = (own_vel_ned - ov0) / dt
            rec.tgt_accel_est = (tgt_vel_ned - tv0) / dt
            rec.own_pqr_est = self._est_pqr(oa0, own_att, dt)
            rec.tgt_pqr_est = self._est_pqr(ta0, tgt_att, dt)
        else:
            rec.own_accel_est = np.zeros(3, dtype=np.float64)
            rec.tgt_accel_est = np.zeros(3, dtype=np.float64)
            rec.own_pqr_est = np.zeros(3, dtype=np.float64)
            rec.tgt_pqr_est = np.zeros(3, dtype=np.float64)

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
        # ── 재구성 feature 갱신 ─────────────────────────────────────────────────
        # claude164r: 적분량(HP/연료/시간)은 0.1s(STEP) 경계에서만 dt=0.1s 로 적분(학습과 동일),
        # 순간량(damage 지시)·미분량(가속도·pqr)은 매 substep 현재 상태 기준으로 신선하게 만든다.
        # 에피소드 첫 substep 은 적분/재계산 전(fresh recon, GPU 학습 reset→build 규약).
        if self._mo:
            if not self._first_boundary:
                if self._sub % self.step == 0:
                    self.recon.advance(own, opp)     # HP/연료/시간 0.1s 적분(+accel/pqr/dmg 는 아래서 override)
                self._recompute_dmg(own, opp)        # damage 지시: 매 substep 현재 기하로 fresh
            self._first_boundary = False
            self._recompute_sliding_rates(own, opp)  # 가속도·pqr = '현재 − 0.1s 전' sliding 차분
        elif self.recon is not None:
            # 기타 reconstructor(gylee 등): 기존 0.1s 경계 advance 규약 유지.
            if self._sub % self.step == 0:
                if not self._first_boundary:
                    self.recon.advance(own, opp)
                self._first_boundary = False
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


def high_rate_from_model(model, obs_rms, *, step_ratio, device="cpu", explore=False):
    """이미 로드된 claude164r(my_observation) 정책 모델 → 60Hz HighRateProvider.

    번들(high_rate_from_bundle)과 달리 in-memory model+obs_rms 를 그대로 받는다
    (runs 체크포인트를 gpu_ckpt_to_bundle.load_ckpt_as_model 로 로드해 60Hz 상대로 붙일 때)."""
    from GeoMathUtil import GeometryInfo
    from claude_code.model import discrete_indices_to_continuous, policy_action_to_command
    from claude_code import my_observation as MO

    geo = GeometryInfo()
    recon = MO.StateReconstructor()

    def build(own, opp, rec):
        return MO.build_observation(own, opp, geo, None, reconstructor=rec)

    return HighRateProvider(
        model=model, obs_rms=obs_rms, build_obs=build, recon=recon, step_ratio=step_ratio,
        device=device, explore=explore, to_continuous=discrete_indices_to_continuous,
        to_command=policy_action_to_command, uses_action_history=True,
        action_hist_len=int(MO.ACTION_HISTORY_LEN), action_dim=int(MO.ACTION_DIM),
        source="high_rate_rl")


__all__ = ["HighRateProvider", "high_rate_from_bundle", "high_rate_from_model"]
