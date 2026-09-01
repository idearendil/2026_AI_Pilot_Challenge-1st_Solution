# -*- coding: utf-8 -*-
"""Stable_MPC_team_share(safe MPC) 를 상대(target)로 붙이기 위한 standalone ActionProvider.

`Stable_MPC_team_share` 패키지의 정본 컨트롤러는 `safe_mpc.provider.SafeMPCActionProvider`
(네이티브 F-16 predictor + CEM + safety shield)이고, 대회 UDP 경로에서는
`safe_mpc.command_policy.SafeMPCCommandPolicy` 가 매 프레임(60Hz) 다음을 채워 provider 를
호출한다:

  state[0:3] = NED pos(m),  state[3:6] = euler(roll,pitch,yaw deg),
  state[6:9] = body vel(u,v,w, m/s),
  state[9:12] = degrees(AngularRateEstimator.update(euler, t))   # pqr
  state[41]  = sim_time(s)  (첫 프레임 기준 상대시간)

이 래퍼는 그 `_to_state` 규약을 **그대로** 재현해, claude env 가 매 substep(60Hz) 넘겨주는
raw state([0:9] 만 유효)에 pqr·시간을 채워 SafeMPCActionProvider 를 호출한다. 따라서
power_test.py 의 target 으로 붙였을 때 대회 서버 구동 방식과 동일한 제어가 나온다.

env 는 target provider 를 매 substep 호출하고 context.ownship_state=상대(자기) 기체,
context.target_state=본 기체로 넘긴다(single_agent_env._step_target_aircraft). action_repeat
(=simulation_hz//policy_hz=6=10Hz replan) 은 provider 내부 tick 으로 처리된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

_ROOT = Path(__file__).resolve().parents[1]
_DEF_MPC_ROOT = _ROOT / "Stable_MPC_team_share"


def _load_stable_mpc(mpc_root: Path, mpc_config_path: str | None,
                     safety_config_path: str | None):
    """Stable_MPC_team_share 의 safe MPC provider 를 로드(경로 격리 후 복원).

    반환: (SafeMPCActionProvider 인스턴스, AngularRateEstimator 클래스, MPCConfig)."""
    root = Path(mpc_root).resolve()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.append(src)          # append: 메인 dogfight/observation import 을 가리지 않게
    from mpc.config import load_config
    from mpc.transforms import AngularRateEstimator
    from safe_mpc.config import load_safe_mpc_config
    from safe_mpc.provider import SafeMPCActionProvider

    mpc_cfg = load_config(mpc_config_path or str(root / "configs" / "mpc.yaml"))
    safe_cfg = load_safe_mpc_config(safety_config_path or str(root / "configs" / "safe_mpc.yaml"))
    provider = SafeMPCActionProvider(root, mpc_cfg, safe_cfg)
    return provider, AngularRateEstimator, mpc_cfg


class StableMPCProvider(ActionProvider):
    """Stable_MPC_team_share safe MPC 를 상대로 굴리는 60Hz provider.

    대회 SafeMPCCommandPolicy 와 동일하게 pqr(각속도 추정)·시간을 채워 호출한다."""

    def __init__(self, mpc_root: str | Path = _DEF_MPC_ROOT, *,
                 mpc_config_path: str | None = None,
                 safety_config_path: str | None = None):
        self.provider, RateEst, self.mpc_cfg = _load_stable_mpc(
            Path(mpc_root), mpc_config_path, safety_config_path)
        self._sim_hz = int(self.mpc_cfg.simulation_hz)
        self._own_rates = RateEst()
        self._tgt_rates = RateEst()
        self._first_sub: int | None = None
        self._sub = 0

    def _to_state(self, plane: np.ndarray, t: float, rate_est) -> np.ndarray:
        """raw env state([0:9] 유효) → SafeMPC 51-D state(pqr·시간 채움). 대회 _to_state 규약."""
        st = np.zeros(51, dtype=np.float64)
        s = np.asarray(plane, dtype=np.float64)
        st[0:9] = s[0:9]                                   # NED pos + euler(deg) + body vel
        st[9:12] = np.degrees(rate_est.update(s[3:6], t))  # pqr(deg); provider 가 rad 로 변환
        st[41] = t
        return st

    def reset(self, context: ActionContext | None = None) -> None:
        self.provider.reset(None)
        self._own_rates.reset()
        self._tgt_rates.reset()
        self._first_sub = None
        self._sub = 0

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = context.ownship_state
        tgt = context.target_state
        if own is None or tgt is None:
            return ActionResult(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                                "stable-mpc-fallback", 0.0, {"error": "missing state"})
        # 첫 호출 기준 상대시간(대회 first_frame 방식과 동일).
        if self._first_sub is None:
            self._first_sub = self._sub
        t = max(0.0, (self._sub - self._first_sub) / float(self._sim_hz))
        own_st = self._to_state(own, t, self._own_rates)
        tgt_st = self._to_state(tgt, t, self._tgt_rates)
        res = self.provider.compute_action(ActionContext(
            sim=None, opponent_sim=None,
            ownship_state=own_st, target_state=tgt_st,
            info={"frame_index": self._sub}))
        self._sub += 1
        return ActionResult(action=np.asarray(res.action, dtype=np.float32),
                            source=res.source, confidence=res.confidence, info=res.info)

    def close(self) -> None:
        try:
            self.provider.close()
        except Exception:
            pass


def make_stable_mpc_provider(mpc_root=_DEF_MPC_ROOT, *, mpc_config_path=None,
                             safety_config_path=None):
    return StableMPCProvider(mpc_root, mpc_config_path=mpc_config_path,
                             safety_config_path=safety_config_path)


__all__ = ["StableMPCProvider", "make_stable_mpc_provider"]
