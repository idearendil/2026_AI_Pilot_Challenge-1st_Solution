"""Self-play: 상대 전투기를 '같은 actor network' 로 조종하는 ActionProvider.

학습 중인 model 을 그대로 참조하므로 정책이 향상되면 상대도 같이 강해진다(완전한
self-learning). env 는 상대 관점(context.ownship_state=상대, context.target_state=본
기체)으로 context 를 주므로, 상대 관점의 관측을 만들어 model 에 통과시킨다.

핵심:
  - 상대는 자신만의 StateReconstructor 로 '상대 관점' HP/damage 를 누적한다(본 기체의
    전역 _RECON 과 독립).
  - action_repeat=step_ratio 로 RL-step 당 1회만 재계산(본 기체와 동일한 제어 주기).
  - 출력은 policy_action_to_command 로 변환(throttle [0,1]) 후 sim.step 에 직접 들어감.
"""
from __future__ import annotations

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from GeoMathUtil import GeometryInfo

from claude_code import my_observation
from claude_code.model import policy_action_to_command, discrete_indices_to_continuous
from claude_code.my_observation import StateReconstructor


class SelfPlayProvider(ActionProvider):
    def __init__(self, model, obs_rms, observation_fn=None, observation_mode="claude16r",
                 step_ratio: int = 6, device: str = "cpu", explore: bool = False):
        self.model = model
        self.obs_rms = obs_rms
        self.observation_fn = observation_fn        # env._observation_fn (None 이면 built-in)
        self.observation_mode = observation_mode
        self.repeat = max(1, int(step_ratio))
        self.device = device
        self.explore = explore
        self._geo = GeometryInfo()
        self._recon = StateReconstructor()
        self._count = 0
        self._cached: ActionResult | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self._recon.reset()
        self._count = 0
        self._cached = None

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        n = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(n, -10.0, 10.0).astype(np.float32)

    def _build_obs(self, own, opp) -> np.ndarray:
        # claude16 재구성 관측: 상대 관점 reconstructor 로 HP 누적 + 관측 생성
        if self.observation_fn is my_observation.build_observation:
            self._recon.advance(own, opp)
            return my_observation.build_observation(own, opp, self._geo, None,
                                                    reconstructor=self._recon)
        # 기타 custom 관측: 실제 state 값(학습 중엔 HP 등 존재)으로 생성
        if self.observation_fn is not None:
            return np.asarray(self.observation_fn(own, opp, self._geo, None), dtype=np.float32)
        # built-in (tactical16 등)
        from dogfight.envs.observation import build_observation as fw_build
        return np.asarray(fw_build(self.observation_mode, own, opp, self._geo), dtype=np.float32)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if self._cached is None or self._count % self.repeat == 0:
            own = np.asarray(context.ownship_state, dtype=np.float64)   # 상대 자신
            opp = np.asarray(context.target_state, dtype=np.float64)    # 본 기체
            obs = self._build_obs(own, opp)
            obs_t = torch.as_tensor(self._normalize(obs), dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            with torch.no_grad():
                if self.explore:
                    raw, _, _, _ = self.model.get_action_and_value(obs_t)
                    raw = raw.squeeze(0).cpu().numpy()
                else:
                    raw = self.model.act_deterministic(obs_t).squeeze(0).cpu().numpy()
            # 이산 정책이면 카테고리 index → 연속값으로 변환 후 command 변환.
            if hasattr(self.model, "num_bins"):
                raw = discrete_indices_to_continuous(raw, self.model.num_bins)
            cmd = policy_action_to_command(raw)
            self._cached = ActionResult(action=cmd, source="self_play", confidence=0.9, info={})
        self._count += 1
        return self._cached

    def close(self) -> None:
        return None


__all__ = ["SelfPlayProvider"]
