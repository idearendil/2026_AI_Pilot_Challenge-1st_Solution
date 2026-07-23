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
from claude_code.bt_rule import BT_RULE_DEFAULTS, DEFAULT_BT_DLL, check_rule_applied
from claude_code.model import policy_action_to_command, discrete_indices_to_continuous
from claude_code.my_observation import StateReconstructor


def make_bt_provider(dll_name: str = DEFAULT_BT_DLL, rule_xml: str = ""):
    """baseline BT(DLL) 상대를 opponent pool 후보로 쓰기 위한 ActionProvider 를 만든다.

    rule XML(AIP_RULE_XML)은 여기서 세팅해도 **이미 늦다** — JSBSimAIPLib.dll 이 로드되는
    claude_code.env_utils import 시점에 한 번만 읽히기 때문이다(claude_code.bt_rule 참고).
    그래서 여기서는 세팅 대신 **검증**만 한다: rule 이 안 걸렸으면 조용히 Task_Empty BT
    (=직진만 하는 표적, 가짜 승률 100%)로 학습하는 대신 즉시 실패시킨다.

    entrypoint(train.py 등)는 claude_code import 보다 먼저 bt_rule.apply_rule_env() 를
    호출해야 하고, Ray worker 는 ray.init 의 runtime_env env_vars 로 주입받는다.
    """
    from dogfight.ai.bt_action_provider import BTActionProvider

    check_rule_applied(dll_name, rule_xml)
    return BTActionProvider(dll_name=dll_name)


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


class PoolSelfPlayProvider(ActionProvider):
    """opponent pool 에서 매 episode 하나를 (가중) 샘플링해 상대로 쓰는 provider.

    pool 은 SelfPlayProvider 리스트(오래된→최신 순). env.reset 이 부르는 reset()마다
    weights 로 index 하나를 뽑아 그 sub-provider 로 해당 episode 를 진행한다.

    compute_action 은 episode 시작(reset) 시 고정한 sub-provider 객체(_active)에
    위임한다. 따라서 iteration 경계에서 pool 이 바뀌어도(추가/제거) 진행 중이던
    episode 는 원래 상대로 일관되게 끝난다. last_index 는 방금 진행한 episode 가 쓴
    opponent 의 (샘플 당시) pool index 로, 승패를 opponent 별로 귀속할 때 쓴다.

    가중치는 'EMA 승률이 낮은 opponent 일수록 크게' 주어 자주 뽑히게 한다(호출부에서
    softmax(-ema/τ) 등으로 계산해 set_weights 로 전달).

    **슬롯 규약**: baseline BT 상대를 쓰는 경우 항상 index 0 에 고정하고(=bt_slots=1),
    학습 snapshot 후보는 index 1.. 에 오래된→최신 순으로 놓는다. pool 초과 시 제거는
    index bt_slots(=가장 오래된 snapshot)부터 하므로 BT 는 절대 evict 되지 않는다.
    (제거 로직은 호출부 = PPOTrainer / RolloutWorker 에 있다.)
    """

    def __init__(self, providers, weights=None, seed: int = 0):
        self.providers = list(providers)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self._rng = np.random.default_rng(int(seed))
        self.current = 0
        self.last_index = 0
        self._active = self.providers[0] if self.providers else None

    def set_pool(self, providers, weights=None) -> None:
        self.providers = list(providers)
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)
        if self.current >= len(self.providers):
            self.current = max(0, len(self.providers) - 1)

    def set_weights(self, weights) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    def _sample_index(self) -> int:
        n = len(self.providers)
        if n <= 1:
            return 0
        w = self.weights
        if (w is None or len(w) != n or not np.all(np.isfinite(w))
                or np.any(w < 0) or float(w.sum()) <= 1e-12):
            p = None                      # weights 이상하면 균등 샘플
        else:
            p = np.asarray(w, dtype=np.float64) / float(w.sum())
        return int(self._rng.choice(n, p=p))

    def reset(self, context: ActionContext | None = None) -> None:
        self.current = self._sample_index()
        self.last_index = self.current
        self._active = self.providers[self.current] if self.providers else None
        if self._active is not None:
            self._active.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        return self._active.compute_action(context)

    def close(self) -> None:
        for p in self.providers:
            try:
                p.close()
            except Exception:
                pass


__all__ = ["SelfPlayProvider", "PoolSelfPlayProvider", "make_bt_provider",
           "BT_RULE_DEFAULTS", "DEFAULT_BT_DLL"]
