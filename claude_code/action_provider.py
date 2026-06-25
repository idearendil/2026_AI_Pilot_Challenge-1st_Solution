"""claude_code MLP 정책을 원본 ActionProvider 계약으로 감싼다.

원본 `RLActionProvider` 와 동일한 인터페이스(`reset`, `compute_action`, `close`)를
구현하므로, 원본 `ProviderCommandPolicy` / `UnrealAIPilotUDPClient` 와 그대로
연결된다. 즉 대결 서버 입장에서는 원본 RL 제출과 완전히 동일하게 동작한다.

관측은 `ProviderCommandPolicy` 가 `build_observation("tactical16", ...)` 로
만들어 `context.observation` 에 넣어주며, 이는 학습 환경의 관측 파이프라인과
동일한 함수다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

from claude_code.model import load_bundle, make_obs_normalizer, policy_action_to_command


class MLPActionProvider(ActionProvider):
    def __init__(self, bundle_dir: str | Path, device: str = "cpu", confidence: float = 0.9):
        self.bundle_dir = str(bundle_dir)
        self.device = device
        self.confidence = confidence
        self.model, self.metadata = load_bundle(bundle_dir, device=device)
        self.obs_dim = int(self.metadata.get("observation_size", 16))
        # 학습 때와 동일한 관측 정규화 적용 (없으면 항등).
        self._normalize_obs = make_obs_normalizer(self.metadata.get("obs_normalization"))
        # claude_code.my_observation 을 쓴 번들이면 HP/damage 재구성을 RL-step 당 1회
        # 갱신한다 (다른 관측 모듈에는 영향 없음).
        self._reconstruct = self.metadata.get("observation_module") == "claude_code.my_observation"
        if self._reconstruct:
            from claude_code.my_observation import reset_reconstructor, advance_reconstructor
            self._reset_recon = reset_reconstructor
            self._advance_recon = advance_reconstructor

    def reset(self, context: ActionContext | None = None) -> None:
        # MLP 정책은 recurrent state 가 없으므로 reset 시 별도 처리 불필요.
        if self._reconstruct:
            self._reset_recon()

    def compute_action(self, context: ActionContext) -> ActionResult:
        # HP/damage 재구성을 RL-step 당 1회 갱신 (관측 빌드 전에 호출되어도 1-step lag 로
        # 학습 경로와 동일). context 에 양측 state 가 채워져 있을 때만.
        if (self._reconstruct and context.ownship_state is not None
                and context.target_state is not None):
            self._advance_recon(context.ownship_state, context.target_state)

        observation = context.observation
        if observation is None:
            raise ValueError(
                "MLPActionProvider 는 context.observation 이 필요합니다 "
                "(ProviderCommandPolicy 가 tactical16 관측을 채워줍니다)."
            )
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"관측 차원 불일치: got {obs.shape[0]}, expected {self.obs_dim}"
            )

        obs = self._normalize_obs(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.model.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()
        command = policy_action_to_command(raw)  # [roll,pitch,rudder]∈[-1,1], throttle∈[0,1]

        return ActionResult(
            action=command,
            source="claude_code_ppo",
            confidence=self.confidence,
            info={"bundle_dir": self.bundle_dir},
        )

    def close(self) -> None:
        return None


__all__ = ["MLPActionProvider"]
