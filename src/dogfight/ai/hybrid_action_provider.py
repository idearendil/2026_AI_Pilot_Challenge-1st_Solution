from __future__ import annotations

from typing import Callable

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action


class HybridActionProvider(ActionProvider):
    def __init__(
        self,
        primary_provider: ActionProvider,
        secondary_provider: ActionProvider,
        mode: str = "residual",
        alpha: float = 0.5,
        residual_scale: float = 0.35,
        selector: Callable[[ActionContext, ActionResult, ActionResult], str | bool] | None = None,
        confidence: float = 0.95,
    ):
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.mode = mode
        self.alpha = alpha
        self.residual_scale = residual_scale
        self.selector = selector
        self.confidence = confidence

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary_provider.reset(context)
        self.secondary_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        primary_result = self.primary_provider.compute_action(context)
        secondary_result = self.secondary_provider.compute_action(context)

        if self.mode == "switch":
            decision = self.selector(context, primary_result, secondary_result) if self.selector else "primary"
            use_primary = decision if isinstance(decision, bool) else decision != "secondary"
            chosen = primary_result if use_primary else secondary_result
            return ActionResult(
                action=clip_action(chosen.action),
                source="hybrid",
                confidence=self.confidence,
                info={
                    "mode": self.mode,
                    "selected": chosen.source,
                    "primary": primary_result.info,
                    "secondary": secondary_result.info,
                },
            )

        if self.mode == "blend":
            action = self.alpha * primary_result.action + (1.0 - self.alpha) * secondary_result.action
        else:
            action = secondary_result.action + self.residual_scale * primary_result.action

        return ActionResult(
            action=clip_action(action),
            source="hybrid",
            confidence=self.confidence,
            info={
                "mode": self.mode,
                "alpha": self.alpha,
                "residual_scale": self.residual_scale,
                "primary_source": primary_result.source,
                "secondary_source": secondary_result.source,
            },
        )

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()
