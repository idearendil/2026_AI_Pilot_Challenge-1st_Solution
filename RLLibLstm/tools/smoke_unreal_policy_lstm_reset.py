"""Smoke-check Unreal command policies propagate reset to action providers.

This is intentionally a no-Ray/no-UDP smoke. It verifies that policy reset hooks
used around Unreal MT_Init events call the underlying ActionProvider.reset(),
which is the episode-boundary contract needed for recurrent SAC inference.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


class FakeProvider:
    """Minimal provider that records reset calls."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self, context=None) -> None:
        self.reset_calls += 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clear_dogfight_modules() -> None:
    for name in list(sys.modules):
        if name == "dogfight" or name.startswith("dogfight."):
            del sys.modules[name]


def _check_tree(label: str, root: Path) -> None:
    _clear_dogfight_modules()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    try:
        policies = importlib.import_module("dogfight.unreal.policies")

        provider_policy_provider = FakeProvider()
        provider_policy = policies.ProviderCommandPolicy(
            action_provider=provider_policy_provider,
        )
        provider_policy.reset(None)

        lightweight_policy_provider = FakeProvider()
        lightweight_policy = policies.RLLightweightCommandPolicy(
            action_provider=lightweight_policy_provider,
        )
        lightweight_policy.reset(None)

        print(
            "[DogFightEnv][unreal_policy_reset_smoke] "
            f"{label} provider_reset_calls="
            f"{provider_policy_provider.reset_calls} "
            f"lightweight_reset_calls={lightweight_policy_provider.reset_calls}"
        )

        if provider_policy_provider.reset_calls != 1:
            raise RuntimeError(f"{label}: ProviderCommandPolicy did not reset provider")
        if lightweight_policy_provider.reset_calls != 1:
            raise RuntimeError(f"{label}: RLLightweightCommandPolicy did not reset provider")
    finally:
        sys.path = [
            item
            for item in sys.path
            if item not in {str(root), str(root / "src")}
        ]
        _clear_dogfight_modules()


def main() -> None:
    repo = _repo_root()
    _check_tree("MyTrainEnv", repo / "DogFightEnv" / "MyTrainEnv")
    _check_tree("Release", repo / "DogFightEnv" / "Release")
    print("[DogFightEnv][unreal_policy_reset_smoke] result=PASS")


if __name__ == "__main__":
    main()
