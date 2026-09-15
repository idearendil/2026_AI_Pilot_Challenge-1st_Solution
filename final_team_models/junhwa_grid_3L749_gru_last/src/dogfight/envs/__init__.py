"""dogfight.envs 패키지.

`DogFightEnv`(→ single_agent_env → FighterSim → JSBSimWrapper → JSBSimAIPLib.dll)
를 지연(lazy) import 한다. 그래야 `dogfight.envs.observation` 같은 순수 모듈을
import 할 때 JSBSim/디버그 CRT 가 딸려오지 않는다. 서브모듈을 직접 import 하던
기존 코드는 영향 없음.
"""
from __future__ import annotations

__all__ = ["DogFightEnv"]


def __getattr__(name: str):
    if name == "DogFightEnv":
        from .single_agent_env import DogFightEnv
        return DogFightEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
