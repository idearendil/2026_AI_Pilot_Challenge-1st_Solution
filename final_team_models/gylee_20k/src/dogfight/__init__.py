"""dogfight 패키지.

`DogFightEnv` 는 JSBSimAIPLib.dll(+디버그 CRT)을 로드하는 무거운 시뮬레이터를
끌어온다. 패키지 import 만으로 그 DLL 이 로드되면, 시뮬레이터가 전혀 필요 없는
경로(대회 제출 UDP 클라이언트 등)까지 JSBSim/디버그 CRT 에 묶여 버린다. 그래서
`DogFightEnv` 를 **지연(lazy) import** 한다 — 실제로 `dogfight.DogFightEnv` 를
접근할 때만 시뮬레이터를 로드한다.

기존 코드는 대부분 `from dogfight.envs.single_agent_env import DogFightEnv` 처럼
서브모듈을 직접 import 하므로 영향이 없고, `from dogfight import DogFightEnv` 도
아래 __getattr__ 로 그대로 동작한다.
"""
from __future__ import annotations

__all__ = ["DogFightEnv"]


def __getattr__(name: str):
    if name == "DogFightEnv":
        from .envs.single_agent_env import DogFightEnv
        return DogFightEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
