"""표준 DogFight 환경 구성 헬퍼.

원본 학생 템플릿 `experiments/student_ppo_mlp.yaml`과 동일한 환경 세팅을
재현한다. 즉:

  - observation_mode: tactical16   (16차원, [-1, 1] 정규화)
  - action: Box([-1, 1]^4)         (roll, pitch, rudder, throttle)
  - target_mode: fixed             (표적은 고정 입력으로 직진/정상 비행)
  - step_ratio: 6                  (RL action 1회당 sim 6 step 유지)
  - max_engage_time: 60s, episode_step_limit: 3600

원본 `DogFightWrapper`를 그대로 인스턴스화하므로 동역학, 보상, 종료 조건,
관측 파이프라인이 RLlib 학습 경로와 100% 동일하다. 차이는 학습 루프(RLlib
대신 claude_code PPO)뿐이다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Release 루트와 src 를 import 경로에 추가 (원본 my_submission.py 와 동일한 방식).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from DogFightEnvWrapper import DogFightWrapper  # noqa: E402


# experiments/student_ppo_mlp.yaml 의 env / env_config 섹션과 동일한 기본값.
STANDARD_ENV_CONFIG = {
    "observation_mode": "tactical16",
    "target_mode": "fixed",
    "target_behavior_dll": "AIP_BASE_target.dll",
    "ownship_control_mode": "rl",
    "max_engage_time": 60.0,
    "episode_step_limit": 3600,
    "step_ratio": 6,
    "reward": {
        "mode": "default",
        "step_penalty": -0.01,
        "damage_scale": 20.0,
        "pursuit_scale": 0.3,
        "low_altitude_penalty": 0.1,
        "win_reward": 100.0,
        "loss_reward": -100.0,
        "draw_reward": -30.0,
    },
}

OBSERVATION_MODE = "tactical16"
OBSERVATION_SIZE = 16
ACTION_SIZE = 4


def resolve_hooks(reward_module: str = "", observation_module: str = ""):
    """claude_code(또는 student) 보상/관측 모듈을 로드해 hook 을 돌려준다.

    원본 `dogfight.ai.student_hooks` 로더를 그대로 재사용하므로 계약이 100% 동일하다.
    빈 문자열이면 None 을 반환(= 프레임워크 기본 보상/관측 사용).
    """
    from dogfight.ai.student_hooks import load_reward_hook, load_observation_hook

    reward_fn = reward_config = None
    if reward_module:
        reward_fn, reward_config = load_reward_hook(reward_module)
    observation_hook = None
    if observation_module:
        observation_hook = load_observation_hook(observation_module)
    return reward_fn, reward_config, observation_hook


def make_env(
    overrides: Optional[dict] = None,
    reward_module: str = "",
    observation_module: str = "",
    runner_index: str = "ppo",
    env_index: int = 0,
):
    """표준 설정으로 DogFightWrapper 를 생성한다.

    overrides 로 일부 키만 바꿔서 self-play, 다른 target_mode 등 실험할 수 있다.
    reward_module/observation_module 에 모듈 경로(예: "claude_code.my_reward")를 주면
    해당 보상/관측 함수를 주입한다.
    """
    import copy

    cfg = copy.deepcopy(STANDARD_ENV_CONFIG)
    if overrides:
        _deep_update(cfg, overrides)
    cfg["_runner_index"] = runner_index
    cfg["_env_index"] = env_index

    reward_fn, reward_config, observation_hook = resolve_hooks(reward_module, observation_module)
    if reward_config is not None:
        cfg["reward"] = dict(reward_config)   # 모듈 계수로 교체
        cfg["reward_module"] = reward_module
    if observation_hook is not None:
        cfg["observation_mode"] = observation_hook["mode"]
        cfg["observation_module"] = observation_module

    return DogFightWrapper(
        cfg,
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
    )


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


__all__ = [
    "ROOT",
    "SRC",
    "STANDARD_ENV_CONFIG",
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "ACTION_SIZE",
    "make_env",
    "resolve_hooks",
    "DogFightWrapper",
]
