from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility fallback for local setups
    import gym as gym

from dogfight.envs.observation import observation_size


class RLLibInferenceEnv(gym.Env):
    """Minimal RLlib env that provides spaces for lightweight inference bundles."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        config = dict(env_config or {})
        mode = config.get("observation_mode", "classic12")
        size = int(config.get("observation_size", observation_size(mode)))

        if mode == "tactical16":
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(size,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(size,),
                dtype=np.float32,
            )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self._zero_observation = np.zeros(size, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self._zero_observation.copy(), {}

    def step(self, action):
        return self._zero_observation.copy(), 0.0, True, False, {}
