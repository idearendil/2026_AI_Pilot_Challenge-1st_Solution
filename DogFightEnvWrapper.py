from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dogfight.envs.single_agent_env import DogFightEnv


class DogFightWrapper(DogFightEnv):
    def __init__(
        self,
        env_config: dict | None = None,
        AIP_ownship=None,
        AIP_target=None,
        ownship_action_provider=None,
        target_action_provider=None,
        reward_fn=None,
        observation_fn=None,
        observation_size=None,
        observation_low=None,
        observation_high=None,
    ):
        super().__init__(
            env_config=env_config,
            AIP_ownship=AIP_ownship,
            AIP_target=AIP_target,
            ownship_action_provider=ownship_action_provider,
            target_action_provider=target_action_provider,
            reward_fn=reward_fn,
            observation_fn=observation_fn,
            observation_size=observation_size,
            observation_low=observation_low,
            observation_high=observation_high,
        )


__all__ = ["DogFightWrapper"]
