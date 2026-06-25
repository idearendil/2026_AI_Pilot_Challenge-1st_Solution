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

import numpy as np

# Release 루트와 src 를 import 경로에 추가 (원본 my_submission.py 와 동일한 방식).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from DogFightEnvWrapper import DogFightWrapper  # noqa: E402
from dogfight.sim.state_schema import StateIndex  # noqa: E402


# experiments/student_ppo_mlp.yaml 기반 기본값.
# max_engage_time=200s: 대결 서버 damage 시간 게이팅(tier2 100s, tier3 150s)을 학습에서
# 겪도록 60→200 으로 상향. (episode_step_limit=3600 RL-step=360s 라 200s 가 먼저 truncate.)
STANDARD_ENV_CONFIG = {
    "observation_mode": "tactical16",
    "target_mode": "fixed",
    "target_behavior_dll": "AIP_BASE_target.dll",
    "ownship_control_mode": "rl",
    "max_engage_time": 200.0,
    "episode_step_limit": 3600,
    "step_ratio": 6,
    # 매 episode 학습 agent 시작 위치를 두 위치(ownship/target 설정) 중 랜덤 선택(좌우 교대).
    "randomize_start_side": True,
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


class TierGatedDogFightEnv(DogFightWrapper):
    """학습 env 에 대결 서버의 '시간 게이팅 3-tier damage' 를 적용한 버전.

    원본 DogFightEnv.update_damage 는 정적 단일-tier(±1°, 500~3000ft)만 적용한다.
    이 서브클래스는 update_damage 를 오버라이드해 episode 시간(SimTime)에 따라
      tier1: 항상, tier2: 100s 부터, tier3: 150s 부터
    를 적용한다. damage 공식은 claude_code.my_observation.damage_rate 와 동일하므로
    env 의 실제 HP 와 관측의 재구성 HP 가 같은 모델을 따른다.

    또한 randomize_start_side=True(기본)면 매 episode 학습 agent(ownship)의 시작 위치를
    두 위치(config 의 ownship / target 설정) 중 랜덤 선택한다(좌우 대칭 교대). self-play
    에서 한쪽 시작 위치에 과적합하지 않게 한다.
    """

    def reset(self, *, seed=None, options=None):
        if getattr(self, "_side_rng", None) is None or seed is not None:
            self._side_rng = np.random.default_rng(seed)
        if self.config.get("randomize_start_side", True):
            self._apply_start_side(bool(self._side_rng.integers(0, 2)))
        return super().reset(seed=seed, options=options)

    def _apply_start_side(self, swap: bool) -> None:
        """swap=False → ownship→A(config ownship)/target→B(config target), True → 교대."""
        a = list(self.config["ownship"])   # [n, e, d, roll, pitch, heading, speed]
        b = list(self.config["target"])
        own, tgt = (b, a) if swap else (a, b)
        self.change_init_position("ownship", own[0], own[1], own[2], own[3], own[4], own[5], own[6])
        self.change_init_position("target", tgt[0], tgt[1], tgt[2], tgt[3], tgt[4], tgt[5], tgt[6])

    def update_damage(self):
        from claude_code.my_observation import damage_rate, METER_TO_FEET

        own = self._sim.get_state()
        tgt = self._target_sim.get_state()
        r_ft = self._geo_info._get_distance(own, tgt) * METER_TO_FEET
        own_ata = self._geo_info._get_antenna_train_angle(own, tgt, False)   # 내 기수→표적
        tgt_ata = self._geo_info._get_antenna_train_angle(tgt, own, False)   # 표적 기수→나
        t_sec = float(own[StateIndex.SIM_TIME])

        # damage_rate 는 초당 rate → env 와 동일하게 ×delta_t(=1/sim_hz) per sub-step 적분.
        target_damage = damage_rate(r_ft, own_ata, t_sec) * self._delta_t
        ownship_damage = damage_rate(r_ft, tgt_ata, t_sec) * self._delta_t

        self.ownship_damage = ownship_damage
        self.target_damage = target_damage
        self._in_wez = target_damage > 0.0
        self._sim.deduct_health(ownship_damage)
        self._target_sim.deduct_health(target_damage)


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
    time_gated_damage: bool = True,
    runner_index: str = "ppo",
    env_index: int = 0,
):
    """표준 설정으로 DogFight 환경을 생성한다.

    overrides 로 일부 키만 바꿔서 self-play, 다른 target_mode 등 실험할 수 있다.
    reward_module/observation_module 에 모듈 경로(예: "claude_code.my_reward")를 주면
    해당 보상/관측 함수를 주입한다.
    time_gated_damage=True(기본)면 대결 서버의 시간 게이팅 3-tier damage 를 적용한
    TierGatedDogFightEnv 를 사용한다. False 면 원본 단일-tier DogFightWrapper.
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

    env_cls = TierGatedDogFightEnv if time_gated_damage else DogFightWrapper
    return env_cls(
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
    "TierGatedDogFightEnv",
]
