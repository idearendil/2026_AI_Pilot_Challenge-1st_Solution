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

import os
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

# 아래 DogFightEnvWrapper import 는 JSBSimWrapper → JSBSimAIPLib.dll 을 로드하고, 그
# 시점에 BT DLL 의 rule XML(AIP_RULE_XML)이 한 번만 읽혀 캐싱된다. 이후 os.environ 을
# 바꿔도 무시되므로, 그때 실제로 걸려 있던 값을 남겨 두고 BT provider 생성 시 검증한다
# (claude_code.bt_rule 참고).
RULE_XML_AT_IMPORT = os.environ.get("AIP_RULE_XML", "")

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
    # 대회 초기 배치: 두 기체가 같은 고도의 남-북 직선 위에 있고, 북(12시) 기체는 동(3시,
    # heading 90)을, 남(6시) 기체는 서(9시, heading 270)를 바라본다(서로 반대 방향이지만
    # 마주보지는 않음, roll/pitch=0). 초기 거리는 매 episode 이 범위[ft]에서 랜덤. 속도는
    # 각자 바라보는 방향으로 config 의 speed. (거리·좌우는 reset 의 _side_rng 로 결정.)
    "start_distance_ft_range": [2000.0, 3000.0],
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

    FEET_TO_METER = 0.3048

    def reset(self, *, seed=None, options=None):
        if getattr(self, "_side_rng", None) is None or seed is not None:
            self._side_rng = np.random.default_rng(seed)
        # 매 episode: 대회 초기 배치(거리 랜덤 + 좌우 배정)로 두 기체 위치를 설정한다.
        dmin, dmax = self.config.get("start_distance_ft_range", [2000.0, 3000.0])
        dist_ft = float(self._side_rng.uniform(float(dmin), float(dmax)))
        if self.config.get("randomize_start_side", True):
            swap = bool(self._side_rng.integers(0, 2))
        else:
            swap = bool(getattr(self, "_forced_swap", False))
        own, tgt = self._competition_positions(dist_ft, swap)
        self.change_init_position("ownship", *own)
        self.change_init_position("target", *tgt)
        return super().reset(seed=seed, options=options)

    def _competition_positions(self, dist_ft: float, swap: bool):
        """대회 초기 배치의 (ownship, target) 위치 [n,e,d,roll,pitch,heading,speed] 를 만든다.

        두 기체는 같은 고도의 남-북 직선 위, 거리 dist_ft(=N 축 간격). 북(12시) 기체는
        동(3시, heading 90°), 남(6시) 기체는 서(9시, heading 270°)를 바라본다. roll/pitch=0.
        고도·속도·중심(N,E)은 config 의 ownship/target 값에서 가져온다. swap=True 면 ownship↔target
        위치 교대(좌우 교대 = 학습 시 시작 위치 과적합 방지, power_test 의 50/50).
        """
        a = list(self.config["ownship"])   # [n, e, d, roll, pitch, heading, speed]
        b = list(self.config["target"])
        center_n = 0.5 * (float(a[0]) + float(b[0]))
        center_e = 0.5 * (float(a[1]) + float(b[1]))
        alt_d = float(a[2])                 # 같은 고도
        speed = float(a[6])                 # 바라보는 방향으로의 속도 크기
        half = 0.5 * float(dist_ft) * self.FEET_TO_METER
        north = [center_n + half, center_e, alt_d, 0.0, 0.0, 90.0, speed]   # 12시 → 동(3시)
        south = [center_n - half, center_e, alt_d, 0.0, 0.0, 270.0, speed]  # 6시 → 서(9시)
        return (south, north) if swap else (north, south)

    def _apply_start_side(self, swap: bool) -> None:
        """다음 reset 의 좌우 배정을 강제 지정한다(power_test 의 정확한 50/50 교대용).

        실제 위치는 reset 에서 대회 배치(거리 랜덤 포함)로 적용된다. randomize_start_side=True
        면 reset 이 좌우를 랜덤으로 정하므로 이 강제값은 무시된다.
        """
        self._forced_swap = bool(swap)

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
    reward_overrides: Optional[dict] = None,
    time_gated_damage: bool = True,
    runner_index: str = "ppo",
    env_index: int = 0,
):
    """표준 설정으로 DogFight 환경을 생성한다.

    overrides 로 일부 키만 바꿔서 self-play, 다른 target_mode 등 실험할 수 있다.
    reward_module/observation_module 에 모듈 경로(예: "claude_code.my_reward")를 주면
    해당 보상/관측 함수를 주입한다.
    reward_overrides 는 (모듈 reward_config 를 적용한 뒤) reward 계수 일부를 덮어쓴다.
    예: phase2 에서 distance 항 끄기 → reward_overrides={"distance_reward_scale": 0.0}.
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
    # 모듈 reward_config 적용 후 일부 계수만 덮어쓴다(phase 전환용).
    if reward_overrides:
        cfg.setdefault("reward", {})
        _deep_update(cfg["reward"], dict(reward_overrides))
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
    "RULE_XML_AT_IMPORT",
    "STANDARD_ENV_CONFIG",
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "ACTION_SIZE",
    "make_env",
    "resolve_hooks",
    "DogFightWrapper",
    "TierGatedDogFightEnv",
]
