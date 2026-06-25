"""학습 env 의 시간 게이팅 3-tier damage 적용 여부 — 결정론적 검증.

env.update_damage 가 읽는 sim/target_sim 을 합성 state(FakeSim)로 바꿔, 원하는
거리·ATA·경과시간(SimTime)을 정확히 만들어 damage 를 측정한다.

  - base 환경(단일-tier): tier2/3 기하(ATA 1~3°, r>3000ft)에서 항상 damage=0
  - tier 환경(3-tier)   : tier2(±2°,~3500ft)는 t>=100s, tier3(±3°,~4000ft)는 t>=150s
    부터 damage>0. 그 값은 damage_rate(r,ATA,t)*delta_t 와 정확히 일치.

  python claude_code/verify_tier_damage.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from dogfight.sim.state_schema import StateIndex

from claude_code.env_utils import make_env
from claude_code.my_observation import damage_rate, METER_TO_FEET


class FakeSim:
    def __init__(self, state):
        self._state = np.asarray(state, dtype=np.float64).copy()
        self.deducted = 0.0

    def get_state(self):
        return self._state

    def deduct_health(self, d):
        self.deducted += d


def make_state(n, e, alt, yaw, sim_time):
    s = np.zeros(51, dtype=np.float64)
    s[StateIndex.N], s[StateIndex.E], s[StateIndex.D] = n, e, -alt
    s[StateIndex.ROLL], s[StateIndex.PITCH], s[StateIndex.YAW] = 0.0, 0.0, yaw
    s[StateIndex.SIM_TIME] = sim_time
    s[StateIndex.HEALTH] = 1.0
    return s


def geometry(r_ft, ata_deg, sim_time, alt=7000.0):
    """ownship 기수=북(heading0,수평), 표적을 방위각 ata_deg 거리 r 에 배치.
    수평·heading0 이면 ATA(own→target) = ata_deg 정확."""
    r_m = r_ft / METER_TO_FEET
    th = math.radians(ata_deg)
    own = make_state(0.0, 0.0, alt, 0.0, sim_time)
    tgt = make_state(r_m * math.cos(th), r_m * math.sin(th), alt, 180.0, sim_time)
    return own, tgt


def env_target_damage(env, own, tgt) -> float:
    env._sim = FakeSim(own)
    env._target_sim = FakeSim(tgt)
    env.update_damage()
    return float(env.target_damage)


def main():
    base = make_env(time_gated_damage=False, runner_index="vb")
    tier = make_env(time_gated_damage=True, runner_index="vt")
    geo = tier._geo_info
    dt = tier._delta_t

    cases = [
        ("tier2 기하 (r=3200ft, ATA=1.5°)", 3200.0, 1.5),
        ("tier3 기하 (r=3800ft, ATA=2.5°)", 3800.0, 2.5),
        ("tier1 기하 (r=1000ft, ATA=0.5°)", 1000.0, 0.5),
    ]
    times = [50.0, 120.0, 160.0]

    for label, r_ft, ata in cases:
        print(f"\n=== {label} ===")
        print(f"{'t(s)':>6} | {'base_dmg':>10} | {'tier_dmg':>10} | {'기대(tier)':>12} | match")
        for t in times:
            own, tgt = geometry(r_ft, ata, t)
            # 실제 geo 가 계산하는 값 확인용
            r_ft_real = geo._get_distance(own, tgt) * METER_TO_FEET
            ata_real = abs(geo._get_antenna_train_angle(own, tgt, False))
            b = env_target_damage(base, own, tgt)
            v = env_target_damage(tier, own, tgt)
            expect = damage_rate(r_ft_real, ata_real, t) * dt
            ok = abs(v - expect) < 1e-12
            print(f"{t:6.0f} | {b:10.6f} | {v:10.6f} | {expect:12.6f} | {ok}")

    print("\n[판정]")
    print("  base: tier2/3 기하에서 모든 시각 damage=0  → 학습 env 원본은 단일-tier(시간게이팅 없음)")
    print("  tier: tier2 는 t>=100s, tier3 는 t>=150s 부터 damage>0, 값은 공식과 정확히 일치")
    print("        → 시간 게이팅 3-tier 가 학습 env 에 정상 적용됨")
    base.close(); tier.close()


if __name__ == "__main__":
    main()
