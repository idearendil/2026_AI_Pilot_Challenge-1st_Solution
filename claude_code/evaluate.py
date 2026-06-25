"""학습한 claude_code PPO 번들을 로컬 DogFight 환경에서 검증한다.

`MLPActionProvider` 를 ownship 조종자로 환경에 주입해 몇 개 episode 를 굴리고
결과(outcome, return, 길이)를 출력한다. 제출 전 정책이 정상 동작하는지 확인하는
용도다. (원본 run_local_dogfight.py 는 RLlib 번들만 backend 로 직접 지원하므로
claude_code 번들은 이 스크립트로 검증한다.)

예시:
  python claude_code/evaluate.py --bundle-dir artifacts/models/team01/ppo_mlp_v1 --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from dogfight.ai.action_provider import ActionProvider

from claude_code.action_provider import MLPActionProvider
from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG


class _ActionRepeatProvider(ActionProvider):
    """내부 provider 를 step_ratio 마다 한 번만 호출하고 그 사이엔 캐시 action 유지.

    환경에 ownship_action_provider 로 주입하면 provider 는 매 sim sub-step(=6회/RL step)
    호출된다. 학습은 RL action 을 step_ratio 동안 유지하므로(그리고 대결 서버 경로도
    ProviderCommandPolicy.action_repeat=6 으로 동일), 로컬 검증도 같은 제어 주기를
    맞춰야 학습과 동일하게 동작한다.
    """

    def __init__(self, inner: ActionProvider, repeat: int):
        self.inner = inner
        self.repeat = max(1, int(repeat))
        self._count = 0
        self._cached = None

    def reset(self, context=None) -> None:
        self.inner.reset(context)
        self._count = 0
        self._cached = None

    def compute_action(self, context):
        if self._cached is None or self._count % self.repeat == 0:
            self._cached = self.inner.compute_action(context)
        self._count += 1
        return self._cached

    def close(self) -> None:
        self.inner.close()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a claude_code PPO bundle locally")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--target-mode", default="loiter",
                   choices=["fixed", "behavior_tree", "loiter", "autopilot"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))
    # 학습/제출과 동일한 action_repeat(=step_ratio) 로 정책을 유지.
    inner = MLPActionProvider(bundle_dir=args.bundle_dir)
    provider = _ActionRepeatProvider(inner, step_ratio)
    # 학습 때 쓴 관측 모듈을 그대로 환경에 주입(번들 메타에 기록됨).
    obs_module = inner.metadata.get("observation_module", "") or ""
    env = make_env(
        overrides={"target_mode": args.target_mode},
        observation_module=obs_module,
        runner_index="eval",
    )
    # ownship 을 학습 정책으로 조종.
    env._ownship_action_provider = provider

    outcomes: dict[str, int] = {}
    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        provider.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            # action 인자는 provider 사용 시 무시되지만 step 시그니처상 필요.
            obs, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
            ep_ret += r
            steps += 1
            done = term or trunc
        outcome = info.get("outcome", "?")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        returns.append(ep_ret)
        print(f"episode {ep + 1:2d} | outcome={outcome:8s} | return={ep_ret:9.3f} | "
              f"steps={steps:4d} | end={info.get('end_condition', '')}")

    env.close()
    provider.close()
    print("\n=== 요약 ===")
    print(f"평균 return: {np.mean(returns):.3f}  (min {np.min(returns):.3f}, max {np.max(returns):.3f})")
    print(f"outcome 분포: {outcomes}")


if __name__ == "__main__":
    main()
