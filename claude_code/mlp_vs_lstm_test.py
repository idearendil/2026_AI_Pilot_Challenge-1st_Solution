"""이전 MLP(은닉 768) 번들과 현재 LSTM 번들을 N판 붙여 어느 쪽이 강한지 통계로 판정한다.

`power_test.py` 의 **rl vs rl** 경로를 그대로 특화한 전용 스크립트다. 두 번들이 정확히
한쪽은 MLP, 한쪽은 LSTM 인지 메타데이터로 자동 검증하고, 어느 아키텍처가 이겼는지
라벨을 붙여 보고한다. 그 외 규약은 power_test 와 동일하다:

  - **양쪽 모두 stochastic**(학습 때와 동일하게 정책 분포에서 샘플링).
  - **판마다 시작 state 가 랜덤**(--seed 를 주면 그 시드에서 판별 시드를 파생해 재현 가능).
  - **시작 위치는 판마다 좌우 교대**(정확히 50/50). 시작 위치 유불리를 상쇄한다.
    한쪽으로 고정하려면 --fixed-side.

역할(role) 주의
--------------
env 는 하나뿐이라 한쪽은 ownship provider(MLPActionProvider→전역 recon), 다른 쪽은
target provider(SelfPlayProvider→자체 recon)로 조종된다. provider 는 **역할로만** 정해지고
MLP/LSTM 둘 다 자동 처리하므로 아키텍처 비교는 공정하다. 좌우 교대(swap)가 시작 위치
유불리를 상쇄하므로, 어느 아키텍처를 ownship 으로 둘지는 결과에 실질적 영향이 없다.
기본은 LSTM=ownship(새 모델을 평가 관점의 주체로) 이고, --lstm-as-target 으로 뒤집을 수 있다.

예시 (LSTM basic2_lstm vs 이전 MLP basic2, 100판):
  python claude_code/mlp_vs_lstm_test.py \
    --lstm-bundle-dir artifacts/models/team01/basic2_lstm \
    --mlp-bundle-dir  artifacts/models/team01/basic2 --games 100
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 콘솔 인코딩(cp949 등)에 없는 문자를 print 해도 죽지 않게(power_test 와 동일 관례).
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np  # noqa: E402

from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402


# ── 승패 판정 (power_test 와 동일 규약) ─────────────────────────────────────────
_HP_EPS = 1e-9
_WIN_ENDS = {"target destroyed", "target altitude below min"}
_LOSS_ENDS = {"ownship destroyed", "ownship altitude below min",
              "two circle headon guard fail"}


def game_outcome(end_condition: str, own_hp: float, tgt_hp: float) -> str:
    if end_condition in _WIN_ENDS:
        return "win"
    if end_condition in _LOSS_ENDS:
        return "loss"
    if own_hp > tgt_hp + _HP_EPS:
        return "win"
    if own_hp < tgt_hp - _HP_EPS:
        return "loss"
    return "draw"


# ── 통계 (power_test 와 동일) ───────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_test_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    total = 2.0 ** n
    obs = math.comb(n, k) / total
    p = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) / total
        if pi <= obs * (1 + 1e-12):
            p += pi
    return min(1.0, p)


# ── 번들 아키텍처 판별 ──────────────────────────────────────────────────────────
def detect_arch(meta: dict) -> tuple[str, tuple]:
    """번들 metadata → ('lstm'|'mlp', hidden dims tuple). model.save_bundle 규약 기준."""
    m = meta.get("model", {}) or {}
    hidden = tuple(m.get("hidden", ()) or ())
    if str(m.get("type")) == "lstm_discrete_actor" or bool(meta.get("lstm")):
        return "lstm", hidden
    return "mlp", hidden


# ── Ray worker (rl vs rl 전용) ─────────────────────────────────────────────────
def _make_worker_cls():
    import ray

    @ray.remote
    class GameWorker:
        """env + 양쪽 rl provider 를 한 번만 만들고 여러 판을 순차로 돌린다."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch

            from claude_code.env_utils import make_env
            from claude_code.action_provider import MLPActionProvider
            from claude_code.evaluate import _ActionRepeatProvider
            from claude_code.model import load_bundle
            from claude_code.normalizers import RunningMeanStd
            from claude_code.self_play import SelfPlayProvider

            torch.set_num_threads(1)
            self.spec = spec
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

            self.env = make_env(overrides=spec["overrides"], reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"ml{spec['wid']}")

            # ownship: MLPActionProvider(항상 stochastic, 전역 recon) + action_repeat(10Hz).
            # MLP/LSTM 무관하게 provider 가 자동 처리(LSTM 이면 내부 hidden stateful).
            inner = MLPActionProvider(bundle_dir=spec["ownship_bundle_dir"], stochastic=True)
            self.own_provider = _ActionRepeatProvider(inner, self.step_ratio)
            self.env._ownship_action_provider = self.own_provider

            # target: SelfPlayProvider(explore=stochastic, 자체 recon). 0-lag(advance→build)로
            # ownship(MLPActionProvider)·학습 learner 와 동일 관측 타이밍.
            m, meta = load_bundle(spec["target_bundle_dir"], device="cpu")
            rms = (RunningMeanStd.from_state_dict(meta["obs_normalization"])
                   if meta.get("obs_normalization") else None)
            self.tgt_provider = SelfPlayProvider(
                m, rms, self.env._observation_fn, self.env._observation_mode,
                self.step_ratio, "cpu", explore=True)
            self.env._target_action_provider = self.tgt_provider

        def play(self, jobs):
            import torch

            out = []
            zero = np.zeros(4, dtype=np.float32)
            for seed, swap, head_swap in jobs:
                torch.manual_seed(int(seed))          # 정책 샘플링 재현(판마다 다른 랜덤)
                self.env._apply_start_side(bool(swap), bool(head_swap))
                obs, _ = self.env.reset(seed=int(seed))
                self.own_provider.reset()
                self.tgt_provider.reset()

                term = trunc = False
                steps = 0
                info = {}
                while not (term or trunc):
                    obs, _r, term, trunc, info = self.env.step(zero)
                    steps += 1

                own_hp = float(info.get("ownship_health", float("nan")))
                tgt_hp = float(info.get("target_health", float("nan")))
                end = str(info.get("end_condition", ""))
                out.append({
                    "seed": int(seed), "swap": int(swap), "head": int(head_swap),
                    "steps": steps, "own_hp": own_hp, "tgt_hp": tgt_hp, "end": end,
                    "outcome": game_outcome(end, own_hp, tgt_hp),
                    "env_outcome": str(info.get("outcome", "")),
                })
            return out

        def close(self):
            try:
                self.env.close()
            except Exception:
                pass

    return GameWorker


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="이전 MLP(768) 번들 vs 현재 LSTM 번들 성능 판정(양쪽 stochastic, 초기 state 랜덤)")
    p.add_argument("--mlp-bundle-dir", required=True, help="이전 MLP(은닉 768) 번들 경로")
    p.add_argument("--lstm-bundle-dir", required=True, help="현재 LSTM 번들 경로")
    p.add_argument("--lstm-as-target", action="store_true",
                   help="역할을 뒤집어 LSTM 을 target(SelfPlayProvider) 쪽에 둔다. "
                        "기본은 LSTM=ownship. 좌우 교대가 위치를 상쇄하므로 결과엔 실질 영향 없음.")
    p.add_argument("--games", type=int, default=100, help="총 대결 판수(기본 100)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="병렬 worker 수(0=물리 코어 수). 각 worker 가 env 1개를 재사용")
    p.add_argument("--seed", type=int, default=-1,
                   help="마스터 시드(생략/-1 이면 매 실행 랜덤). 판별 시드는 여기서 파생")
    p.add_argument("--fixed-side", action="store_true",
                   help="시작 위치 좌우 교대 끄기(기본은 정확히 50/50 교대)")
    p.add_argument("--max-engage-time", type=float,
                   default=float(STANDARD_ENV_CONFIG["max_engage_time"]),
                   help=f"기본 {STANDARD_ENV_CONFIG['max_engage_time']}s (학습 env 와 동일)")
    p.add_argument("--episode-step-limit", type=int,
                   default=int(STANDARD_ENV_CONFIG["episode_step_limit"]))
    p.add_argument("--min-altitude", type=float, default=None, help="생략하면 env 기본값")
    p.add_argument("--csv", default="", help="판별 결과를 CSV 로 저장할 경로(선택)")
    return p.parse_args()


def main():
    args = parse_args()

    from claude_code.model import load_bundle

    mlp_dir, lstm_dir = args.mlp_bundle_dir, args.lstm_bundle_dir
    for lbl, d in (("--mlp-bundle-dir", mlp_dir), ("--lstm-bundle-dir", lstm_dir)):
        if not Path(d).exists():
            raise ValueError(f"{lbl} 번들 경로를 찾을 수 없습니다: {d}")

    _, mlp_meta = load_bundle(mlp_dir, device="cpu")
    _, lstm_meta = load_bundle(lstm_dir, device="cpu")
    mlp_arch, mlp_hidden = detect_arch(mlp_meta)
    lstm_arch, lstm_hidden = detect_arch(lstm_meta)

    # 아키텍처가 라벨과 맞는지 검증(뒤바꿔 지정하는 사고 방지).
    if mlp_arch != "mlp":
        raise ValueError(f"--mlp-bundle-dir 이 MLP 가 아닙니다(감지: {mlp_arch}, hidden={mlp_hidden}): {mlp_dir}")
    if lstm_arch != "lstm":
        raise ValueError(f"--lstm-bundle-dir 이 LSTM 이 아닙니다(감지: {lstm_arch}, hidden={lstm_hidden}): {lstm_dir}")

    # env 는 하나뿐 → 두 번들이 같은 관측 모듈이어야 공정 비교.
    mlp_obs = mlp_meta.get("observation_module", "") or ""
    lstm_obs = lstm_meta.get("observation_module", "") or ""
    if mlp_obs != lstm_obs:
        raise ValueError(
            f"두 번들의 observation_module 이 다릅니다: MLP={mlp_obs!r} / LSTM={lstm_obs!r}. "
            "같은 관측 모듈로 학습한 번들끼리만 한 env 에서 비교할 수 있습니다.")
    obs_module = mlp_obs

    # 역할 배정: 기본 LSTM=ownship, MLP=target. --lstm-as-target 이면 반대.
    if args.lstm_as_target:
        ownship_dir, ownship_arch = mlp_dir, "MLP"
        target_dir, target_arch = lstm_dir, "LSTM"
    else:
        ownship_dir, ownship_arch = lstm_dir, "LSTM"
        target_dir, target_arch = mlp_dir, "MLP"

    overrides = {
        "target_mode": "fixed",            # target 은 provider(SelfPlayProvider)로 조종
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "randomize_start_side": False,     # 아래에서 정확히 50/50 로 직접 교대
    }
    if args.min_altitude is not None:
        overrides["min_altitude"] = args.min_altitude

    games = max(1, int(args.games))
    n_workers = args.num_workers if args.num_workers > 0 else physical_cpu_count()
    n_workers = max(1, min(int(n_workers), games))

    master_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(master_seed)
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=games)]
    swaps = [0] * games if args.fixed_side else [i % 2 for i in range(games)]
    head_swaps = [0] * games if args.fixed_side else [(i // 2) % 2 for i in range(games)]

    own_desc = f"{ownship_arch}({ownship_dir}, hidden={mlp_hidden if ownship_arch=='MLP' else lstm_hidden})"
    tgt_desc = f"{target_arch}({target_dir}, hidden={mlp_hidden if target_arch=='MLP' else lstm_hidden})"
    print(f"[mlp_vs_lstm] ownship = {own_desc}")
    print(f"[mlp_vs_lstm] target  = {tgt_desc}")
    print(f"[mlp_vs_lstm] games={games} workers={n_workers} master_seed={master_seed} "
          f"양쪽 stochastic obs_module={obs_module or '(env default)'}")
    print(f"[mlp_vs_lstm] max_engage={args.max_engage_time}s step_limit={args.episode_step_limit} "
          f"시작위치={'고정' if args.fixed_side else '좌우 교대(50/50)'}")

    import ray

    if not ray.is_initialized():
        pythonpath = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "ownship_bundle_dir": ownship_dir, "target_bundle_dir": target_dir,
    }
    WorkerCls = _make_worker_cls()
    workers = [WorkerCls.remote({**spec, "wid": i}) for i in range(n_workers)]

    jobs = [[] for _ in range(n_workers)]
    for i, (s, w, h) in enumerate(zip(seeds, swaps, head_swaps)):
        jobs[i % n_workers].append((s, w, h))

    import time

    t0 = time.time()
    pending = {w.play.remote(j): i for i, (w, j) in enumerate(zip(workers, jobs)) if j}
    results = []
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        for ref in done:
            results.extend(ray.get(ref))
            pending.pop(ref)
        print(f"  ... {len(results)}/{games} 판 완료 ({time.time() - t0:.0f}s)", flush=True)
    ray.get([w.close.remote() for w in workers])
    elapsed = time.time() - t0

    _report(args, results, own_desc, tgt_desc, ownship_arch, target_arch, master_seed, elapsed)

    if args.csv:
        import csv as _csv

        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            wtr = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wtr.writeheader()
            wtr.writerows(results)
        print(f"\n판별 결과 CSV → {path}")

    ray.shutdown()


def _report(args, results, own_desc, tgt_desc, ownship_arch, target_arch,
            master_seed, elapsed) -> None:
    n = len(results)
    w = sum(1 for r in results if r["outcome"] == "win")     # ownship 승
    l = sum(1 for r in results if r["outcome"] == "loss")    # ownship 패
    d = n - w - l
    dec = w + l

    # 아키텍처 관점으로 환산: ownship 승 = ownship_arch 승.
    print(f"\n{'=' * 72}")
    print(f"MLP vs LSTM 성능 판정  ({n}판, {elapsed:.0f}s, master_seed={master_seed})")
    print(f"  ownship = {own_desc}")
    print(f"  target  = {tgt_desc}")
    print(f"{'=' * 72}")
    print(f"  W/L/D (ownship={ownship_arch} 관점) = {w}/{l}/{d}")
    print(f"  → {ownship_arch} {w}승 / {target_arch} {l}승 / 무 {d}")

    lo, hi = wilson_ci(w, n)
    print(f"  {ownship_arch} 승률(무승부 포함) = {w / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    if dec > 0:
        lo2, hi2 = wilson_ci(w, dec)
        p = binom_test_two_sided(w, dec)
        print(f"  {ownship_arch} 승률(무승부 제외) = {w / dec:.3f}   95% CI [{lo2:.3f}, {hi2:.3f}]"
              f"   (n={dec})")
        print(f"  이항검정 p-value    = {p:.3g}  (H0: 두 아키텍처 실력 동일)")
        if p < 0.05:
            strong, weak = ((ownship_arch, target_arch) if w > l
                            else (target_arch, ownship_arch))
            print(f"  → **{strong} 이 {weak} 보다 유의하게 강하다** (α=0.05)")
        else:
            print("  → 유의한 차이 없음 (α=0.05). 판수를 늘리면 결론이 갈릴 수 있다.")
    else:
        print("  모든 판이 무승부 — 판정 불가")

    if not args.fixed_side:
        for swap, name in ((0, "A측 시작"), (1, "B측 시작")):
            sub = [r for r in results if r["swap"] == swap]
            if not sub:
                continue
            sw = sum(1 for r in sub if r["outcome"] == "win")
            sl = sum(1 for r in sub if r["outcome"] == "loss")
            print(f"    {name}: {ownship_arch} W/L/D {sw}/{sl}/{len(sub) - sw - sl} "
                  f"({sw / max(1, len(sub)):.2f})")

    own_hp = np.mean([r["own_hp"] for r in results])
    tgt_hp = np.mean([r["tgt_hp"] for r in results])
    steps = np.mean([r["steps"] for r in results])
    sim_s = steps * float(STANDARD_ENV_CONFIG["step_ratio"]) / 60.0
    print(f"  평균 최종 HP: {ownship_arch} {own_hp:.3f} / {target_arch} {tgt_hp:.3f}   "
          f"(HP 차 {own_hp - tgt_hp:+.3f})")
    print(f"  평균 길이: {steps:.0f} RL-step (~{sim_s:.0f}s)")
    print("  end_condition:")
    for k, c in Counter(r["end"] for r in results).most_common():
        print(f"    {c:4d}  {k}")


if __name__ == "__main__":
    main()
