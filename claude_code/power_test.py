# -*- coding: utf-8 -*-
"""두 agent 를 N판 붙여 어느 쪽이 강한지 통계로 판정한다(리플레이 없음).

`run_local_dogfight.py` 와 **동일한 env·초기화·제어 주기**로 싸우되, 로그는 안 만들고
Ray 로 여러 판을 동시에 돌린다. 판정은 승/패/무 집계 + 이항검정이라 "우연히 몇 판 이겼다"
와 "실제로 더 강하다" 를 구분한다.

ownship·target 슬롯에 넣을 수 있는 agent(=agent spec, claude_code.agents 참고):
  - ``bundle:<경로>``  CPU/CUDA 학습 번들(metadata.json + policy_weights.pkl.gz)
  - ``ckpt:<경로>``    CUDA 학습 체크포인트(runs/*.pt)
  - ``bt:<이름>``      baselines/ BT DLL(Lee_BT1/Jeon_BT1/Jeon_BT2/Shin_BT_best/Shin_BT_def)
  - ``release_mpc`` / ``stable_mpc`` / ``unreal_exe``   baselines/ 의 MPC·외부 BT exe

각 슬롯마다 **10Hz/60Hz**(--*-hz)와 신경망의 **argmax/stochastic**(--*-action)을 따로 고른다
(baseline 은 자체 고정 제어주기라 hz/action 무시).

예시:
  # CUDA 학습 번들 vs baseline BT
  python claude_code/power_test.py --ownship bundle:artifacts/gpu_ppo_final \
      --target bt:Lee_BT1 --games 100
  # CUDA runs 체크포인트(변환 없이) vs CPU 학습 번들
  python -m claude_code.power_test --ownship "ckpt:runs/mlp_768(only_headon).pt" \
      --target bundle:artifacts/cpu_ppo_final --games 100
  # 내 번들 vs Stable MPC, ownship 은 argmax·60Hz
  python -m claude_code.power_test --ownship bundle:artifacts/gpu_ppo_final \
      --ownship-action argmax --ownship-hz 60 --target stable_mpc --games 100
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

# 콘솔 인코딩(cp949 등)에 없는 문자를 print 해도 UnicodeEncodeError 로 죽지 않게.
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

# ── BT rule XML: 반드시 claude_code 의 다른 import 보다 먼저 ────────────────────
# AIP_RULE_XML 은 JSBSimAIPLib.dll 로드 시점(= claude_code.env_utils import 체인)에 한 번만
# 캐싱된다. 늦게 세팅하면 DLL 이 Rule_forTraining.xml(Task_Empty)로 폴백해 BT 가 조종을
# 전혀 안 한다 → 가짜 승률. argv 를 미리 훑어 여기서 세팅한다.
from claude_code.agents import ENV_KEY, parse_agent_spec, bt_rule_for_specs  # noqa: E402


def _preparse_specs() -> tuple[dict, dict]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ownship", default="")
    pre.add_argument("--target", default="bt:Lee_BT1")
    known, _ = pre.parse_known_args()
    own = parse_agent_spec(known.ownship) if known.ownship else None
    tgt = parse_agent_spec(known.target) if known.target else None
    return own, tgt


def _apply_bt_rule_env() -> None:
    own, tgt = _preparse_specs()
    specs = [s for s in (own, tgt) if s]
    rule = bt_rule_for_specs(*specs) if specs else ""
    if rule:
        os.environ[ENV_KEY] = rule
        print(f"[power_test] BT rule XML = {rule} ({ENV_KEY})")


_apply_bt_rule_env()   # ← 반드시 아래 import 들보다 먼저!

import numpy as np  # noqa: E402

from claude_code import agents  # noqa: E402
from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402


# ── 승패 판정(ownship 관점) ───────────────────────────────────────────────────
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


# ── 통계 ─────────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_test_two_sided(k: int, n: int) -> float:
    """H0: p=0.5 양측 이항검정 exact p-value."""
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


# ── Ray worker ───────────────────────────────────────────────────────────────
def _make_worker_cls():
    import ray

    @ray.remote
    class GameWorker:
        """env + 양측 provider 를 한 번만 만들고 여러 판을 순차로 돌린다(JSBSim init 이 비쌈)."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch

            from claude_code.env_utils import make_env

            torch.set_num_threads(1)
            self.spec = spec
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

            self.env = make_env(overrides=spec["overrides"], reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"pt{spec['wid']}")

            # ownship / target: 슬롯 무관하게 make_side_provider 로 통일(신경망은 각자 독립
            # reconstructor → 공정 mirror, baseline 은 자체 제어주기). unreal_exe 가 양 슬롯에
            # 동시에 올 수 있어 포트 대역을 슬롯별로 분리한다.
            self.own_provider = agents.make_side_provider(
                self.env, spec["own_spec"], self.step_ratio,
                hz=spec["own_hz"], deterministic=spec["own_det"], wid=spec["wid"],
                root=spec["root"], base_port=agents.UNREAL_BASE_PORT + 1000)
            self.env._ownship_action_provider = self.own_provider

            self.tgt_provider = agents.make_side_provider(
                self.env, spec["tgt_spec"], self.step_ratio,
                hz=spec["tgt_hz"], deterministic=spec["tgt_det"], wid=spec["wid"],
                root=spec["root"], base_port=agents.UNREAL_BASE_PORT)
            self.env._target_action_provider = self.tgt_provider

        def play(self, jobs):
            import torch

            out = []
            zero = np.zeros(4, dtype=np.float32)
            for seed, swap, head_swap in jobs:
                torch.manual_seed(int(seed))
                self.env._apply_start_side(bool(swap), bool(head_swap))
                obs, _ = self.env.reset(seed=int(seed))
                if self.own_provider is not None:
                    self.own_provider.reset()
                if self.tgt_provider is not None:
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
            for p in (getattr(self, "tgt_provider", None), getattr(self, "own_provider", None)):
                try:
                    if p is not None:
                        p.close()
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass

    return GameWorker


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="두 agent 를 N판 붙여 어느 쪽이 강한지 통계로 판정(리플레이 없음)")
    p.add_argument("--ownship", required=True,
                   help="ownship agent spec. bundle:<경로> / ckpt:<경로> / bt:<이름> / "
                        "release_mpc / stable_mpc / unreal_exe")
    p.add_argument("--target", default="bt:Lee_BT1",
                   help="target agent spec(형식은 --ownship 과 동일, 기본 bt:Lee_BT1)")
    for side in ("ownship", "target"):
        p.add_argument(f"--{side}-hz", type=int, choices=[10, 60], default=10,
                       help=f"{side} 신경망 제어 주기(baseline 은 무시, 기본 10)")
        p.add_argument(f"--{side}-action", choices=["stochastic", "argmax"],
                       default="stochastic",
                       help=f"{side} 신경망 action 선택(baseline 은 무시, 기본 stochastic)")
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


def _desc(spec: dict, hz: int, action: str) -> str:
    if spec["kind"] in agents.NN_KINDS:
        return f"{spec['kind']}:{spec['name']}({hz}Hz, {action})"
    return spec["name"]


def main():
    args = parse_args()

    own_spec = parse_agent_spec(args.ownship)
    tgt_spec = parse_agent_spec(args.target)

    # rule XML 은 import 전에 세팅됐다. 다시 계산해 어긋나면 실패시킨다(가짜 승률 방지).
    bt_rule = bt_rule_for_specs(own_spec, tgt_spec) or ""
    if bt_rule != os.environ.get(ENV_KEY, ""):
        raise RuntimeError(
            f"BT rule XML 불일치: 최종={bt_rule!r} / import 시점={os.environ.get(ENV_KEY)!r}.")

    # 존재 검증(신경망 경로 / baseline 자산).
    for label, spec in (("ownship", own_spec), ("target", tgt_spec)):
        if spec["kind"] in ("bundle", "ckpt") and not Path(spec["path"]).exists():
            raise FileNotFoundError(f"--{label} 경로를 찾을 수 없습니다: {spec['path']}")
        if spec["kind"] == "bt" and not (ROOT / spec["dll"]).is_file():
            raise FileNotFoundError(f"--{label} BT DLL 이 없습니다: {spec['dll']}")

    any_nn = agents.is_nn(own_spec) or agents.is_nn(tgt_spec)
    obs_module = agents.OBS_MODULE if any_nn else ""

    overrides = {
        "target_mode": "fixed",            # 양 슬롯 모두 provider 가 조종
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

    own_desc = _desc(own_spec, args.ownship_hz, args.ownship_action)
    tgt_desc = _desc(tgt_spec, args.target_hz, args.target_action)
    print(f"[power_test] ownship = {own_desc}")
    print(f"[power_test] target  = {tgt_desc}")
    print(f"[power_test] games={games} workers={n_workers} master_seed={master_seed} "
          f"obs_module={obs_module or '(env default)'}")
    print(f"[power_test] max_engage={args.max_engage_time}s step_limit={args.episode_step_limit} "
          f"시작위치={'고정' if args.fixed_side else '좌우 교대(50/50)'}")

    import ray

    if not ray.is_initialized():
        pythonpath = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
        env_vars = {"PYTHONPATH": pythonpath}
        if bt_rule:
            env_vars[ENV_KEY] = bt_rule   # worker 프로세스 시작 시점에 있어야 DLL 이 읽는다
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": env_vars})

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "own_spec": own_spec, "tgt_spec": tgt_spec,
        "own_hz": int(args.ownship_hz), "tgt_hz": int(args.target_hz),
        "own_det": (args.ownship_action == "argmax"),
        "tgt_det": (args.target_action == "argmax"),
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

    _report(args, results, own_desc, tgt_desc, master_seed, elapsed)

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


def _report(args, results, own_desc, tgt_desc, master_seed, elapsed) -> None:
    n = len(results)
    w = sum(1 for r in results if r["outcome"] == "win")
    l = sum(1 for r in results if r["outcome"] == "loss")
    d = n - w - l
    dec = w + l

    print(f"\n{'=' * 72}")
    print(f"power test 결과  ({n}판, {elapsed:.0f}s, master_seed={master_seed})")
    print(f"  ownship = {own_desc}")
    print(f"  target  = {tgt_desc}")
    print(f"{'=' * 72}")
    print(f"  W/L/D (ownship 관점) = {w}/{l}/{d}")

    lo, hi = wilson_ci(w, n)
    print(f"  승률(무승부 포함)   = {w / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    if dec > 0:
        lo2, hi2 = wilson_ci(w, dec)
        p = binom_test_two_sided(w, dec)
        print(f"  승률(무승부 제외)   = {w / dec:.3f}   95% CI [{lo2:.3f}, {hi2:.3f}]"
              f"   (n={dec})")
        print(f"  이항검정 p-value    = {p:.3g}  (H0: 두 agent 실력 동일)")
        if p < 0.05:
            strong, weak = ("ownship", "target") if w > l else ("target", "ownship")
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
            print(f"    {name}: W/L/D {sw}/{sl}/{len(sub) - sw - sl} "
                  f"({sw / max(1, len(sub)):.2f})")

    own_hp = np.mean([r["own_hp"] for r in results])
    tgt_hp = np.mean([r["tgt_hp"] for r in results])
    steps = np.mean([r["steps"] for r in results])
    sim_s = steps * float(STANDARD_ENV_CONFIG["step_ratio"]) / 60.0
    print(f"  평균 최종 HP: ownship {own_hp:.3f} / target {tgt_hp:.3f}   "
          f"(HP 차 {own_hp - tgt_hp:+.3f})")
    print(f"  평균 길이: {steps:.0f} RL-step (~{sim_s:.0f}s)")
    print("  end_condition:")
    for k, c in Counter(r["end"] for r in results).most_common():
        print(f"    {c:4d}  {k}")


if __name__ == "__main__":
    main()
