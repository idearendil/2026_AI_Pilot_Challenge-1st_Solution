# -*- coding: utf-8 -*-
"""RL 번들 vs `unreal_bt_client.exe`(외부 BT UDP 클라이언트) 파워 테스트.

`power_test.py` 와 **동일한 env·제어 주기·판정·통계** 로 N 판을 붙이되, 상대(target)를
in-process BT DLL 대신 실행 중인 `unreal_bt_client.exe` 로 조종한다. exe 는 UDP 클라이언트라
서버가 필요하므로, worker 마다 로컬 UDP 서버 브리지(claude_code.unreal_exe_provider.
UnrealExeProvider)를 띄우고 exe 프로세스를 붙인다(worker 마다 다른 port → 병렬).

물리/데미지/WEZ/종료/승패 판정·통계는 전부 power_test 의 코드를 그대로 재사용한다.

예시 (RL 번들 vs unreal_bt_client.exe, 100판):
  python claude_code/power_test_unreal.py \
    --ownship-bundle-dir artifacts/models/team01/basic --games 100

  # exe 경로/포트/워커 수 지정
  python claude_code/power_test_unreal.py \
    --ownship-bundle-dir artifacts/models/team01/basic \
    --exe-path unreal_bt_client.exe --base-port 9000 --num-workers 4 --games 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# power_test 의 순수 통계/판정/리포트 함수를 재사용한다. 단, power_test 는 import 시점에
# _apply_bt_rule_env() 를 돌려 (기본값 기준) BT rule 환경변수를 세팅하는 side effect 가 있다.
# 우리는 in-process BT 를 전혀 안 쓰므로 그 side effect 가 안 생기도록 argv 를 잠깐 바꿔치기해
# import 한 뒤 되돌린다(target-backend rl → BT DLL 없음 → rule 미설정).
_saved_argv = sys.argv
sys.argv = [_saved_argv[0], "--ownship-backend", "rl", "--target-backend", "rl"]
try:
    from claude_code.power_test import (  # noqa: E402
        binom_test_two_sided,
        game_outcome,
        wilson_ci,
    )
finally:
    sys.argv = _saved_argv

import numpy as np  # noqa: E402

from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402


def _make_worker_cls():
    import ray

    @ray.remote
    class GameWorker:
        """env(+RL ownship provider)+exe 브리지를 한 번 만들고 여러 판을 순차로 돌린다."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch

            from claude_code.env_utils import make_env

            torch.set_num_threads(1)
            self.spec = spec
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

            self.env = make_env(overrides=spec["overrides"],
                                reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"ptx{spec['wid']}")

            # ownship: RL 번들(항상 stochastic) + action_repeat=step_ratio (power_test 와 동일)
            from claude_code.action_provider import MLPActionProvider
            from claude_code.evaluate import _ActionRepeatProvider
            inner = MLPActionProvider(bundle_dir=spec["ownship_bundle_dir"],
                                      stochastic=bool(spec["stochastic"]))
            self.own_provider = _ActionRepeatProvider(inner, self.step_ratio)
            self.env._ownship_action_provider = self.own_provider

            # target: unreal_bt_client.exe 브리지(worker 마다 다른 port 로 exe 실행)
            from claude_code.unreal_exe_provider import UnrealExeProvider
            self.tgt_provider = UnrealExeProvider(
                exe_path=spec["exe_path"],
                port=int(spec["base_port"]) + int(spec["wid"]),
                own_plane_id=1, enemy_plane_id=0,
                ownship_force_side=int(spec["ownship_force_side"]),
                target_force_side=int(spec["target_force_side"]),
                cwd=spec["root"],
                step_timeout_sec=float(spec["step_timeout_sec"]),
                quiet=True,
            )
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
                    "steps": steps,
                    "own_hp": own_hp, "tgt_hp": tgt_hp, "end": end,
                    "outcome": game_outcome(end, own_hp, tgt_hp),
                    "env_outcome": str(info.get("outcome", "")),
                    "exe_fail": int(getattr(self.tgt_provider, "_fail_count", 0)),
                })
            return out

        def close(self):
            try:
                self.tgt_provider.close()
            except Exception:
                pass
            try:
                self.env.close()
            except Exception:
                pass

    return GameWorker


def parse_args():
    p = argparse.ArgumentParser(
        description="RL 번들 vs unreal_bt_client.exe 파워 테스트(리플레이 없음)")
    p.add_argument("--ownship-bundle-dir", required=True, help="RL(ownship) claude 번들 경로")
    p.add_argument("--deterministic", action="store_true",
                   help="RL action 을 정책 분포 샘플링 대신 argmax(deterministic)로 결정. "
                        "기본은 학습과 동일한 stochastic 샘플링")
    p.add_argument("--exe-path", default=str(ROOT / "unreal_bt_client.exe"),
                   help="unreal_bt_client.exe 경로(기본: 프로젝트 루트)")
    p.add_argument("--base-port", type=int, default=9000,
                   help="worker0 이 쓸 UDP 포트. worker i 는 base_port+i 사용(기본 9000)")
    p.add_argument("--ownship-force-side", type=int, default=1,
                   help="exe 관점에서 자기(상대기)의 force side (기본 1)")
    p.add_argument("--target-force-side", type=int, default=2,
                   help="exe 관점에서 적(본기체)의 force side (기본 2)")
    p.add_argument("--step-timeout-sec", type=float, default=0.5,
                   help="한 substep 에서 exe 의 CMD 를 기다리는 최대 시간(기본 0.5s)")
    p.add_argument("--games", type=int, default=100, help="총 대결 판수(기본 100)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="병렬 worker 수(0=물리 코어 수). worker 마다 exe 1개를 띄운다")
    p.add_argument("--seed", type=int, default=-1,
                   help="마스터 시드(생략/-1 이면 매 실행 랜덤). 판별 시드는 여기서 파생")
    p.add_argument("--fixed-side", action="store_true",
                   help="시작 위치 좌우 교대 끄기(기본은 정확히 50/50 교대)")
    p.add_argument("--max-engage-time", type=float,
                   default=float(STANDARD_ENV_CONFIG["max_engage_time"]))
    p.add_argument("--episode-step-limit", type=int,
                   default=int(STANDARD_ENV_CONFIG["episode_step_limit"]))
    p.add_argument("--min-altitude", type=float, default=None,
                   help="생략하면 env 기본값")
    p.add_argument("--csv", default="", help="판별 결과를 CSV 로 저장할 경로(선택)")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.exe_path).exists():
        raise FileNotFoundError(f"exe 를 찾을 수 없습니다: {args.exe_path}")

    from claude_code.model import load_bundle

    _, meta = load_bundle(args.ownship_bundle_dir, device="cpu")
    obs_module = meta.get("observation_module", "") or ""

    # target 은 브리지 provider 가 조종하므로 env 자체 AI 를 만들지 않도록 fixed.
    overrides = {
        "target_mode": "fixed",
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

    rl_mode = "deterministic(argmax)" if args.deterministic else "stochastic"
    own_desc = f"rl({args.ownship_bundle_dir})"
    tgt_desc = f"unreal_bt_client.exe({Path(args.exe_path).name})"
    print(f"[power_test_unreal] ownship = {own_desc}")
    print(f"[power_test_unreal] target  = {tgt_desc}")
    print(f"[power_test_unreal] games={games} workers={n_workers} master_seed={master_seed} "
          f"rl={rl_mode} obs_module={obs_module or '(env default)'}")
    print(f"[power_test_unreal] ports={args.base_port}..{args.base_port + n_workers - 1} "
          f"max_engage={args.max_engage_time}s step_limit={args.episode_step_limit} "
          f"시작위치={'고정' if args.fixed_side else '좌우 교대(50/50)'}")

    import ray

    if not ray.is_initialized():
        pythonpath = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "ownship_bundle_dir": args.ownship_bundle_dir,
        "exe_path": str(args.exe_path), "base_port": int(args.base_port),
        "ownship_force_side": int(args.ownship_force_side),
        "target_force_side": int(args.target_force_side),
        "step_timeout_sec": float(args.step_timeout_sec),
        "stochastic": (not args.deterministic),
    }
    WorkerCls = _make_worker_cls()
    workers = [WorkerCls.remote({**spec, "wid": i}) for i in range(n_workers)]

    jobs = [[] for _ in range(n_workers)]
    for i, (s, w, h) in enumerate(zip(seeds, swaps, head_swaps)):
        jobs[i % n_workers].append((s, w, h))

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
    print(f"power test (unreal exe) 결과  ({n}판, {elapsed:.0f}s, master_seed={master_seed})")
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
        print(f"  이항검정 p-value    = {p:.3g}  (H0: 두 모델 실력 동일)")
        if p < 0.05:
            strong, weak = ("ownship(RL)", "target(BT exe)") if w > l else ("target(BT exe)", "ownship(RL)")
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
    exe_fail = sum(r.get("exe_fail", 0) for r in results)
    print(f"  평균 최종 HP: ownship {own_hp:.3f} / target {tgt_hp:.3f}   "
          f"(HP 차 {own_hp - tgt_hp:+.3f})")
    print(f"  평균 길이: {steps:.0f} RL-step (~{sim_s:.0f}s)")
    if exe_fail:
        print(f"  ⚠ exe CMD 타임아웃(안전행동 대체) 총 {exe_fail}회 — step-timeout 을 늘려보세요")
    print("  end_condition:")
    for k, c in Counter(r["end"] for r in results).most_common():
        print(f"    {c:4d}  {k}")


if __name__ == "__main__":
    main()
