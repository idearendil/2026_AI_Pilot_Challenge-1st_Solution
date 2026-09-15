# -*- coding: utf-8 -*-
"""최종 파워 테스트: ownship 모델 하나를 **모든 baseline 상대**로 한 번에 붙인다.

`power_test.py` 와 **동일한 env·초기화·제어 주기·판정·통계**를 그대로 쓰되, 이 스크립트
한 번으로 지정한 ownship(학습 번들 또는 CUDA 체크포인트)이 baselines/ 의 모든 상대
(BT DLL 5종 + Release/Stable team-share MPC + unreal_bt_client.exe)와 차례로 싸우고 종합
리포트를 출력한다.

옵션은 딱 3개:
  --ownship <spec>   bundle:<경로> 또는 ckpt:<경로> (CPU/CUDA 학습 모델)
  --hz {10,60}       ownship 신경망 제어 주기(모든 target 에 동일 적용, 기본 10)
  --action {stochastic,argmax}   ownship 신경망 action 선택(기본 stochastic)
  --games N          각 target 당 대결 판수(기본 100)

target 의 제어 주기는 각 baseline 고유값(BT/MPC/exe 모두 60Hz).

BT rule XML(AIP_RULE_XML)은 프로세스 전역이라 target 마다 rule 이 다르면 한 프로세스에
하나만 로드된다. 그래서 **target 마다 독립 Ray 세션**을 새로 띄우고(그 target 의 rule 을
worker 시작 시점에 주입), 끝나면 세션을 내린다(rule 격리 + 장애 격리).

내결함성: worker 를 하나씩 게임 배정(work-queue)하고 크래시/타임아웃 난 worker 는 즉시
제외한 뒤 남은 worker 로 큐를 끝까지 돌린다. 살아남은 게임 결과만 종합한다.

예시:
  python -m claude_code.final_power_test --ownship bundle:artifacts/gpu_ppo_final --games 100
  python -m claude_code.final_power_test --ownship "ckpt:runs/mlp_768(only_headon).pt" \
      --hz 10 --action argmax --games 50
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 콘솔 인코딩(cp949 등)에 없는 문자를 print 해도 죽지 않게(다른 스크립트와 동일 관례).
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np  # noqa: E402

from claude_code import agents  # noqa: E402
from claude_code.agents import ENV_KEY, parse_agent_spec  # noqa: E402
from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402

CUTOFF_BASE_PORT = 9600


# ── 승패 판정(power_test 와 동일. import side effect 피하려 인라인) ─────────────
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


# ── Ray worker ────────────────────────────────────────────────────────────────
def _make_worker_cls():
    import ray

    @ray.remote
    class GameWorker:
        """env + ownship provider + (이 target 전용) baseline provider 를 한 번 만들고,
        게임을 한 판씩(play_one) 배정받아 돌린다(내결함성: 크래시 시 이 worker 만 제외)."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch

            from claude_code.env_utils import make_env

            torch.set_num_threads(1)
            self.spec = spec
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

            self.env = make_env(overrides=spec["overrides"], reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"fpt{spec['wid']}")

            # ownship: 학습 모델(bundle/ckpt). hz·action 은 모든 target 공통.
            self.own_provider = agents.make_side_provider(
                self.env, spec["own_spec"], self.step_ratio,
                hz=spec["own_hz"], deterministic=spec["own_det"], wid=spec["wid"],
                root=spec["root"])
            self.env._ownship_action_provider = self.own_provider

            # target: 이 worker 세션 전용 baseline.
            self.tgt_provider = agents.make_baseline_provider(
                self.env, spec["target"], self.step_ratio, wid=spec["wid"],
                root=spec["root"], base_port=CUTOFF_BASE_PORT)
            self.env._target_action_provider = self.tgt_provider

        def play_one(self, job):
            import torch

            seed, swap, head_swap = job
            torch.manual_seed(int(seed))
            self.env._apply_start_side(bool(swap), bool(head_swap))
            obs, _ = self.env.reset(seed=int(seed))
            if self.own_provider is not None:
                self.own_provider.reset()
            if self.tgt_provider is not None:
                self.tgt_provider.reset()

            zero = np.zeros(4, dtype=np.float32)
            term = trunc = False
            steps = 0
            info = {}
            while not (term or trunc):
                obs, _r, term, trunc, info = self.env.step(zero)
                steps += 1

            own_hp = float(info.get("ownship_health", float("nan")))
            tgt_hp = float(info.get("target_health", float("nan")))
            end = str(info.get("end_condition", ""))
            return {
                "seed": int(seed), "swap": int(swap), "head": int(head_swap), "steps": steps,
                "own_hp": own_hp, "tgt_hp": tgt_hp, "end": end,
                "outcome": game_outcome(end, own_hp, tgt_hp),
                "exe_fail": int(getattr(self.tgt_provider, "_fail_count", 0)),
            }

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


# ── 한 target 을 독립 Ray 세션에서 내결함성 있게 실행 ─────────────────────────
def run_target(target: dict, own_spec: dict, own_hz: int, own_det: bool, games: int,
               n_workers: int, master_seed: int, max_engage: float, step_limit: int,
               game_timeout: float, obs_module: str) -> dict:
    import ray

    rng = np.random.default_rng(master_seed)
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=games)]
    swaps = [i % 2 for i in range(games)]
    head_swaps = [(i // 2) % 2 for i in range(games)]
    jobs = list(zip(seeds, swaps, head_swaps))

    overrides = {
        "target_mode": "fixed",           # target 은 provider 가 조종
        "max_engage_time": max_engage,
        "episode_step_limit": step_limit,
        "randomize_start_side": False,
    }

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "own_spec": own_spec, "own_hz": int(own_hz), "own_det": bool(own_det),
        "target": target,
    }

    # ── target 전용 Ray 세션(BT rule 은 worker 시작 시점에 주입) ──
    pythonpath = os.pathsep.join([str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
    env_vars = {"PYTHONPATH": pythonpath}
    if target["kind"] == "bt" and target.get("rule"):
        env_vars[ENV_KEY] = target["rule"]   # DLL 로드 전(=worker 프로세스 시작)에 있어야 함
    ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
             log_to_driver=False, runtime_env={"env_vars": env_vars})

    WorkerCls = _make_worker_cls()
    workers = [WorkerCls.remote({**spec, "wid": i}) for i in range(n_workers)]

    alive = set(range(n_workers))
    free = list(range(n_workers))
    queue = deque(jobs)
    pending: dict = {}            # future -> (widx, start_time)
    results: list = []
    dead = 0
    t0 = time.time()

    def _dispatch():
        while free and queue:
            widx = free.pop()
            if widx not in alive:
                continue
            fut = workers[widx].play_one.remote(queue.popleft())
            pending[fut] = (widx, time.time())

    _dispatch()
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1, timeout=5.0)
        now = time.time()
        for fut in done:
            widx, _st = pending.pop(fut)
            try:
                results.append(ray.get(fut))
                free.append(widx)                     # 정상 → 재사용
            except Exception as e:
                alive.discard(widx)
                dead += 1
                print(f"    [worker {widx}] 크래시 — 제외: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:120]}", flush=True)
        # 행(hang) 감지: game_timeout 초과 게임은 취소하고 그 worker 제외
        for fut, (widx, st) in list(pending.items()):
            if now - st > game_timeout:
                try:
                    ray.cancel(fut, force=True)
                except Exception:
                    pass
                pending.pop(fut, None)
                alive.discard(widx)
                dead += 1
                print(f"    [worker {widx}] 게임 타임아웃({game_timeout:.0f}s) — 제외", flush=True)
        _dispatch()
        if not pending and queue and not alive:
            print(f"    모든 worker 사망 — {len(queue)} 판 미실행", flush=True)
            break
        if done:
            print(f"    ... {target['name']}: {len(results)}/{games} 완료 "
                  f"({now - t0:.0f}s, 제외 worker {dead})", flush=True)

    for widx in list(alive):
        try:
            ray.get(workers[widx].close.remote(), timeout=15)
        except Exception:
            pass
    ray.shutdown()

    return {"target": target, "results": results,
            "dead": dead, "played": len(results), "requested": games,
            "elapsed": time.time() - t0}


def _summ(results: list) -> dict:
    n = len(results)
    w = sum(1 for r in results if r["outcome"] == "win")
    l = sum(1 for r in results if r["outcome"] == "loss")
    d = n - w - l
    return {"n": n, "w": w, "l": l, "d": d}


def _report(all_runs: list, own_desc: str, requested: int, master_seed: int,
            total_elapsed: float) -> None:
    print(f"\n{'=' * 84}")
    print(f"최종 파워 테스트 결과   ownship = {own_desc}   "
          f"(target 당 {requested}판, master_seed={master_seed}, {total_elapsed:.0f}s)")
    print(f"{'=' * 84}")
    hdr = (f"{'target':<22}{'W/L/D':>12}{'승률(D포함)':>13}"
           f"{'승률(D제외)':>13}{'HP차':>8}{'판수':>7}{'제외':>5}")
    print(hdr)
    print("-" * 84)

    tot_w = tot_l = tot_d = tot_played = tot_dead = 0
    for run in all_runs:
        name = run["target"]["name"]
        res = run["results"]
        s = _summ(res)
        tot_w += s["w"]; tot_l += s["l"]; tot_d += s["d"]
        tot_played += run["played"]; tot_dead += run["dead"]
        if s["n"] == 0:
            print(f"{name:<22}{'-':>12}{'(결과 없음)':>26}"
                  f"{'-':>8}{run['played']:>7}{run['dead']:>5}")
            continue
        wl_d = s["w"] / s["n"]
        dec = s["w"] + s["l"]
        wl_nd = (s["w"] / dec) if dec else float("nan")
        hp = np.mean([r["own_hp"] - r["tgt_hp"] for r in res])
        wld = f"{s['w']}/{s['l']}/{s['d']}"
        print(f"{name:<22}{wld:>12}{wl_d:>13.3f}"
              f"{wl_nd:>13.3f}{hp:>+8.3f}{s['n']:>7}{run['dead']:>5}")

    print("-" * 84)
    tot_n = tot_w + tot_l + tot_d
    if tot_n:
        dec = tot_w + tot_l
        wld = f"{tot_w}/{tot_l}/{tot_d}"
        print(f"{'TOTAL':<22}{wld:>12}{tot_w / tot_n:>13.3f}"
              f"{(tot_w / dec) if dec else float('nan'):>13.3f}"
              f"{'':>8}{tot_n:>7}{tot_dead:>5}")
    print(f"{'=' * 84}")

    # target 별 end_condition / 유의성
    for run in all_runs:
        res = run["results"]
        if not res:
            print(f"\n[{run['target']['name']}] 결과 없음 "
                  f"(요청 {run['requested']}판, 제외 worker {run['dead']})")
            continue
        s = _summ(res)
        dec = s["w"] + s["l"]
        print(f"\n[{run['target']['name']}]  W/L/D {s['w']}/{s['l']}/{s['d']}  "
              f"(판수 {s['n']}, 제외 worker {run['dead']}, {run['elapsed']:.0f}s)")
        if dec:
            lo, hi = wilson_ci(s["w"], dec)
            p = binom_test_two_sided(s["w"], dec)
            verdict = ("ownship 유의 우세" if (p < 0.05 and s["w"] > s["l"])
                       else "target 유의 우세" if (p < 0.05) else "유의차 없음")
            print(f"    승률(D제외) {s['w'] / dec:.3f}  95%CI[{lo:.3f},{hi:.3f}]  "
                  f"p={p:.3g}  → {verdict}")
        exe_fail = sum(r.get("exe_fail", 0) for r in res)
        if exe_fail:
            print(f"    ⚠ exe CMD 타임아웃(안전행동 대체) 총 {exe_fail}회")
        ends = Counter(r["end"] for r in res).most_common()
        print("    end_condition: " + ", ".join(f"{c}×{k}" for k, c in ends))


def parse_args():
    p = argparse.ArgumentParser(
        description="최종 파워 테스트: ownship 을 모든 baseline 상대로 한 번에 붙인다")
    p.add_argument("--ownship", required=True,
                   help="ownship agent spec. bundle:<경로> 또는 ckpt:<경로>")
    p.add_argument("--hz", type=int, choices=[10, 60], default=10,
                   help="ownship 신경망 제어 주기(모든 target 공통, 기본 10)")
    p.add_argument("--action", choices=["stochastic", "argmax"], default="stochastic",
                   help="ownship 신경망 action 선택(기본 stochastic)")
    p.add_argument("--games", type=int, default=100, help="각 target 당 판수(기본 100)")
    return p.parse_args()


def main():
    args = parse_args()
    games = max(1, int(args.games))

    own_spec = parse_agent_spec(args.ownship)
    if not agents.is_nn(own_spec):
        raise ValueError("--ownship 은 학습 모델(bundle:/ckpt:)이어야 합니다.")
    if not Path(own_spec["path"]).exists():
        raise FileNotFoundError(f"--ownship 경로를 찾을 수 없습니다: {own_spec['path']}")

    # ownship 관측 모듈 검증(두 pipeline 모두 claude164r). load_policy 가 내부 검증하지만
    # 시작 전에 한 번 확인해 빠르게 실패시킨다.
    agents.load_policy(own_spec, device="cpu")
    obs_module = agents.OBS_MODULE

    targets = agents.list_baselines()
    if not targets:
        raise RuntimeError("붙일 수 있는 baseline 상대를 하나도 찾지 못했습니다(baselines/ 확인).")

    own_det = (args.action == "argmax")
    own_desc = f"{own_spec['kind']}:{own_spec['name']}({args.hz}Hz, {args.action})"

    max_engage = float(STANDARD_ENV_CONFIG["max_engage_time"])
    step_limit = int(STANDARD_ENV_CONFIG["episode_step_limit"])
    game_timeout = max(180.0, max_engage * 2.0)   # 한 판 wall-clock 상한(행 감지)
    n_workers = max(1, min(physical_cpu_count(), games))
    master_seed = int.from_bytes(os.urandom(4), "little")

    print(f"[final_power_test] ownship = {own_desc}")
    print(f"[final_power_test] target 당 {games}판, workers={n_workers}, master_seed={master_seed}")
    print(f"[final_power_test] max_engage={max_engage}s step_limit={step_limit} "
          f"game_timeout={game_timeout:.0f}s")
    print(f"[final_power_test] targets({len(targets)}): "
          + ", ".join(t["name"] for t in targets))

    all_runs = []
    t_all = time.time()
    for idx, target in enumerate(targets, 1):
        print(f"\n{'#' * 84}")
        print(f"# [{idx}/{len(targets)}] target = {target['name']}")
        print(f"{'#' * 84}", flush=True)
        try:
            run = run_target(target, own_spec, args.hz, own_det, games, n_workers,
                             master_seed, max_engage, step_limit, game_timeout, obs_module)
        except Exception as e:
            # 한 target 이 통째로 실패해도 나머지 target 은 계속 진행.
            import traceback
            print(f"  !! target {target['name']} 실행 실패 — 건너뜀: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass
            run = {"target": target, "results": [], "dead": 0,
                   "played": 0, "requested": games, "elapsed": 0.0}
        all_runs.append(run)
        print(f"  → {target['name']}: {run['played']}/{games} 판 완료, "
              f"제외 worker {run['dead']}", flush=True)

    _report(all_runs, own_desc, games, master_seed, time.time() - t_all)


if __name__ == "__main__":
    main()
