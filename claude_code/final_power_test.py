# -*- coding: utf-8 -*-
"""최종 파워 테스트: 한 번 실행으로 ownship 모델을 **모든 target** 상대로 붙인다.

`power_test.py` / `power_test_unreal.py` 와 **동일한 env·초기화·제어 주기·판정·통계**를
그대로 쓰되, 이 스크립트 한 번으로 지정한 ownship 이 프로젝트의 모든 상대(BT DLL 전부,
Release_MPC_team_share MPC, 내 provider 컷오프, 팀원 provider 컷오프, gylee)와 차례로
싸우고 종합 리포트를 출력한다.

지정할 수 있는 옵션은 **딱 2개**:
  --ownship {basic, altguard, altblend}   ownship 모델
      basic    : basic 번들(10Hz stochastic)
      altguard : 평상시 basic 번들(10Hz) + 고도 3000ft 이하에서 team-share MPC(60Hz) 하드 스위치
      altblend : 4000ft~2000ft 구간에서 actor·MPC action 을 고도 선형 가중평균
                 (4000ft=순수 actor, 3000ft=50:50, 2000ft=순수 MPC)
  --games N                     각 target 당 대결 판수(기본 100)

target: 모든 BT DLL + Release MPC(team_MPC) + Stable MPC(stable_MPC, safe_mpc) +
내 provider 컷오프 + 팀원 provider 컷오프 + gylee.

제어 주기(대회 서버 방식 그대로):
  - 모든 BT / team-share MPC(Release·Stable) / 내 provider 컷오프(UnrealExeProvider) → 60Hz(매 substep)
  - 팀원 provider 컷오프(CutoffUDPActionProvider) / gylee            → 10Hz
  - ownship: basic 10Hz stochastic, altguard 는 저고도에서 MPC 60Hz,
    altblend 는 4000~2000ft 에서 actor·MPC 가중평균(MPC 부분 60Hz)
  - gylee 와 ownship 모델은 stochastic.

BT rule XML(AIP_RULE_XML)은 프로세스 전역이라 DLL 마다 rule 이 다르면 한 프로세스에
하나만 로드된다. 그래서 **target 마다 독립 Ray 세션**을 새로 띄우고(그 target 의 rule 을
worker 시작 시점에 주입), target 이 끝나면 세션을 내린다. 이렇게 하면 rule 격리와
장애 격리(한 target 이 죽어도 다음 target 은 정상)가 동시에 된다.

내결함성: BT 상대와 붙을 때 종종 worker 가 크래시/행 되는 문제가 있어, worker 를
하나씩 게임을 배정(work-queue)하고 크래시/타임아웃 난 worker 는 그 즉시 제외한 뒤 남은
worker 들로 큐를 끝까지 돌린다. 살아남은 게임 결과만 종합한다.

예시:
  python claude_code/final_power_test.py --ownship basic --games 100
  python claude_code/final_power_test.py --ownship altguard --games 50
  python claude_code/final_power_test.py --ownship altblend --games 50
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

from claude_code.bt_rule import BT_RULE_DEFAULTS, ENV_KEY  # noqa: E402 (leaf 모듈)
from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402

# ── 고정 자산 경로 ────────────────────────────────────────────────────────────
BASIC_BUNDLE = str(ROOT / "artifacts" / "models" / "team01" / "basic")
MPC_ROOT = str(ROOT / "Release_MPC_team_share")
STABLE_MPC_ROOT = str(ROOT / "Stable_MPC_team_share")
CUTOFF_EXE = str(ROOT / "unreal_bt_client.exe")
GYLEE_SNAPSHOT = str(ROOT / "model2_gylee" / "model" / "iter_1415.pt")
GUARD_ALT_FT = 3000.0
BLEND_HI_FT = 4000.0   # altblend: 이 고도 이상 = 순수 actor (MPC 가중 0)
BLEND_LO_FT = 2000.0   # altblend: 이 고도 이하 = 순수 MPC (MPC 가중 1)
CUTOFF_BASE_PORT = 9600


# ── team-share MPC 계열 target 래퍼 ───────────────────────────────────────────
class _MPCTargetWrapper:
    """env 상태(pqr/시간 없음)를 각속도 추정으로 보강해 team-share MPC 계열 provider
    (Release `MPCActionProvider` / Stable `SafeMPCActionProvider`)를 60Hz target 으로 감싼다.

    env 는 target provider 에 [0:9](NED pos·euler·body vel)만 채운 상태를 준다. MPC 는
    pqr(deg)@[9:12], 시간@[41] 을 기대하므로 각속도 추정기(60Hz)와 substep 카운터로 채워
    51D state 를 만든다(altguard·대회 서버와 동일). 내부 action_repeat=6 이 10Hz replan /
    60Hz 출력을 만든다. 의존 클래스(각속도추정기/ActionContext/ActionResult)는 mpc 패키지
    경로가 sys.path 에 붙은 뒤라야 import 되므로 생성 측에서 주입한다."""

    def __init__(self, base, rate_cls, sim_hz, ac_cls, ar_cls, source):
        self.mpc = base
        self._own_rates = rate_cls()
        self._tgt_rates = rate_cls()
        self._sim_hz = float(sim_hz)
        self._AC = ac_cls
        self._AR = ar_cls
        self._source = source
        self._sub = 0

    def reset(self, context=None):
        self.mpc.reset(None)
        self._own_rates.reset()
        self._tgt_rates.reset()
        self._sub = 0

    @staticmethod
    def _st(s9, pqr_deg, t):
        st = np.zeros(51, dtype=np.float64)
        st[0:9] = np.asarray(s9, dtype=np.float64)[0:9]
        st[9:12] = np.asarray(pqr_deg, dtype=np.float64)
        st[41] = float(t)
        return st

    def compute_action(self, context):
        own = np.asarray(context.ownship_state, dtype=np.float64)   # MPC 자신
        tgt = np.asarray(context.target_state, dtype=np.float64)    # 본 기체
        t = self._sub / self._sim_hz
        own_pqr = np.degrees(self._own_rates.update(own[3:6], t))
        tgt_pqr = np.degrees(self._tgt_rates.update(tgt[3:6], t))
        res = self.mpc.compute_action(self._AC(
            sim=None, opponent_sim=None,
            ownship_state=self._st(own, own_pqr, t),
            target_state=self._st(tgt, tgt_pqr, t),
            info={"frame_index": self._sub}))
        self._sub += 1
        return self._AR(action=np.asarray(res.action, dtype=np.float32),
                        source=self._source, confidence=1.0, info={})

    def close(self):
        try:
            self.mpc.close()
        except Exception:
            pass


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


# ── target 목록 구성 ──────────────────────────────────────────────────────────
def discover_targets() -> list[dict]:
    """이 프로젝트에서 붙일 수 있는 모든 target 스펙 목록을 만든다.

    각 dict: kind, name, hz, (BT 면) dll·rule.
    - BT: BT_RULE_DEFAULTS 에 등록됐고 실제 DLL 파일이 있는 것 전부(60Hz).
    - team_mpc: Release_MPC_team_share MPC(60Hz).
    - cutoff_mine: 내 UnrealExeProvider 로 구동하는 컷오프 exe(60Hz).
    - cutoff_team: 팀원 CutoffUDPActionProvider 로 구동하는 컷오프 exe(10Hz).
    - gylee: 팀원 model2_gylee 에이전트(10Hz, stochastic).
    """
    targets: list[dict] = []
    for dll, rule in sorted(BT_RULE_DEFAULTS.items()):
        if (ROOT / dll).is_file():
            targets.append({"kind": "bt", "name": f"BT:{dll}", "hz": 60,
                            "dll": dll, "rule": rule})
    if Path(MPC_ROOT).is_dir():
        targets.append({"kind": "team_mpc", "name": "team_MPC", "hz": 60})
    if (Path(STABLE_MPC_ROOT) / "src" / "safe_mpc").is_dir():
        targets.append({"kind": "stable_mpc", "name": "stable_MPC", "hz": 60})
    if Path(CUTOFF_EXE).is_file():
        targets.append({"kind": "cutoff_mine", "name": "cutoff(mine/60Hz)", "hz": 60})
        targets.append({"kind": "cutoff_team", "name": "cutoff(team/10Hz)", "hz": 10})
    if Path(GYLEE_SNAPSHOT).is_file():
        targets.append({"kind": "gylee", "name": "gylee", "hz": 10})
    return targets


# ── Ray worker ────────────────────────────────────────────────────────────────
def _make_worker_cls():
    import ray

    @ray.remote
    class GameWorker:
        """env + ownship provider + (이 target 전용) target provider 를 한 번 만들고,
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

            # ── ownship ──────────────────────────────────────────────────────
            if spec["ownship"] == "altguard":
                # 복합 에이전트: 매 substep 호출 → 내부에서 basic 10Hz / MPC 60Hz.
                from claude_code.altguard_provider import AltGuardMPCProvider
                self.own_provider = AltGuardMPCProvider(
                    spec["basic_bundle"], mpc_root=spec["mpc_root"],
                    mpc_config_path=None, step_ratio=self.step_ratio,
                    device="cpu", stochastic=True, guard_altitude_ft=spec["guard_alt_ft"])
            elif spec["ownship"] == "altblend":
                # 고도 선형 블렌딩: 매 substep 호출 → actor(10Hz)·MPC(60Hz) 가중평균.
                from claude_code.altblend_provider import AltBlendMPCProvider
                self.own_provider = AltBlendMPCProvider(
                    spec["basic_bundle"], mpc_root=spec["mpc_root"],
                    mpc_config_path=None, step_ratio=self.step_ratio,
                    device="cpu", stochastic=True,
                    blend_hi_ft=spec["blend_hi_ft"], blend_lo_ft=spec["blend_lo_ft"])
            else:  # basic
                from claude_code.action_provider import MLPActionProvider
                from claude_code.evaluate import _ActionRepeatProvider
                inner = MLPActionProvider(bundle_dir=spec["basic_bundle"], stochastic=True)
                self.own_provider = _ActionRepeatProvider(inner, self.step_ratio)
            self.env._ownship_action_provider = self.own_provider

            # ── target(이 worker 세션 전용 kind) ──────────────────────────────
            kind = spec["target_kind"]
            if kind == "bt":
                # 학습 opponent pool 과 동일한 BTActionProvider 경로(60Hz, 매 substep).
                from claude_code.self_play import make_bt_provider
                self.tgt_provider = make_bt_provider(spec["bt_dll"], spec.get("bt_rule", ""))
            elif kind == "team_mpc":
                # Release team-share MPC(60Hz). env 상태엔 pqr/시간이 없어 _MPCTargetWrapper
                # 가 각속도 추정으로 채운다(대회 서버 방식). 내부 action_repeat=6=10Hz replan.
                from claude_code.self_play import make_mpc_provider
                from claude_code.my_observation import SIM_HZ
                from dogfight.ai.action_provider import ActionContext, ActionResult
                base = make_mpc_provider(spec["mpc_root"], "")
                # make_mpc_provider 가 <mpc_root>/src 를 sys.path 에 붙인 뒤라야 import 가능.
                from mpc.transforms import AngularRateEstimator
                self.tgt_provider = _MPCTargetWrapper(
                    base, AngularRateEstimator, SIM_HZ, ActionContext, ActionResult,
                    "team_mpc_target")
            elif kind == "stable_mpc":
                # Stable team-share MPC(safe_mpc: Release MPC + predictive 지상안전 실드, 60Hz).
                # provider 는 <root>/src 의 safe_mpc + (shared) mpc 를 쓴다. mpc 핵심 모듈은
                # Release 와 byte-identical 이라 altguard 가 먼저 Release mpc 를 로드해도 무방.
                import sys as _sys
                from claude_code.my_observation import SIM_HZ
                from dogfight.ai.action_provider import ActionContext, ActionResult
                sroot = Path(spec["stable_mpc_root"]).resolve()
                ssrc = str(sroot / "src")
                if ssrc not in _sys.path:
                    _sys.path.append(ssrc)
                from mpc.config import load_config
                from mpc.transforms import AngularRateEstimator
                from safe_mpc import SafeMPCActionProvider, load_safe_mpc_config
                base = SafeMPCActionProvider(
                    sroot, load_config(str(sroot / "configs" / "mpc.yaml")),
                    load_safe_mpc_config(str(sroot / "configs" / "safe_mpc.yaml")))
                self.tgt_provider = _MPCTargetWrapper(
                    base, AngularRateEstimator, SIM_HZ, ActionContext, ActionResult,
                    "stable_mpc_target")
            elif kind == "cutoff_mine":
                # 내 provider(UnrealExeProvider): z=-D 규약, action-repeat=1=60Hz, 자동 복구.
                from claude_code.unreal_exe_provider import UnrealExeProvider
                self.tgt_provider = UnrealExeProvider(
                    exe_path=spec["cutoff_exe"], port=int(spec["cutoff_base_port"]) + int(spec["wid"]),
                    own_plane_id=1, enemy_plane_id=0, ownship_force_side=1, target_force_side=2,
                    cwd=spec["root"], step_timeout_sec=0.5, quiet=True)
            elif kind == "cutoff_team":
                # 팀원 정본 provider(organizer 프로토콜, action_repeat=6=10Hz).
                from cutoff_udp_provider import CutoffUDPActionProvider
                self.tgt_provider = CutoffUDPActionProvider(
                    spec["cutoff_exe"], spec["cutoff_log_dir"], spec["wid"], action_repeat=6)
            elif kind == "gylee":
                # 팀원 model2_gylee 에이전트(자체 47D 관측·RMS·reconstructor, 10Hz, stochastic).
                from model2_gylee import make_opponent_provider
                self.tgt_provider = make_opponent_provider(
                    snapshot_path=spec["gylee_snapshot"], step_ratio=self.step_ratio,
                    device="cpu", explore=True, verify_checksum=False)
            else:
                raise ValueError(f"unknown target kind: {kind}")
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
def run_target(target: dict, ownship: str, games: int, n_workers: int,
               master_seed: int, max_engage: float, step_limit: int,
               game_timeout: float, obs_module: str) -> dict:
    import ray

    kind = target["kind"]
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
        "ownship": ownship, "basic_bundle": BASIC_BUNDLE,
        "mpc_root": MPC_ROOT, "stable_mpc_root": STABLE_MPC_ROOT,
        "guard_alt_ft": GUARD_ALT_FT,
        "blend_hi_ft": BLEND_HI_FT, "blend_lo_ft": BLEND_LO_FT,
        "target_kind": kind,
        "bt_dll": target.get("dll", ""), "bt_rule": target.get("rule", ""),
        "cutoff_exe": CUTOFF_EXE, "cutoff_base_port": CUTOFF_BASE_PORT,
        "cutoff_log_dir": str(ROOT / "artifacts" / "cutoff_logs"),
        "gylee_snapshot": GYLEE_SNAPSHOT,
    }

    # ── target 전용 Ray 세션(BT rule 은 worker 시작 시점에 주입) ──
    pythonpath = os.pathsep.join([str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
    env_vars = {"PYTHONPATH": pythonpath}
    if kind == "bt" and target.get("rule"):
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


def _report(all_runs: list, ownship: str, requested: int, master_seed: int,
            total_elapsed: float) -> None:
    print(f"\n{'=' * 84}")
    print(f"최종 파워 테스트 결과   ownship = {ownship}   "
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
        description="최종 파워 테스트: ownship 을 모든 target 상대로 한 번에 붙인다")
    p.add_argument("--ownship", choices=["basic", "altguard", "altblend"], default="basic",
                   help="basic = basic 번들(10Hz stochastic), "
                        "altguard = basic + 고도 3000ft 이하 team-share MPC(60Hz) 하드 스위치, "
                        "altblend = 4000ft~2000ft 에서 actor·MPC action 고도 선형 가중평균")
    p.add_argument("--games", type=int, default=100, help="각 target 당 판수(기본 100)")
    return p.parse_args()


def main():
    args = parse_args()
    games = max(1, int(args.games))

    if not Path(BASIC_BUNDLE).is_dir():
        raise FileNotFoundError(f"basic 번들을 찾을 수 없습니다: {BASIC_BUNDLE}")

    # ownship 관측 모듈(basic/altguard/altblend 모두 basic 번들 관측을 씀).
    from claude_code.model import load_bundle
    _, meta = load_bundle(BASIC_BUNDLE, device="cpu")
    obs_module = meta.get("observation_module", "") or ""
    if args.ownship in ("altguard", "altblend") and obs_module != "claude_code.my_observation":
        raise ValueError(
            f"{args.ownship} 는 claude164r(my_observation) 번들이 필요합니다: {obs_module!r}")

    targets = discover_targets()
    if not targets:
        raise RuntimeError("붙일 수 있는 target 을 하나도 찾지 못했습니다.")

    max_engage = float(STANDARD_ENV_CONFIG["max_engage_time"])
    step_limit = int(STANDARD_ENV_CONFIG["episode_step_limit"])
    game_timeout = max(180.0, max_engage * 2.0)   # 한 판 wall-clock 상한(행 감지)
    n_workers = max(1, min(physical_cpu_count(), games))
    master_seed = int.from_bytes(os.urandom(4), "little")

    print(f"[final_power_test] ownship = {args.ownship}  (obs_module={obs_module or '(env default)'})")
    print(f"[final_power_test] target 당 {games}판, workers={n_workers}, "
          f"master_seed={master_seed}")
    print(f"[final_power_test] max_engage={max_engage}s step_limit={step_limit} "
          f"game_timeout={game_timeout:.0f}s")
    print(f"[final_power_test] targets({len(targets)}): "
          + ", ".join(f"{t['name']}({t['hz']}Hz)" for t in targets))

    all_runs = []
    t_all = time.time()
    for idx, target in enumerate(targets, 1):
        print(f"\n{'#' * 84}")
        print(f"# [{idx}/{len(targets)}] target = {target['name']} ({target['hz']}Hz)")
        print(f"{'#' * 84}", flush=True)
        try:
            run = run_target(target, args.ownship, games, n_workers, master_seed,
                             max_engage, step_limit, game_timeout, obs_module)
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

    _report(all_runs, args.ownship, games, master_seed, time.time() - t_all)


if __name__ == "__main__":
    main()
