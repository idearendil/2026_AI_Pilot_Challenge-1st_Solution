# -*- coding: utf-8 -*-
"""리그 파워테스트: 10개 에이전트를 모든 쌍끼리 붙여 승률 행렬·heatmap 을 만든다.

환경
----
초기 분포는 CUDA PPO 학습과 동일하게 **시나리오 A(3-9 line) : B(head-on) = 4:1**
(scenario_b_prob=0.2, head-on 시작 거리 5,539m). JSBSim env(claude_code.env_utils.make_env)를
중재자로 쓰고, 양쪽 슬롯에 각 에이전트의 ActionProvider 를 꽂아 싸운다. 모든 신경망 모델은
**argmax·10Hz**.

대상(10)
--------
  - mine_mlp  : final_team_models/mine/mine_mlp_actor.pt   (내 PPO MLP, 최종 main actor 추출본)
  - final_team_models/<7개>                    (팀원 패키지: 번들 inference.FlightPolicy, argmax·10Hz)
  - mpc       : baselines/Release_MPC_team_share   (CEM + 네이티브 predictor)
  - cutoff    : baselines/unreal_bt_client.exe     (외부 BT UDP 클라이언트)

통합 규약
---------
  - 팀 7모델은 각자 번들 claude_code(my_observation·observation_contract)를 쓰므로
    이름충돌(claude_code.my_observation)을 피하려 **독립 서브프로세스**(_league_team_worker)
    로 격리 실행하고 stdio 로 (own,tgt)->command 브리지한다(관측 충실성 유지).
  - mine_mlp 는 메인 claude_code.my_observation 으로 학습됐으므로 in-process SelfPlayProvider
    (explore=False=argmax)로 굴린다(독립 reconstructor → 싱글톤 오염 없음).
  - mpc/cutoff 는 기존 provider(MPCActionProvider / UnrealExeProvider)를 그대로 쓴다.
  - env 는 provider 에 '자기/상대' 관점 context(ownship_state=자기 기체)를 주므로 어떤
    provider 든 슬롯 무관하게 동작한다.

출력
----
승률 행렬(행=해당 에이전트가 ownship, 열=상대)과 각 에이전트 평균 승률을 색으로 칠한
heatmap PNG 를 프로젝트 루트에 저장한다.

예:
  python -m claude_code.league --games 50 --num-workers 8 --out league_winrate.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np  # noqa: E402

FT2M = 0.3048
HEADON_DIST_M = 5539.0                     # 본선/CUDA 학습 env 와 동일한 head-on 시작 거리
HEADON_DIST_FT = HEADON_DIST_M / FT2M      # env_utils 는 ft 로 받는다


# ── 에이전트 레지스트리 ────────────────────────────────────────────────────────
def default_agents() -> list[dict]:
    tm = ROOT / "final_team_models"
    return [
        # mine_* 은 소형 actor+norm 파일(전체 PPO ckpt=2.25GB/634MB 에서 최종 main actor 만
        # 추출한 것; 워커마다 거대 ckpt 를 로드하면 OOM 이라 사전 추출본을 쓴다).
        # mine_mlp = final_team_models/mine 의 소형 actor+norm 체크포인트(MLP; LSTM 은 제외).
        {"name": "mine_mlp", "kind": "mine_mlp",
         "ckpt": str(tm / "mine" / "mine_mlp_actor.pt")},
        {"name": "gylee_20k", "kind": "team", "pkg": str(tm / "gylee_20k")},
        {"name": "gylee_46k", "kind": "team", "pkg": str(tm / "gylee_46k_headon")},
        {"name": "gylee_69k", "kind": "team", "pkg": str(tm / "gylee_69k")},
        {"name": "jh_exploiter", "kind": "team", "pkg": str(tm / "junhwa_exploiter_survive_v1")},
        {"name": "jh_final", "kind": "team", "pkg": str(tm / "junhwa_final")},
        {"name": "jh_gru", "kind": "team", "pkg": str(tm / "junhwa_grid_3L749_gru_last")},
        {"name": "jh_wide1", "kind": "team", "pkg": str(tm / "junhwa_wide1_v2")},
        {"name": "mpc", "kind": "mpc", "mpc_root": str(ROOT / "baselines" / "Release_MPC_team_share")},
        {"name": "cutoff", "kind": "cutoff", "exe": str(ROOT / "baselines" / "unreal_bt_client.exe")},
    ]


# ── 승패 판정(power_test 와 동일: ownship 관점, 대칭) ──────────────────────────
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


# ── provider 빌더(Ray worker 내부에서 호출) ──────────────────────────────────
def _load_mine(kind: str, ckpt_path: str):
    """내 PPO 학습 checkpoint 에서 **최종 main actor net** 만 추출해 (model, rms) 반환.

    ckpt["model"] = 학습된 main ActorCritic state_dict(actor+critic+aux). 여기서 actor 파라미터
    (LSTM=actor_lstm/actor_logits, MLP=actor_body/actor_logits; aux head·critic 제외)만 뽑아
    추론용 모델에 싣는다. obs_rms 는 ckpt["norm"](running mean/var/count)에서 가져온다.
    """
    import torch
    from collections import OrderedDict
    from claude_code.normalizers import RunningMeanStd

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_sd = ck["model"]
    # 최종 main actor net 만 추출(critic·aux head 제외).
    sd = OrderedDict((k, v) for k, v in model_sd.items()
                     if k.startswith(("actor_body.", "actor_logits."))
                     and not k.startswith("actor_aux_head."))
    norm = ck.get("norm") or {}
    rms = None
    if norm:
        def _np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        rms = RunningMeanStd.from_state_dict({
            "mean": _np(norm["mean"]).astype(np.float64),
            "var": _np(norm["var"]).astype(np.float64),
            "count": float(_np(norm["count"]))})
    num_bins = int(ck.get("cfg", {}).get("num_bins", 21))

    # mine_mlp: actor_body/actor_logits → MLPDiscreteActor(actor_logits.{0,2,..,2L})
    from claude_code.model import MLPDiscreteActor
    hidden = tuple(sd[f"actor_body.{i}.weight"].shape[0]
                   for i in range(0, 99, 2) if f"actor_body.{i}.weight" in sd)
    obs = int(sd["actor_body.0.weight"].shape[1])
    head = 2 * len(hidden)
    remap = OrderedDict()
    for k, v in sd.items():
        if k.startswith("actor_body."):
            remap["actor_logits." + k[len("actor_body."):]] = v
        elif k.startswith("actor_logits."):
            remap[f"actor_logits.{head}." + k[len("actor_logits."):]] = v
    model = MLPDiscreteActor(obs_dim=obs, act_dim=4, num_bins=num_bins, hidden=hidden)
    model.load_state_dict(remap, strict=True)
    model.eval()
    return model, rms


def build_provider(spec: dict, env, step_ratio: int, port: int):
    """에이전트 spec → ActionProvider. env 의 한 슬롯에 꽂는다."""
    kind = spec["kind"]
    if kind == "mine_mlp":
        from claude_code.self_play import SelfPlayProvider
        model, rms = _load_mine(kind, spec["ckpt"])
        return SelfPlayProvider(model, rms, env._observation_fn, env._observation_mode,
                                step_ratio, "cpu", explore=False)   # explore=False → argmax
    if kind == "team":
        return SubprocTeamProvider(spec["pkg"], step_ratio)
    if kind == "mpc":
        from claude_code.self_play import make_mpc_provider
        return make_mpc_provider(spec["mpc_root"])
    if kind == "cutoff":
        from claude_code.unreal_exe_provider import UnrealExeProvider
        return UnrealExeProvider(exe_path=spec["exe"], port=port)
    raise ValueError(f"unknown agent kind: {kind}")


# ── 팀 모델 서브프로세스 브리지 provider ──────────────────────────────────────
def _make_subproc_team_cls():
    from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

    class _SubprocTeamProvider(ActionProvider):
        """팀 패키지 inference.FlightPolicy 를 독립 서브프로세스로 돌리는 10Hz provider.

        env 는 substep(60Hz)마다 compute_action 을 부르지만, 팀 모델은 10Hz 정책이라
        step_ratio 마다 한 번만 서브프로세스에 (own,tgt)->command 질의하고 사이엔 반복한다.
        """

        def __init__(self, pkg: str, step_ratio: int):
            import json as _json
            import subprocess
            self._json = _json
            self.repeat = max(1, int(step_ratio))
            worker = str(ROOT / "claude_code" / "_league_team_worker.py")
            self.proc = subprocess.Popen(
                [sys.executable, worker, pkg],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                cwd=str(ROOT), text=True, bufsize=1)
            ready = self.proc.stdout.readline()             # {"ready":1}
            if not ready or "ready" not in ready:
                raise RuntimeError(f"team worker 준비 실패: {pkg!r} (got {ready!r})")
            self._count = 0
            self._cached = None

        def _rpc(self, msg: dict) -> dict:
            self.proc.stdin.write(self._json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("team worker 응답 없음(프로세스 종료?)")
            return self._json.loads(line)

        def reset(self, context: ActionContext | None = None) -> None:
            self._count = 0
            self._cached = None
            self._rpc({"t": "reset"})

        def compute_action(self, context: ActionContext) -> ActionResult:
            if self._cached is None or self._count % self.repeat == 0:
                own = np.asarray(context.ownship_state, dtype=np.float64).reshape(-1)[:9]
                tgt = np.asarray(context.target_state, dtype=np.float64).reshape(-1)[:9]
                r = self._rpc({"t": "act", "own": own.tolist(), "tgt": tgt.tolist()})
                cmd = np.asarray(r["a"], dtype=np.float32).reshape(-1)[:4]
                self._cached = ActionResult(action=cmd, source="team", confidence=1.0, info={})
            self._count += 1
            return self._cached

        def close(self) -> None:
            try:
                self._rpc_quit()
            except Exception:
                pass

        def _rpc_quit(self):
            try:
                self.proc.stdin.write(self._json.dumps({"t": "quit"}) + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    return _SubprocTeamProvider


# lazy global (Ray worker 프로세스에서 클래스 생성). 드라이버 프로세스에선 안 만든다.
SubprocTeamProvider = None   # type: ignore


# ── Ray worker: 한 쌍(i vs j) N판 ─────────────────────────────────────────────
def _make_pair_worker_cls():
    import ray

    @ray.remote
    class PairWorker:
        def __init__(self, root: str, overrides: dict, obs_module: str, step_ratio: int):
            os.chdir(root)
            import torch
            torch.set_num_threads(1)
            global SubprocTeamProvider
            if SubprocTeamProvider is None:
                SubprocTeamProvider = _make_subproc_team_cls()
            self.root = root
            self.overrides = overrides
            self.obs_module = obs_module
            self.step_ratio = step_ratio

        def play_pair(self, agent_i, agent_j, jobs, port):
            import torch
            from claude_code.env_utils import make_env
            env = make_env(overrides=self.overrides, reward_module="",
                           observation_module=self.obs_module, runner_index=f"lg{port}")
            pi = build_provider(agent_i, env, self.step_ratio, port)
            pj = build_provider(agent_j, env, self.step_ratio, port + 10000)
            env._ownship_action_provider = pi     # i = ownship
            env._target_action_provider = pj      # j = target
            zero = np.zeros(4, dtype=np.float32)
            wins_i = wins_j = draws = 0
            per = []
            for seed, swap, head in jobs:
                torch.manual_seed(int(seed))
                env._apply_start_side(bool(swap), bool(head))
                env.reset(seed=int(seed))
                for p in (pi, pj):
                    if hasattr(p, "reset"):
                        p.reset()
                term = trunc = False
                info = {}
                steps = 0
                while not (term or trunc):
                    _o, _r, term, trunc, info = env.step(zero)
                    steps += 1
                own_hp = float(info.get("ownship_health", float("nan")))
                tgt_hp = float(info.get("target_health", float("nan")))
                end = str(info.get("end_condition", ""))
                oc = game_outcome(end, own_hp, tgt_hp)
                if oc == "win":
                    wins_i += 1
                elif oc == "loss":
                    wins_j += 1
                else:
                    draws += 1
                per.append({"seed": int(seed), "swap": int(swap), "head": int(head),
                            "steps": steps, "own_hp": own_hp, "tgt_hp": tgt_hp,
                            "end": end, "outcome": oc})
            for p in (pi, pj):
                if hasattr(p, "close"):
                    try:
                        p.close()
                    except Exception:
                        pass
            try:
                env.close()
            except Exception:
                pass
            return {"i": agent_i["name"], "j": agent_j["name"],
                    "wins_i": wins_i, "wins_j": wins_j, "draws": draws,
                    "games": len(jobs), "per_game": per}

    return PairWorker


# ── heatmap 렌더 ──────────────────────────────────────────────────────────────
def render_heatmap(names, win_rate, avg, out_path, games_per_pair, draw_rate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    # 행렬 + 평균 열을 하나의 이미지로: 좌측 NxN heatmap, 우측 평균 막대.
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            M[i, j] = win_rate[i][j]

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(max(9, n * 0.95 + 3), max(7, n * 0.7 + 2)),
        gridspec_kw={"width_ratios": [n, 2.2], "wspace": 0.05})

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#444444")
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("opponent (column)", fontsize=10)
    ax.set_ylabel("agent (row = ownship)", fontsize=10)
    ax.set_title(f"League win rate  (row beats column, {games_per_pair} games/pair, "
                 f"3-9:head-on=4:1)", fontsize=11)
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(M[i, j]):
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < v < 0.8 else "white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="win rate")

    order = np.argsort(-np.asarray(avg))
    y = np.arange(n)
    axb.barh(y, [avg[k] for k in range(n)], color=[cmap(avg[k]) for k in range(n)],
             edgecolor="#333", height=0.7)
    axb.set_yticks(y); axb.set_yticklabels([])
    axb.set_ylim(ax.get_ylim())
    axb.set_xlim(0, 1)
    axb.set_xlabel("avg win rate", fontsize=10)
    axb.set_title("overall avg", fontsize=10)
    for k in range(n):
        axb.text(min(avg[k] + 0.02, 0.98), k, f"{avg[k]:.2f}", va="center", fontsize=8)
    axb.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 순위 텍스트도 같이 반환
    return [(names[k], float(avg[k])) for k in order]


# ── CLI / 드라이버 ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="리그 파워테스트(3-9:head-on=4:1) → 승률 heatmap")
    p.add_argument("--games", type=int, default=50, help="각 쌍이 치르는 판수(>=50 권장)")
    p.add_argument("--num-workers", type=int, default=0, help="병렬 worker 수(0=물리 코어)")
    p.add_argument("--seed", type=int, default=-1, help="마스터 시드(생략/-1=랜덤)")
    p.add_argument("--base-port", type=int, default=9300, help="cutoff exe UDP 시작 포트")
    p.add_argument("--max-engage-time", type=float, default=200.0, help="최대 교전 시간(s)")
    p.add_argument("--out", default=str(ROOT / "league_winrate.png"),
                   help="heatmap PNG 저장 경로(프로젝트 루트 기본)")
    p.add_argument("--csv", default="", help="쌍별 집계 CSV 저장 경로(선택)")
    return p.parse_args()


def main():
    args = parse_args()
    from claude_code.env_utils import STANDARD_ENV_CONFIG
    from claude_code.parallel import physical_cpu_count

    agents = default_agents()
    # 존재 확인
    for a in agents:
        probe = a.get("pkg") or a.get("ckpt") or a.get("mpc_root") or a.get("exe")
        if probe and not Path(probe).exists():
            raise FileNotFoundError(f"에이전트 {a['name']} 경로 없음: {probe}")

    n = len(agents)
    names = [a["name"] for a in agents]
    games = max(1, int(args.games))
    step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))
    master_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(master_seed)

    # 초기 분포: 시나리오 A(3-9 line) : B(head-on) = 4:1 (scenario_b_prob=0.2, CUDA 학습과 동일).
    # head-on(B) 시작 거리는 본선/CUDA 학습 env 와 같은 5539m. 좌우는 직접 교대.
    overrides = {
        "scenario_b_prob": 0.2,
        "start_headon_distance_ft": HEADON_DIST_FT,
        "randomize_start_side": False,
        "target_mode": "fixed",            # 양쪽 다 provider 로 조종(env 자체 AI 비활성)
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": int(STANDARD_ENV_CONFIG["episode_step_limit"]),
    }
    obs_module = "claude_code.my_observation"

    # 모든 비순서 쌍
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"[league] agents={n} pairs={len(pairs)} games/pair={games} "
          f"total_games={len(pairs) * games} master_seed={master_seed}")
    print(f"[league] 3-9:head-on=4:1 (head-on {HEADON_DIST_M:.0f}m), 10Hz, argmax, "
          f"max_engage={args.max_engage_time}s")

    n_workers = args.num_workers if args.num_workers > 0 else physical_cpu_count()
    n_workers = max(1, min(int(n_workers), len(pairs)))

    import ray
    import time
    if not ray.is_initialized():
        pythonpath = os.pathsep.join([str(ROOT), str(ROOT / "src"),
                                      os.environ.get("PYTHONPATH", "")])
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})

    PairWorker = _make_pair_worker_cls()
    workers = [PairWorker.remote(str(ROOT), overrides, obs_module, step_ratio)
               for _ in range(n_workers)]

    # 쌍 작업 분배(라운드로빈). 각 쌍마다 고유 시드열·좌우 교대.
    tasks = []
    for pidx, (i, j) in enumerate(pairs):
        seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=games)]
        swaps = [g % 2 for g in range(games)]
        heads = [(g // 2) % 2 for g in range(games)]
        jobs = list(zip(seeds, swaps, heads))
        w = workers[pidx % n_workers]
        port = args.base_port + pidx * 2
        tasks.append(w.play_pair.remote(agents[i], agents[j], jobs, port))

    t0 = time.time()
    wins = np.zeros((n, n))           # wins[i][j] = i 가 j 를 상대로 이긴 판수
    gmat = np.zeros((n, n))
    idx = {nm: k for k, nm in enumerate(names)}
    agg_rows = []
    done_cnt = 0
    pending = list(tasks)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for ref in ready:
            r = ray.get(ref)
            i, j = idx[r["i"]], idx[r["j"]]
            wins[i][j] += r["wins_i"]; gmat[i][j] += r["games"]
            wins[j][i] += r["wins_j"]; gmat[j][i] += r["games"]
            agg_rows.append(r)
            done_cnt += 1
            print(f"  ... {done_cnt}/{len(pairs)} 쌍  "
                  f"{r['i']} vs {r['j']}: {r['wins_i']}/{r['wins_j']}/{r['draws']} "
                  f"(W/L/D, i관점)  ({time.time()-t0:.0f}s)", flush=True)
    ray.get([w.__ray_terminate__.remote() for w in workers]) if False else None

    # 승률 행렬 + 평균
    win_rate = [[float("nan")] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and gmat[i][j] > 0:
                win_rate[i][j] = wins[i][j] / gmat[i][j]
    avg = [float(np.nansum([wins[i][j] for j in range(n) if j != i]) /
                 max(1.0, np.nansum([gmat[i][j] for j in range(n) if j != i])))
           for i in range(n)]

    ranking = render_heatmap(names, win_rate, avg, args.out, games)
    print(f"\n[league] heatmap 저장 → {args.out}")
    print("[league] 평균 승률 순위:")
    for rank, (nm, a) in enumerate(ranking, 1):
        print(f"   {rank:2d}. {nm:16s} {a:.3f}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wtr = _csv.writer(f)
            wtr.writerow(["i", "j", "wins_i", "wins_j", "draws", "games"])
            for r in agg_rows:
                wtr.writerow([r["i"], r["j"], r["wins_i"], r["wins_j"], r["draws"], r["games"]])
        print(f"[league] 쌍별 집계 CSV → {args.csv}")

    ray.shutdown()


if __name__ == "__main__":
    main()
