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

예시 (팀원 model2_gylee 에이전트 vs unreal exe(컷오프 모델), 100판):
  python claude_code/power_test_unreal.py --ownship-gylee --games 100
    (다른 snapshot 은 --ownship-gylee-snapshot <경로>, stochastic 이면 --ownship-gylee-explore)
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

# 콘솔 인코딩(cp949 등)에 없는 문자(em-dash, 화살표 등)를 print 해도 UnicodeEncodeError 로
# 죽지 않게 한다(train_redq.py·power_test.py 와 동일 관례).
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

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

# 팀원(gyLee) 패키지 기본 opponent snapshot = 문서 계보의 iter_1415(동봉 SHA 일치·검증됨).
# 다른 snapshot(예: iter_2350)을 --ownship-gylee-snapshot 으로 지정하면 SHA 가 달라 checksum 이
# 안 맞으므로 provider 생성 시 verify_checksum=False 로 로드한다(power_test.py 와 동일).
_DEF_GYLEE_SNAPSHOT = str(ROOT / "model2_gylee" / "model" / "iter_1415.pt")


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

            # ownship: 팀원 gylee 에이전트 / MLP 단일 정책 / neural-MPC(obs-WM lookahead). 모두 10Hz.
            gylee = spec.get("gylee")
            if gylee:
                # gylee provider 는 자체 47D 관측·RMS·reconstructor 와 자체 action_repeat 를
                # 가지므로(_ActionRepeatProvider 불필요) ownship provider 로 직접 주입한다.
                # context.ownship_state=본 기체(ownship), target_state=적(exe)로 들어와
                # gylee 가 본 기체를 자기 관점으로 조종한다.
                from model2_gylee import make_opponent_provider
                self.own_provider = make_opponent_provider(
                    snapshot_path=gylee["snapshot"], step_ratio=self.step_ratio,
                    device="cpu", explore=bool(gylee["explore"]), verify_checksum=False)
                self.env._ownship_action_provider = self.own_provider
            elif spec.get("altguard"):
                # 고도 안전망 복합 에이전트: 평상시 basic 번들(10Hz) / 저고도 team-share
                # MPC(60Hz). 매 substep 호출을 받아 내부에서 주기를 맞추므로 직접 주입.
                from claude_code.altguard_provider import AltGuardMPCProvider
                ag = spec["altguard"]
                self.own_provider = AltGuardMPCProvider(
                    spec["ownship_bundle_dir"], mpc_root=ag["mpc_root"],
                    mpc_config_path=(ag["config"] or None), step_ratio=self.step_ratio,
                    device="cpu", stochastic=bool(spec["stochastic"]),
                    guard_altitude_ft=ag["alt_ft"])
                self.env._ownship_action_provider = self.own_provider
            else:
                # MLP/MPC 는 action_repeat=step_ratio 로 감싸 RL-step(0.1s)마다 1회 호출.
                from claude_code.evaluate import _ActionRepeatProvider
                mpc = spec.get("mpc")
                if mpc:
                    from claude_code.mpc_action_provider import MPCActionProvider
                    inner = MPCActionProvider(
                        mpc["wm"], mpc["ac"], device=mpc["device"],
                        K=mpc["K"], M=mpc["M"], H=mpc["H"], decide_every=mpc["decide_every"],
                        use_fast=bool(mpc["use_fast"]))
                else:
                    from claude_code.action_provider import MLPActionProvider
                    inner = MLPActionProvider(bundle_dir=spec["ownship_bundle_dir"],
                                              stochastic=bool(spec["stochastic"]))
                self.own_provider = _ActionRepeatProvider(inner, self.step_ratio)
                self.env._ownship_action_provider = self.own_provider

            # target: exe 브리지. --cutoff-provider 면 팀원 정본 CutoffUDPActionProvider
            # (action_repeat=6=10Hz, organizer 프로토콜)를, 아니면 기존 UnrealExeProvider 사용.
            if spec.get("cutoff_provider"):
                from cutoff_udp_provider import CutoffUDPActionProvider
                self.tgt_provider = CutoffUDPActionProvider(
                    spec["exe_path"], spec["cutoff_log_dir"], spec["wid"],
                    action_repeat=int(spec["cutoff_action_repeat"]))
            else:
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
    p.add_argument("--ownship-bundle-dir",
                   default=str(ROOT / "artifacts" / "models" / "team01" / "basic2"),
                   help="RL(ownship) claude 번들 경로 (--ownship-gylee 면 생략 가능). "
                        "기본 basic2 = 현재 모델 구조(obs 214, accel+aux)로 학습한 번들")
    p.add_argument("--deterministic", action="store_true",
                   help="RL action 을 정책 분포 샘플링 대신 argmax(deterministic)로 결정. "
                        "기본은 학습과 동일한 stochastic 샘플링")
    # ── ownship 을 팀원 model2_gylee 에이전트로 (RL 번들 대신) ──
    p.add_argument("--ownship-gylee", action="store_true",
                   help="ownship 을 내 RL 번들 대신 팀원 model2_gylee 에이전트로 조종해 "
                        "unreal exe(컷오프 모델)와 붙인다. --ownship-bundle-dir 불필요.")
    p.add_argument("--ownship-gylee-snapshot", default=_DEF_GYLEE_SNAPSHOT,
                   help=f"gylee ownship snapshot .pt (기본 {Path(_DEF_GYLEE_SNAPSHOT).name})")
    p.add_argument("--ownship-gylee-explore", action="store_true",
                   help="gylee ownship 을 stochastic(sample)으로. 기본은 deterministic(argmax).")
    # ── ownship = 고도 안전망 복합 에이전트(basic 10Hz + 저고도 team-share MPC 60Hz) ──
    p.add_argument("--ownship-altguard", action="store_true",
                   help="ownship 을 altguard 복합 에이전트로: 평상시 --ownship-bundle-dir(basic) "
                        "10Hz stochastic, 고도 임계값 이하에서 team-share MPC 60Hz 로 자동 전환.")
    p.add_argument("--ownship-guard-altitude-ft", type=float, default=3000.0,
                   help="altguard: 이 고도(ft) 이하에서 MPC 가 조종(기본 3000)")
    p.add_argument("--altguard-mpc-root", default=str(ROOT / "Release_MPC_team_share"),
                   help="altguard team-share MPC 폴더(기본 Release_MPC_team_share)")
    p.add_argument("--altguard-mpc-config", default="",
                   help="altguard MPC config yaml(생략 시 <mpc-root>/configs/mpc.yaml)")
    p.add_argument("--exe-path", default=str(ROOT / "unreal_bt_client.exe"),
                   help="cutoff/unreal_bt_client.exe 경로(기본: 프로젝트 루트)")
    p.add_argument("--cutoff-provider", action="store_true",
                   help="팀원 정본 CutoffUDPActionProvider(organizer 프로토콜, action-repeat=6=10Hz)"
                        "로 exe 를 구동한다. 기본(꺼짐)은 UnrealExeProvider(60Hz). 컷오프 모델을 "
                        "팀원 벤치마크와 동일 규약으로 붙일 때 사용.")
    p.add_argument("--cutoff-action-repeat", type=int, default=6,
                   help="--cutoff-provider 의 exe action-repeat(기본 6=10Hz, 학습·검증 주기).")
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
    # ── neural-MPC ownship (obs-WM lookahead) 옵션 ──
    p.add_argument("--ownship-mpc", action="store_true",
                   help="ownship 을 단일 MLP 대신 neural-MPC(obs-WM lookahead)로 조종")
    p.add_argument("--mpc-wm", default=str(ROOT / "claude_code/models/wm/wm_model_obs.pt"),
                   help="추론호환 obs-WM 체크포인트")
    p.add_argument("--mpc-ac", default="",
                   help="actor/critic ac_ckpt(.pt). 생략하면 --ownship-bundle-dir 에서 자동 생성")
    p.add_argument("--mpc-h", type=int, default=10, help="lookahead 스텝(기본 10=1초)")
    p.add_argument("--mpc-k", type=int, default=12, help="후보 수 K(기본 12)")
    p.add_argument("--mpc-m", type=int, default=8, help="상대샘플 수 M(기본 8)")
    p.add_argument("--mpc-decide-every", type=int, default=1, help="actor 재결정 주기(기본 1)")
    p.add_argument("--mpc-device", default="cuda", help="MPC 추론 디바이스(기본 cuda)")
    p.add_argument("--mpc-eager", action="store_true",
                   help="CUDA-graph plan_fast 대신 eager plan() 사용(느림; 디버그용)")
    return p.parse_args()


def _bundle_to_ac_ckpt(bundle_dir: str, out_path: str) -> str:
    """team01/basic 번들 → planner ac_ckpt(model_kwargs+state_dict+obs_rms) 변환·저장."""
    import numpy as _np
    import torch as _torch
    from claude_code.model import load_bundle
    model, meta = load_bundle(bundle_dir, device="cpu")
    mm = meta["model"]
    mk = dict(obs_dim=int(meta["observation_size"]), act_dim=int(meta.get("action_size", 4)),
              hidden=tuple(mm["hidden"]), activation=mm.get("activation", "tanh"),
              critic_hidden=tuple(mm["critic_hidden"]) if mm.get("critic_hidden") else None,
              critic_activation=mm.get("critic_activation"), num_bins=int(mm["num_bins"]))
    on = meta["obs_normalization"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _torch.save({"model_kwargs": mk,
                 "state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
                 "obs_rms": {"mean": _np.asarray(on["mean"], _np.float64),
                             "var": _np.asarray(on["var"], _np.float64)}}, out_path)
    return out_path


def main():
    args = parse_args()

    if not Path(args.exe_path).exists():
        raise FileNotFoundError(f"exe 를 찾을 수 없습니다: {args.exe_path}")

    # ownship = gylee 에이전트인지 결정. gylee 면 번들 불필요(자체 관측·정책).
    gylee_cfg = None
    if args.ownship_gylee:
        if not Path(args.ownship_gylee_snapshot).is_file():
            raise FileNotFoundError(
                f"--ownship-gylee snapshot 을 찾을 수 없습니다: {args.ownship_gylee_snapshot}")
        gylee_cfg = {"snapshot": str(args.ownship_gylee_snapshot),
                     "explore": bool(args.ownship_gylee_explore)}
        obs_module = ""     # gylee 는 자체 47D 관측을 만들어 env obs_module 과 무관
    else:
        if not args.ownship_bundle_dir:
            raise ValueError("--ownship-bundle-dir 이 필요합니다(또는 --ownship-gylee 사용).")
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

    # 고도 안전망 복합 에이전트(선택). basic 번들 필요(위 else 에서 obs_module 확보).
    altguard_cfg = None
    if args.ownship_altguard:
        if gylee_cfg:
            raise ValueError("--ownship-altguard 와 --ownship-gylee 는 함께 쓸 수 없습니다.")
        if args.ownship_mpc:
            raise ValueError("--ownship-altguard 와 --ownship-mpc(neural) 는 함께 쓸 수 없습니다.")
        if not Path(args.altguard_mpc_root).is_dir():
            raise FileNotFoundError(f"altguard MPC 폴더 없음: {args.altguard_mpc_root}")
        altguard_cfg = {"mpc_root": str(args.altguard_mpc_root),
                        "config": str(args.altguard_mpc_config),
                        "alt_ft": float(args.ownship_guard_altitude_ft)}

    # neural-MPC ownship 설정(선택). ac_ckpt 는 번들에서 자동 생성 가능. (gylee/altguard 면 무시)
    mpc_cfg = None
    if args.ownship_mpc and not gylee_cfg and not altguard_cfg:
        ac_path = args.mpc_ac
        if not ac_path:
            ac_path = str(ROOT / "claude_code/models/wm/_ac_from_bundle.pt")
            _bundle_to_ac_ckpt(args.ownship_bundle_dir, ac_path)
            print(f"[power_test_unreal] ac_ckpt 자동생성: {ac_path} (from {args.ownship_bundle_dir})")
        for pth in (args.mpc_wm, ac_path):
            if not Path(pth).exists():
                raise FileNotFoundError(f"MPC 체크포인트 없음: {pth}")
        mpc_cfg = {"wm": args.mpc_wm, "ac": ac_path, "device": args.mpc_device,
                   "K": int(args.mpc_k), "M": int(args.mpc_m), "H": int(args.mpc_h),
                   "decide_every": int(args.mpc_decide_every), "use_fast": (not args.mpc_eager)}

    games = max(1, int(args.games))
    # MPC 는 GPU 를 쓰므로 기본 worker 1 (여러 워커가 한 GPU 경합 방지). 명시하면 그 값 사용.
    if args.num_workers > 0:
        n_workers = args.num_workers
    else:
        n_workers = 1 if mpc_cfg else physical_cpu_count()
    n_workers = max(1, min(int(n_workers), games))

    master_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(master_seed)
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=games)]
    swaps = [0] * games if args.fixed_side else [i % 2 for i in range(games)]
    head_swaps = [0] * games if args.fixed_side else [(i // 2) % 2 for i in range(games)]

    rl_mode = "deterministic(argmax)" if args.deterministic else "stochastic"
    if gylee_cfg:
        own_desc = (f"gylee({Path(gylee_cfg['snapshot']).name}, "
                    f"{'stochastic' if gylee_cfg['explore'] else 'deterministic'})")
    elif altguard_cfg:
        own_desc = (f"altguard(basic={args.ownship_bundle_dir} 10Hz / "
                    f"team-share MPC<{altguard_cfg['alt_ft']:.0f}ft 60Hz)")
    elif mpc_cfg:
        own_desc = (f"neural-MPC(H={mpc_cfg['H']} K={mpc_cfg['K']} M={mpc_cfg['M']} "
                    f"de={mpc_cfg['decide_every']} wm=obs, ac={args.ownship_bundle_dir})")
    else:
        own_desc = f"rl({args.ownship_bundle_dir})"
    tgt_desc = (f"cutoff-exe({Path(args.exe_path).name}, CutoffUDPProvider, "
                f"action-repeat={args.cutoff_action_repeat})" if args.cutoff_provider
                else f"unreal_bt_client.exe({Path(args.exe_path).name}, 60Hz)")
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
        init_kw = dict(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                       log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        if mpc_cfg and mpc_cfg["device"].startswith("cuda"):
            init_kw["num_gpus"] = 1          # MPC 는 GPU 필요 (Ray 가 CUDA 숨기지 않도록)
        ray.init(**init_kw)

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "ownship_bundle_dir": args.ownship_bundle_dir,
        "exe_path": str(args.exe_path), "base_port": int(args.base_port),
        "ownship_force_side": int(args.ownship_force_side),
        "target_force_side": int(args.target_force_side),
        "step_timeout_sec": float(args.step_timeout_sec),
        "stochastic": (not args.deterministic),
        "mpc": mpc_cfg,
        "altguard": altguard_cfg,
        "gylee": gylee_cfg,
        "cutoff_provider": bool(args.cutoff_provider),
        "cutoff_action_repeat": int(args.cutoff_action_repeat),
        "cutoff_log_dir": str(ROOT / "artifacts" / "cutoff_logs"),
    }
    WorkerCls = _make_worker_cls()
    if mpc_cfg and mpc_cfg["device"].startswith("cuda"):
        gpu_frac = 1.0 / n_workers            # 워커들이 한 GPU 를 분할 점유
        workers = [WorkerCls.options(num_gpus=gpu_frac).remote({**spec, "wid": i})
                   for i in range(n_workers)]
    else:
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
            strong, weak = ("ownship", "target(exe)") if w > l else ("target(exe)", "ownship")
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
