# -*- coding: utf-8 -*-
"""unreal_bt_client.exe(외부 BT UDP 클라이언트)의 행동을 데이터셋으로 수집한다(행동복제용).

목적
----
exe 를 상대로 정해진 **시간** 동안 게임을 붙이면서, 매 RL-step 마다 **exe 관점의 claude164r
관측 + exe 가 낸 raw action([-1,1]^4)** 을 기록한다. 이 (obs, action) 쌍으로 나중에 exe 와
똑같이 행동하는 actor net 을 지도학습(behavioral cloning)해 self-play opponent pool 에
넣을 수 있다.

- 관측은 학습 파이프라인과 동일한 claude164r(184D) 로 기록(→ SelfPlayProvider 로 바로 사용).
  exe 관점: context.ownship_state=exe 기체, target_state=상대(플레이어).
- exe 를 상대하는 **플레이어(ownship)** 는 내 RL 번들과 gylee 에이전트를 게임별로 섞고,
  둘 다 **stochastic** 으로 굴려 데이터를 최대한 다양하게 만든다.
- exe 는 매 substep(6/RL-step) 재계산하지만, 기록은 RL-step 시작(substep-0) 1회/RL-step 만
  한다(정책 10Hz 결정 주기와 일치). advance→build_obs→push_action 순서는 MLPActionProvider
  와 동일(action history feature 일관).
- **시간 기준**: --duration-hours 만큼만 새 게임을 시작하고, deadline 이후엔 진행 중인 게임만
  마친다. 데이터는 워커가 주기적으로 **샤드(.npz)로 flush** 하므로 몇 시간을 돌려도 메모리가
  일정하게 유지된다. 최종 출력은 **샤드 디렉토리 + manifest.json**.

예시(4시간):
  python claude_code/collect_exe_dataset.py \
    --rl-bundle-dir artifacts/models/team01/basic --duration-hours 4 \
    --num-workers 6 --rl-fraction 0.5 --out claude_code/models/wm/exe_bc_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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

from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402

_DEF_GYLEE_SNAPSHOT = str(ROOT / "model2_gylee" / "model" / "iter_1415.pt")
_DEF_EXE = str(ROOT / "unreal_bt_client.exe")
_FLUSH_EVERY = 200_000          # 워커 버퍼가 이 sample 수를 넘으면 샤드로 flush(메모리 상한)


def _make_worker_cls():
    import ray

    @ray.remote
    class CollectWorker:
        """env + 기록 exe(target) + RL/gylee 플레이어(ownship) 를 만들고 deadline 까지 돈다."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch
            from GeoMathUtil import GeometryInfo

            from claude_code.env_utils import make_env
            from claude_code import my_observation as MO
            from claude_code.unreal_exe_provider import UnrealExeProvider
            from dogfight.ai.action_provider import ActionProvider

            torch.set_num_threads(1)
            self.spec = spec
            self.wid = int(spec["wid"])
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))
            self.shard_dir = Path(spec["shard_dir"])
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            self._shard_k = 0
            self._shards = []            # (path, count)

            self.env = make_env(overrides=spec["overrides"], reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"col{self.wid}")

            # ── 플레이어(ownship): RL 번들 + gylee 둘 다 준비(게임별 교체). 둘 다 stochastic. ──
            from claude_code.action_provider import MLPActionProvider
            from claude_code.evaluate import _ActionRepeatProvider
            self.rl_provider = None
            if spec["rl_bundle_dir"]:
                rl_inner = MLPActionProvider(bundle_dir=spec["rl_bundle_dir"], stochastic=True)
                self.rl_provider = _ActionRepeatProvider(rl_inner, self.step_ratio)
            self.gylee_provider = None
            if spec["gylee_snapshot"]:
                from model2_gylee import make_opponent_provider
                self.gylee_provider = make_opponent_provider(
                    snapshot_path=spec["gylee_snapshot"], step_ratio=self.step_ratio,
                    device="cpu", explore=True, verify_checksum=False)

            # ── 기록 exe(target) ──
            class RecordingExeProvider(ActionProvider):
                def __init__(rself, exe, step_ratio):
                    rself.exe = exe
                    rself.repeat = int(step_ratio)
                    rself.geo = GeometryInfo()
                    rself.recon = MO.StateReconstructor()
                    rself._count = 0
                    rself.current_src = 0          # 0=rl, 1=gylee (worker 가 게임마다 설정)
                    rself.obs_buf = []
                    rself.act_buf = []
                    rself.src_buf = []

                def reset(rself, context=None):
                    rself.exe.reset(context)
                    rself.recon.reset()
                    rself._count = 0

                def compute_action(rself, context):
                    result = rself.exe.compute_action(context)     # 매 substep 위임
                    own = context.ownship_state                    # exe 기체
                    opp = context.target_state                     # 상대(플레이어)
                    if (rself._count % rself.repeat == 0 and own is not None
                            and opp is not None):
                        own = np.asarray(own, dtype=np.float64)
                        opp = np.asarray(opp, dtype=np.float64)
                        rself.recon.advance(own, opp)              # RL-step 당 1회
                        obs = MO.build_observation(own, opp, rself.geo, None,
                                                   reconstructor=rself.recon)
                        cmd = np.asarray(result.action, dtype=np.float64)   # throttle∈[0,1]
                        raw = cmd.copy()
                        raw[3] = 2.0 * cmd[3] - 1.0                # throttle→[-1,1] (policy raw)
                        raw = np.clip(raw, -1.0, 1.0)
                        rself.obs_buf.append(np.asarray(obs, dtype=np.float32))
                        rself.act_buf.append(raw.astype(np.float32))
                        rself.src_buf.append(np.int8(rself.current_src))
                        rself.recon.push_action(raw)
                    rself._count += 1
                    return result

                def close(rself):
                    try:
                        rself.exe.close()
                    except Exception:
                        pass

            exe = UnrealExeProvider(
                exe_path=spec["exe_path"],
                port=int(spec["base_port"]) + self.wid,
                own_plane_id=1, enemy_plane_id=0,
                ownship_force_side=int(spec["ownship_force_side"]),
                target_force_side=int(spec["target_force_side"]),
                cwd=spec["root"], step_timeout_sec=float(spec["step_timeout_sec"]),
                quiet=True)
            self.recorder = RecordingExeProvider(exe, self.step_ratio)
            self.env._target_action_provider = self.recorder

        def _flush(self):
            r = self.recorder
            n = len(r.obs_buf)
            if n == 0:
                return
            path = self.shard_dir / f"shard_w{self.wid:02d}_{self._shard_k:04d}.npz"
            np.savez_compressed(path,
                                obs=np.stack(r.obs_buf).astype(np.float32),
                                act=np.stack(r.act_buf).astype(np.float32),
                                src=np.asarray(r.src_buf, dtype=np.int8))
            self._shards.append((str(path), int(n)))
            self._shard_k += 1
            r.obs_buf.clear(); r.act_buf.clear(); r.src_buf.clear()

        def play(self, jobs):
            import torch
            out = []
            zero = np.zeros(4, dtype=np.float32)
            for seed, swap, head, kind in jobs:
                torch.manual_seed(int(seed))
                if kind == "gylee" and self.gylee_provider is not None:
                    player = self.gylee_provider
                elif self.rl_provider is not None:
                    player, kind = self.rl_provider, "rl"
                elif self.gylee_provider is not None:
                    player, kind = self.gylee_provider, "gylee"
                else:
                    raise RuntimeError("플레이어 provider 가 하나도 없습니다.")
                self.env._ownship_action_provider = player
                self.recorder.current_src = 0 if kind == "rl" else 1
                self.env._apply_start_side(bool(swap), bool(head))

                n_before = len(self.recorder.obs_buf) + sum(c for _, c in self._shards)
                self.env.reset(seed=int(seed))
                player.reset()
                self.recorder.reset()
                term = trunc = False
                steps = 0
                info = {}
                while not (term or trunc):
                    _o, _r, term, trunc, info = self.env.step(zero)
                    steps += 1
                n_after = len(self.recorder.obs_buf) + sum(c for _, c in self._shards)
                out.append({"seed": int(seed), "kind": kind, "steps": steps,
                            "end": str(info.get("end_condition", "")),
                            "samples": int(n_after - n_before),
                            "exe_fail": int(getattr(self.recorder.exe, "_fail_count", 0))})
                if len(self.recorder.obs_buf) >= _FLUSH_EVERY:
                    self._flush()
            return out

        def finalize(self):
            self._flush()
            total = sum(c for _, c in self._shards)
            return {"shards": [p for p, _ in self._shards], "count": int(total)}

        def close(self):
            try:
                self.recorder.close()
            except Exception:
                pass
            try:
                self.env.close()
            except Exception:
                pass

    return CollectWorker


def parse_args():
    p = argparse.ArgumentParser(
        description="unreal_bt_client.exe 행동 데이터셋 수집(BC). 시간 기준·플레이어=RL+gylee 혼합·stochastic")
    p.add_argument("--rl-bundle-dir", default="artifacts/models/team01/basic",
                   help="플레이어로 쓸 내 RL 번들(stochastic). 빈 문자열이면 gylee 만 사용")
    p.add_argument("--gylee-snapshot", default=_DEF_GYLEE_SNAPSHOT,
                   help=f"플레이어로 쓸 gylee snapshot(stochastic). 기본 {Path(_DEF_GYLEE_SNAPSHOT).name}")
    p.add_argument("--rl-fraction", type=float, default=0.5,
                   help="플레이어를 RL 번들로 쓸 게임 비율(나머지는 gylee). 기본 0.5")
    p.add_argument("--duration-hours", type=float, default=4.0,
                   help="이 시간만큼 수집(기본 4.0). deadline 이후엔 진행 중인 게임만 마친다.")
    p.add_argument("--games", type=int, default=0,
                   help="추가 게임 수 상한(0=상한 없음, 시간 기준만)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="병렬 worker 수(0=물리 코어 수). worker 마다 exe 1개")
    p.add_argument("--exe-path", default=_DEF_EXE)
    p.add_argument("--base-port", type=int, default=9200)
    p.add_argument("--ownship-force-side", type=int, default=1)
    p.add_argument("--target-force-side", type=int, default=2)
    p.add_argument("--step-timeout-sec", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--fixed-side", action="store_true")
    p.add_argument("--max-engage-time", type=float,
                   default=float(STANDARD_ENV_CONFIG["max_engage_time"]))
    p.add_argument("--episode-step-limit", type=int,
                   default=int(STANDARD_ENV_CONFIG["episode_step_limit"]))
    p.add_argument("--out", default=str(ROOT / "claude_code/models/wm/exe_bc_dataset"),
                   help="데이터셋 디렉토리(샤드 .npz + manifest.json 저장)")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.exe_path).exists():
        raise FileNotFoundError(f"exe 를 찾을 수 없습니다: {args.exe_path}")
    if not args.rl_bundle_dir and not args.gylee_snapshot:
        raise ValueError("--rl-bundle-dir 과 --gylee-snapshot 중 최소 하나는 있어야 합니다.")
    if args.rl_bundle_dir and not (ROOT / args.rl_bundle_dir).exists() \
            and not Path(args.rl_bundle_dir).exists():
        raise FileNotFoundError(f"RL 번들을 찾을 수 없습니다: {args.rl_bundle_dir}")
    if args.gylee_snapshot and not Path(args.gylee_snapshot).is_file():
        raise FileNotFoundError(f"gylee snapshot 을 찾을 수 없습니다: {args.gylee_snapshot}")

    obs_module = "claude_code.my_observation"
    if args.rl_bundle_dir:
        from claude_code.model import load_bundle
        _, meta = load_bundle(args.rl_bundle_dir, device="cpu")
        obs_module = meta.get("observation_module", "") or "claude_code.my_observation"

    overrides = {
        "target_mode": "fixed",
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "randomize_start_side": False,
    }

    games_cap = int(args.games) if args.games and args.games > 0 else 0
    duration = float(args.duration_hours) * 3600.0
    n_workers = args.num_workers if args.num_workers > 0 else physical_cpu_count()
    if games_cap:
        n_workers = min(int(n_workers), games_cap)
    n_workers = max(1, int(n_workers))

    master_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(master_seed)
    have_rl = bool(args.rl_bundle_dir)
    have_gy = bool(args.gylee_snapshot)
    rl_frac = float(np.clip(args.rl_fraction, 0.0, 1.0))

    def _make_job(i):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        swap = 0 if args.fixed_side else i % 2
        head = 0 if args.fixed_side else (i // 2) % 2
        if have_rl and have_gy:
            kind = "rl" if rng.random() < rl_frac else "gylee"
        else:
            kind = "rl" if have_rl else "gylee"
        return (seed, swap, head, kind)

    out_dir = Path(args.out)
    shard_dir = out_dir / "shards"
    print(f"[collect] exe={Path(args.exe_path).name}  duration={args.duration_hours}h"
          f"{'' if not games_cap else f' (게임 상한 {games_cap})'}  workers={n_workers}  "
          f"master_seed={master_seed}")
    print(f"[collect] 플레이어: RL={args.rl_bundle_dir or '(off)'}  gylee="
          f"{Path(args.gylee_snapshot).name if have_gy else '(off)'}  "
          f"rl_fraction={rl_frac}  (둘 다 stochastic)")
    print(f"[collect] obs={obs_module}(184D, exe 관점)  max_engage={args.max_engage_time}s  "
          f"out={out_dir}")

    import ray
    if not ray.is_initialized():
        pythonpath = os.pathsep.join([str(ROOT), str(ROOT / "src"),
                                      os.environ.get("PYTHONPATH", "")])
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "rl_bundle_dir": args.rl_bundle_dir, "gylee_snapshot": args.gylee_snapshot,
        "exe_path": str(args.exe_path), "base_port": int(args.base_port),
        "ownship_force_side": int(args.ownship_force_side),
        "target_force_side": int(args.target_force_side),
        "step_timeout_sec": float(args.step_timeout_sec),
        "shard_dir": str(shard_dir),
    }
    WorkerCls = _make_worker_cls()
    workers = [WorkerCls.remote({**spec, "wid": i}) for i in range(n_workers)]

    t0 = time.time()
    deadline = t0 + duration
    results = []
    n_dispatched = 0
    pending = {}

    def _can_dispatch():
        if time.time() >= deadline:
            return False
        if games_cap and n_dispatched >= games_cap:
            return False
        return True

    for wi, wk in enumerate(workers):
        if _can_dispatch():
            pending[wk.play.remote([_make_job(n_dispatched)])] = wi
            n_dispatched += 1

    last_print = 0.0
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        for ref in done:
            wi = pending.pop(ref)
            results.extend(ray.get(ref))
            if _can_dispatch():
                pending[workers[wi].play.remote([_make_job(n_dispatched)])] = wi
                n_dispatched += 1
        now = time.time()
        if now - last_print > 30.0 or not pending:
            remain = max(0.0, deadline - now)
            samp = sum(r["samples"] for r in results)
            print(f"  ... {len(results)}판 완료, sample {samp:,}  경과 {(now - t0)/60:.1f}분 / "
                  f"남음 {remain/60:.1f}분 (진행중 {len(pending)}판)", flush=True)
            last_print = now

    # 남은 버퍼 flush + 샤드 목록 수집.
    finals = ray.get([wk.finalize.remote() for wk in workers])
    ray.get([wk.close.remote() for wk in workers])
    ray.shutdown()

    all_shards = []
    total = 0
    for f in finals:
        for p in f["shards"]:
            all_shards.append(os.path.relpath(p, out_dir))
        total += f["count"]

    # src 분포 집계(샤드 헤더만 빠르게 읽어서).
    n_rl = n_gy = 0
    for rel in all_shards:
        z = np.load(out_dir / rel)
        s = z["src"]
        n_rl += int((s == 0).sum()); n_gy += int((s == 1).sum())

    manifest = {
        "obs_mode": obs_module, "obs_dim": 184, "act_dim": 4,
        "total_samples": int(total), "n_rl": n_rl, "n_gylee": n_gy,
        "n_games": len(results), "shards": sorted(all_shards),
        "duration_hours": args.duration_hours, "master_seed": int(master_seed),
        "rl_bundle_dir": args.rl_bundle_dir, "gylee_snapshot": args.gylee_snapshot,
        "action_space": "raw [-1,1]^4 (roll,pitch,rudder,throttle_remapped)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"\n[collect] 저장 완료: {out_dir}  (shard {len(all_shards)}개, "
          f"총 sample {total:,}, rl {n_rl:,} / gylee {n_gy:,})")
    print(f"[collect] manifest: {out_dir / 'manifest.json'}")

    _report(results, time.time() - t0)


def _report(results, elapsed):
    n = len(results)
    ends = Counter(r["end"] for r in results)
    by_kind = Counter(r["kind"] for r in results)
    exe_fail = sum(r["exe_fail"] for r in results)
    steps = np.mean([r["steps"] for r in results]) if n else 0.0
    print(f"\n{'=' * 60}")
    print(f"수집 요약: {n}판, {elapsed/60:.1f}분, 평균 {steps:.0f} step/판")
    print(f"  플레이어 배분: {dict(by_kind)}")
    print(f"  end_condition:")
    for k, c in ends.most_common():
        print(f"    {c:5d}  {k}")
    if exe_fail:
        print(f"  ⚠ exe CMD 타임아웃 총 {exe_fail}회 — --step-timeout-sec 를 늘려보세요")


if __name__ == "__main__":
    main()
