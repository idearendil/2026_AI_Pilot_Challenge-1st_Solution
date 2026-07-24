# -*- coding: utf-8 -*-
"""REDQ 병렬 driver (Phase 1): Ray CPU 워커 수집 + **GPU 학습을 별도 Ray 액터로 분리**.

구조:
  driver(=이 프로세스, **CUDA 미사용**) ── 오케스트레이션만
    ├─ N × RolloutWorker (Ray, CPU): env rollout → raw transition
    └─ 1 × LearnerActor  (Ray, num_gpus=1, CUDA): RedqLearner + 중앙 replay 소유

  매 사이클:
    workers.collect ─(raw transitions + opp_slot)→ driver
    driver: slot→gen 매핑 → learner.add_transitions → learner.update(UTD 배)
    learner ─(metrics + actor weights)→ driver ─(weights)→ workers

**왜 이렇게?** Windows 에서 CUDA context 를 Ray driver 프로세스에 두고 Ray 워커를 함께
돌리면 몇 사이클 후 네이티브 access violation 이 난다(Ray 백그라운드 스레드 ↔ CUDA 충돌).
그래서 CUDA 는 driver 가 절대 만지지 않고 **전용 Ray GPU 액터 프로세스**에 격리한다.
driver 는 CPU 오케스트레이터로만 남는다. replay 도 GPU 업데이트와 같은 프로세스에 둬야
UTD 배 업데이트가 IPC 없이 로컬에서 돈다(사이클당 raw transition 만 액터로 전송).

opponent pool 메타데이터(gen/ema/state/rms)는 **driver 가 소유**한다. replay 의 gen
태그와 일치시켜야 하므로(snapshot evict 시 그 gen 의 transition 을 purge). slot 규약은 PPO
와 동일: slot0=BT(gen=-1, 영구 고정), slot1.. = snapshot(gen 0,1,2… 오래된→최신).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.bt_rule import ENV_KEY as BT_RULE_ENV_KEY
from claude_code.normalizers import RunningMeanStd
from claude_code.redq.workers import make_redq_worker_cls

BT_GEN = -1   # baseline BT 의 gen (영구 고정, replay purge 대상 아님)


def make_learner_actor_cls():
    """Ray import 시점에 GPU 학습 액터 클래스를 정의.

    RedqLearner(CUDA) + OpponentTaggedReplay 를 소유한다. driver 는 CUDA 를 안 만지고
    이 액터에게만 GPU 업데이트를 위임한다. replay 가 액터 안에 있어 UTD 배 업데이트가
    로컬(무 IPC)로 돈다.
    """
    import ray

    @ray.remote
    class LearnerActor:
        def __init__(self, cfg, obs_dim, act_dim):
            for _p in (str(ROOT), str(ROOT / "src")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            # CUDA allocator: Ray 액터 프로세스라 driver 와 격리돼 있지만, 안전하게 async
            # allocator 를 쓴다(단독 CUDA 프로세스라 원래 문제는 없다).
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
            from claude_code.redq.sac import RedqLearner
            from claude_code.redq.replay import OpponentTaggedReplay

            self.cfg = cfg
            self.learner = RedqLearner(cfg, obs_dim)
            self.replay = OpponentTaggedReplay(cfg.buffer_size, obs_dim, act_dim, seed=cfg.seed)
            self.grad_steps = 0

        def device_str(self):
            return str(self.learner.device)

        def add_transitions(self, obs, next_obs, actions, rewards, terminated, opp_gen, global_step):
            self.replay.add_batch(obs, next_obs, actions, rewards, terminated, opp_gen, global_step)

        def buffer_len(self):
            return len(self.replay)

        def age_stats(self, global_step):
            return self.replay.age_stats(global_step)

        def purge_opponent(self, gen):
            return self.replay.purge_opponent(gen)

        def set_obs_stats(self, mean, var):
            self.learner.set_obs_stats(mean, var)

        def update(self, n_env_steps, global_step):
            """UTD 배 gradient update. (metrics, grad_steps) 반환."""
            if (len(self.replay) < self.cfg.min_buffer_for_update
                    or global_step < self.cfg.warmup_steps):
                return {}, self.grad_steps
            n_updates = int(self.cfg.utd_ratio) * int(n_env_steps)
            acc: dict = {}
            for _ in range(n_updates):
                batch = self.replay.sample(self.cfg.batch_size)
                m = self.learner.update(batch)
                for k, v in m.items():
                    acc.setdefault(k, []).append(v)
                self.grad_steps += 1
            return {k: float(np.mean(v)) for k, v in acc.items()}, self.grad_steps

        def get_actor_state(self):
            """CPU numpy actor state_dict (rollout worker broadcast + 번들 저장용)."""
            return {k: v.numpy() for k, v in self.learner.actor_state_dict_cpu().items()}

        def alpha(self):
            return self.learner.alpha

        def save_ckpt(self, path):
            """learner 전체 상태 + grad_steps 를 원자적으로 저장(replay 는 제외)."""
            import torch as _torch
            tmp = str(path) + ".tmp"
            _torch.save({"learner": self.learner.full_state(),
                         "grad_steps": int(self.grad_steps)}, tmp)
            os.replace(tmp, str(path))

        def load_ckpt(self, path):
            import torch as _torch
            ck = _torch.load(str(path), map_location="cpu", weights_only=False)
            self.learner.load_full_state(ck["learner"])
            self.grad_steps = int(ck.get("grad_steps", 0))
            return self.grad_steps

    return LearnerActor


class RedqParallelTrainer:
    def __init__(self, cfg, env_kwargs, obs_dim, act_dim,
                 bt_dll: str = "", bt_rule: str = ""):
        import ray
        self.cfg = cfg
        self.num_workers = max(1, int(cfg.num_workers))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.global_step = 0
        self.grad_steps = 0

        np.random.seed(cfg.seed)

        self._actor_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, num_bins=cfg.num_bins,
                                  hidden=tuple(cfg.actor_hidden), activation=cfg.actor_activation)
        cfg_dict = dict(normalize_obs=cfg.normalize_obs, reconstruct_state=cfg.reconstruct_state,
                        warmup_steps=cfg.warmup_steps)

        if not ray.is_initialized():
            pythonpath = os.pathsep.join(
                [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
            env_vars = {"PYTHONPATH": pythonpath}
            rule = os.environ.get(BT_RULE_ENV_KEY, "")
            if rule:
                env_vars[BT_RULE_ENV_KEY] = rule
            want_gpu = 1 if (cfg.device == "cuda") else 0
            _log2drv = os.environ.get("REDQ_RAY_LOG_TO_DRIVER", "0") == "1"
            ray.init(num_cpus=self.num_workers + 1, num_gpus=want_gpu,
                     include_dashboard=False, ignore_reinit_error=True, log_to_driver=_log2drv,
                     runtime_env={"env_vars": env_vars})

        # GPU 학습 액터(전용 프로세스, CUDA 격리). driver 프로세스는 CUDA 를 만지지 않는다.
        LearnerCls = make_learner_actor_cls()
        num_gpus = 1 if (cfg.device == "cuda") else 0
        self.learner = LearnerCls.options(num_gpus=num_gpus).remote(cfg, obs_dim, act_dim)
        self._device_str = ray.get(self.learner.device_str.remote())

        WorkerCls = make_redq_worker_cls()
        self.workers = [
            WorkerCls.remote(i, env_kwargs, self._actor_kwargs, cfg_dict,
                             cfg.self_play, cfg.seed + 1 + i)
            for i in range(self.num_workers)
        ]
        print(f"[REDQ] Ray: workers={self.num_workers}(CPU) + learner(1, {self._device_str}) "
              f"- CUDA 는 학습 액터 프로세스에 격리")

        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if cfg.normalize_obs else None
        self.pool: list = []
        self._next_gen = 0
        self._bt_dll = bt_dll
        self._bt_rule = bt_rule
        self._bt_slots = 1 if bt_dll else 0

    @property
    def device_str(self):
        return self._device_str

    # ── pool 초기화 ──────────────────────────────────────────────────────────
    def install_pool(self, pool_max, seed):
        import ray
        state, mean, var, count = self._current_actor_state()
        self.pool = []
        if self._bt_dll:
            self.pool.append({"kind": "bt", "gen": BT_GEN, "ema": 0.5,
                              "state": None, "rms": None,
                              "dll": self._bt_dll, "rule": self._bt_rule})
        gen0 = self._alloc_gen()
        self.pool.append({"kind": "net", "gen": gen0, "ema": 0.5,
                          "state": state, "rms": {"mean": mean, "var": var, "count": count}})
        n = len(self.pool)
        weights = [1.0 / n] * n
        ref = ray.put(state)
        ray.get([w.pool_init.remote(ref, mean, var, count, weights, int(pool_max),
                                    self.cfg.seed + 101 + i, self._bt_dll, self._bt_rule)
                 for i, w in enumerate(self.workers)])
        self._pool_max = int(pool_max)

    def _alloc_gen(self) -> int:
        g = self._next_gen
        self._next_gen += 1
        return g

    def current_gen_list(self) -> list:
        return [e["gen"] for e in self.pool]

    def add_current_to_pool(self):
        """현재 actor 를 새 snapshot 후보로 pool 에 추가. 초과 시 oldest snapshot evict +
        그 gen 의 replay transition purge. (new_gen, evicted_gen|None, purged_count) 반환."""
        import ray
        state, mean, var, count = self._current_actor_state()
        new_gen = self._alloc_gen()
        self.pool.append({"kind": "net", "gen": new_gen, "ema": 0.5,
                          "state": state, "rms": {"mean": mean, "var": var, "count": count}})
        evicted_gen = None
        purged = 0
        if len(self.pool) > self._pool_max:
            ev = self.pool.pop(self._bt_slots)   # 가장 오래된 snapshot (BT 보존)
            evicted_gen = int(ev["gen"])
            if self.cfg.purge_evicted_opponents:
                purged = int(ray.get(self.learner.purge_opponent.remote(evicted_gen)))
        ref = ray.put(state)
        ray.get([w.pool_add.remote(ref, mean, var, count) for w in self.workers])
        return new_gen, evicted_gen, purged

    def set_pool_weights(self, weights):
        import ray
        ray.get([w.pool_set_weights.remote(list(weights)) for w in self.workers])

    def _current_actor_state(self):
        import ray
        state = ray.get(self.learner.get_actor_state.remote())
        if self.obs_rms is not None:
            return state, self.obs_rms.mean.copy(), self.obs_rms.var.copy(), float(self.obs_rms.count)
        return state, np.zeros(self.obs_dim), np.ones(self.obs_dim), 1.0

    # ── broadcast + collect + 중앙 replay(액터) 삽입 ───────────────────────────
    def _broadcast(self):
        import ray
        state = ray.get(self.learner.get_actor_state.remote())
        ref = ray.put(state)
        calls = [w.set_weights.remote(ref) for w in self.workers]
        calls += [w.set_global_step.remote(self.global_step) for w in self.workers]
        if self.obs_rms is not None:
            calls += [w.set_obs_rms.remote(self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count)
                      for w in self.workers]
        ray.get(calls)

    def collect_and_store(self, total_steps):
        import ray
        self._broadcast()
        per = max(1, int(total_steps) // self.num_workers)
        gen_list = np.asarray(self.current_gen_list(), dtype=np.int64)
        results = ray.get([w.collect.remote(per) for w in self.workers])

        add_futs = []
        for r in results:
            slots = np.clip(np.asarray(r["opp_slot"], dtype=np.int64), 0, len(gen_list) - 1)
            opp_gen = gen_list[slots]
            add_futs.append(self.learner.add_transitions.remote(
                r["obs"], r["next_obs"], r["actions"], r["rewards"],
                r["terminated"], opp_gen, self.global_step))
            self.global_step += len(r["rewards"])
            if self.obs_rms is not None and r["rms_mean"] is not None:
                self.obs_rms._update_from_moments(
                    np.asarray(r["rms_mean"]), np.asarray(r["rms_var"]), r["rms_count"])
        ray.get(add_futs)

        agg = {
            "ep_returns": [x for r in results for x in r["ep_returns"]],
            "ep_lengths": [x for r in results for x in r["ep_lengths"]],
            "ep_components": [x for r in results for x in r["ep_components"]],
            "ep_outcomes": [x for r in results for x in r["ep_outcomes"]],
            "ep_opp_indices": [x for r in results for x in r["ep_opp_indices"]],
            "ep_end_conditions": [x for r in results for x in r["ep_end_conditions"]],
        }
        return agg

    # ── UTD 배 gradient update (액터에서 실행) ─────────────────────────────────
    def update(self, n_env_steps):
        import ray
        if self.obs_rms is not None:
            ray.get(self.learner.set_obs_stats.remote(self.obs_rms.mean, self.obs_rms.var))
        metrics, grad_steps = ray.get(
            self.learner.update.remote(n_env_steps, self.global_step))
        self.grad_steps = int(grad_steps)
        return metrics

    def buffer_age_stats(self):
        import ray
        return ray.get(self.learner.age_stats.remote(self.global_step))

    def actor_state_for_bundle(self):
        """번들 저장용 (actor state_dict numpy, obs_rms dict)."""
        import ray
        state = ray.get(self.learner.get_actor_state.remote())
        rms = self.obs_rms.state_dict() if self.obs_rms is not None else None
        return state, rms

    # ── crash 자동 재시작용 체크포인트 (replay 는 제외; 네트워크/pool/카운터만) ──
    def save_checkpoint(self, ckpt_dir, cycle):
        """learner(액터) + driver 메타(pool/obs_rms/카운터)를 원자적으로 저장."""
        import ray
        import torch
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ray.get(self.learner.save_ckpt.remote(str(ckpt_dir / "learner.pt")))
        meta = {
            "pool": self.pool,                     # snapshot state 는 numpy dict (pickle 가능)
            "next_gen": self._next_gen,
            "pool_max": self._pool_max,
            "bt_dll": self._bt_dll, "bt_rule": self._bt_rule, "bt_slots": self._bt_slots,
            "global_step": self.global_step, "grad_steps": self.grad_steps,
            "cycle": int(cycle),
            "obs_rms": (self.obs_rms.state_dict() if self.obs_rms is not None else None),
        }
        tmp = str(ckpt_dir / "driver.pt") + ".tmp"
        torch.save(meta, tmp)
        os.replace(tmp, str(ckpt_dir / "driver.pt"))

    def resume_from(self, ckpt_dir):
        """저장된 체크포인트에서 learner + pool + obs_rms + 카운터 복원. cycle 반환.

        replay 는 저장하지 않았으므로 비어 있는 채로 시작한다(로드한 actor 로 다시 채움)."""
        import ray
        import torch
        ckpt_dir = Path(ckpt_dir)
        ray.get(self.learner.load_ckpt.remote(str(ckpt_dir / "learner.pt")))
        meta = torch.load(str(ckpt_dir / "driver.pt"), map_location="cpu", weights_only=False)
        self.pool = meta["pool"]
        self._next_gen = int(meta["next_gen"])
        self._pool_max = int(meta["pool_max"])
        self._bt_dll = meta.get("bt_dll", self._bt_dll)
        self._bt_rule = meta.get("bt_rule", self._bt_rule)
        self._bt_slots = int(meta.get("bt_slots", self._bt_slots))
        self.global_step = int(meta["global_step"])
        self.grad_steps = int(meta["grad_steps"])
        if self.obs_rms is not None and meta.get("obs_rms") is not None:
            self.obs_rms = RunningMeanStd.from_state_dict(meta["obs_rms"])
        # 워커들에 pool 전체 복원(BT slot0 + snapshot 후보들).
        snaps = [e for e in self.pool if e["kind"] == "net"]
        states = [e["state"] for e in snaps]
        means = [e["rms"]["mean"] for e in snaps]
        vars_ = [e["rms"]["var"] for e in snaps]
        counts = [e["rms"]["count"] for e in snaps]
        weights = [1.0 / len(self.pool)] * len(self.pool)
        sref = ray.put(states)
        ray.get([w.pool_set_all.remote(sref, means, vars_, counts, weights, self._pool_max,
                                       self.cfg.seed + 101 + i, self._bt_dll, self._bt_rule)
                 for i, w in enumerate(self.workers)])
        return int(meta["cycle"])

    def close(self):
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


__all__ = ["RedqParallelTrainer", "BT_GEN"]
