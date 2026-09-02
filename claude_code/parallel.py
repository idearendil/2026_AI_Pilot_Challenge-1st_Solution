"""Ray 기반 병렬 rollout 데이터 수집.

큰 모델 + GPU 학습을 대비해, CPU-bound 한 환경 stepping 을 여러 프로세스(Ray actor)로
병렬화한다. 각 worker 는 자신의 env(self-play 포함) + 로컬 model 복사본을 갖고 rollout
을 모은다. 매 iteration:
  driver --(policy weights + obs_rms)--> workers --(rollout batch)--> driver --update(GPU 가능)

worker 수는 **물리 CPU 코어 수** 기준(논리 코어 아님)으로 정한다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.bt_rule import ENV_KEY as BT_RULE_ENV_KEY
from claude_code.model import make_actor_critic, discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd
from claude_code.ppo import (PPOConfig, PPOTrainer, IterationStats, compute_gae, OBS_CLIP,
                             rollout_outcome, _outcome_counts, _outcome_counts_by_opp,
                             _count_altitude_terms)


def physical_cpu_count() -> int:
    """물리 CPU 코어 수 (논리 아님). psutil 없으면 OS 별 조회 → 마지막엔 logical//2."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | "
                 "Measure-Object -Property NumberOfCores -Sum).Sum"],
                text=True, timeout=15)
            n = int(out.strip())
            if n > 0:
                return n
        elif sys.platform.startswith("linux"):
            ids = set()
            with open("/proc/cpuinfo") as fh:
                phys = core = None
                for line in fh:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        if phys is not None:
                            ids.add((phys, core))
            if ids:
                return len(ids)
    except Exception:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


# ── Ray worker ────────────────────────────────────────────────────────────────

def _make_worker_cls():
    """Ray 가 import 된 시점에 actor 클래스를 정의(미설치 환경에서 import 안전)."""
    import ray

    @ray.remote
    class RolloutWorker:
        def __init__(self, worker_id, env_kwargs, model_kwargs, cfg_dict, self_play, seed):
            for _p in (str(ROOT), str(ROOT / "src")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            os.chdir(str(ROOT))   # 상대 경로 자산(aircraft/engine/xml) 로드 안전
            import torch as _torch
            _torch.manual_seed(seed)
            np.random.seed(seed)

            from claude_code.env_utils import make_env, STANDARD_ENV_CONFIG
            self.env = make_env(runner_index=f"w{worker_id}", **env_kwargs)
            self.model = make_actor_critic(**model_kwargs)
            self.model.eval()
            self._model_kwargs = dict(model_kwargs)   # 평가용 상대 network 재생성에 사용
            self._opp_model = None

            self.gamma = cfg_dict["gamma"]
            self.gae_lambda = cfg_dict["gae_lambda"]
            # reward 단계 스케줄용 shaping base(있으면). my_reward 만 이 키를 가진다.
            self._base_shaping = None
            try:
                _rc = self.env.config.get("reward") if hasattr(self.env, "config") else None
                if isinstance(_rc, dict) and "shaping_reward_scale" in _rc:
                    self._base_shaping = float(_rc["shaping_reward_scale"])
            except Exception:
                self._base_shaping = None
            self.normalize_obs = cfg_dict["normalize_obs"]
            self.reconstruct = cfg_dict["reconstruct_state"]
            self.obs_dim = model_kwargs["obs_dim"]
            self.act_dim = model_kwargs["act_dim"]

            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,)) if self.normalize_obs else None

            if self.reconstruct:
                from claude_code.my_observation import (reset_reconstructor,
                                                        advance_reconstructor,
                                                        push_action_reconstructor)
                self._reset_recon = reset_reconstructor
                self._advance_recon = advance_reconstructor
                self._push_action = push_action_reconstructor
                self._reset_recon()
            else:
                self._reset_recon = self._advance_recon = self._push_action = None

            if self_play:
                from claude_code.self_play import SelfPlayProvider
                sr = int(STANDARD_ENV_CONFIG["step_ratio"])
                self.env._target_action_provider = SelfPlayProvider(
                    self.model, self.obs_rms, self.env._observation_fn,
                    self.env._observation_mode, sr, "cpu", explore=True)

            obs, _ = self.env.reset(seed=seed)
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0

        def set_weights(self, state_dict):
            self.model.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})

        def set_obs_rms(self, mean, var, count):
            if self.obs_rms is not None:
                self.obs_rms.mean = np.asarray(mean, dtype=np.float64)
                self.obs_rms.var = np.asarray(var, dtype=np.float64)
                self.obs_rms.count = float(count)

        def set_schedule(self, gamma, own_damage_weight, shaping_mult):
            """드라이버가 매 iteration 브로드캐스트하는 단계 스케줄을 적용한다.

            gamma 는 이 워커의 GAE 계산(self.gamma)에 즉시 반영된다. reward 파라미터는
            my_reward(=shaping_reward_scale 키 보유) env 에만 in-place 적용한다.
            shaping_reward_scale = base_shaping × shaping_mult, own_damage_weight 는 절대값.
            """
            self.gamma = float(gamma)
            try:
                rc = self.env.config.get("reward") if hasattr(self.env, "config") else None
            except Exception:
                rc = None
            if isinstance(rc, dict) and "shaping_reward_scale" in rc:
                if self._base_shaping is None:
                    self._base_shaping = float(rc["shaping_reward_scale"])
                rc["shaping_reward_scale"] = self._base_shaping * float(shaping_mult)
                rc["own_damage_weight"] = float(own_damage_weight)

        def set_frozen_opponent(self, state_dict, rms_mean, rms_var, rms_count):
            """self-play 상대를 '초기 actor net 고정'(별도 frozen 모델)으로 교체한다.

            기본 self-play 는 상대가 self.model(매 iter 갱신되는 학습 agent)을 참조한다.
            여기서는 학습 시작 시점의 weights 로 별도 frozen 모델을 만들어 상대 provider 가
            그걸 쓰게 한다(학습이 진행돼도 상대는 고정). obs_rms 도 그 시점 값으로 고정.
            """
            from claude_code.env_utils import STANDARD_ENV_CONFIG
            from claude_code.self_play import SelfPlayProvider
            self._frozen_opp_model = make_actor_critic(**self._model_kwargs)
            self._frozen_opp_model.load_state_dict(
                {k: torch.as_tensor(v) for k, v in state_dict.items()})
            self._frozen_opp_model.eval()
            frozen_rms = None
            if self.obs_rms is not None:
                frozen_rms = RunningMeanStd(shape=(self.obs_dim,))
                frozen_rms.mean = np.asarray(rms_mean, dtype=np.float64)
                frozen_rms.var = np.asarray(rms_var, dtype=np.float64)
                frozen_rms.count = float(rms_count)
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            self.env._target_action_provider = SelfPlayProvider(
                self._frozen_opp_model, frozen_rms, self.env._observation_fn,
                self.env._observation_mode, sr, "cpu", explore=True)

        def _build_opp_provider(self, state_dict, mean, var, count, explore=True):
            """broadcast 받은 state+rms 로 frozen opponent SelfPlayProvider 생성(worker)."""
            from claude_code.env_utils import STANDARD_ENV_CONFIG
            from claude_code.self_play import SelfPlayProvider
            m = make_actor_critic(**self._model_kwargs)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in state_dict.items()})
            m.eval()
            rms = None
            if self.obs_rms is not None:
                rms = RunningMeanStd(shape=(self.obs_dim,))
                rms.mean = np.asarray(mean, dtype=np.float64)
                rms.var = np.asarray(var, dtype=np.float64)
                rms.count = float(count)
            sr = int(STANDARD_ENV_CONFIG["step_ratio"])
            return SelfPlayProvider(
                m, rms, self.env._observation_fn, self.env._observation_mode,
                sr, "cpu", explore=explore)

        def _get_bt_provider(self, bt_dll, bt_rule):
            """baseline BT 상대 provider(worker 당 1개, 재사용). DLL 은 1회만 로드."""
            if getattr(self, "_bt_provider", None) is None:
                from claude_code.self_play import make_bt_provider
                self._bt_provider = make_bt_provider(bt_dll, bt_rule)
            return self._bt_provider

        def _get_mpc_provider(self, mpc_root, mpc_config):
            """MPC 상대 provider(worker 당 1개, 재사용). native predictor DLL 은 1회만 로드.

            BT 와 달리 프로세스 제약이 없어 모든 워커가 동일 MPC 를 로컬 pool 에 갖는다.
            """
            if getattr(self, "_mpc_provider", None) is None:
                from claude_code.self_play import make_mpc_provider
                self._mpc_provider = make_mpc_provider(mpc_root, mpc_config)
            return self._mpc_provider

        def _get_cutoff_provider(self, exe_path, port, force_own, force_tgt, timeout):
            """컷오프 exe 상대 provider(worker 당 1개, 재사용). exe 를 1회만 띄운다.

            UnrealExeProvider(내가 쓰던 provider)로 unreal_bt_client.exe 를 포트 port 로 구동한다
            (팀원 CutoffUDPProvider 아님). context.ownship_state=상대기(exe), target_state=학습기.
            """
            if getattr(self, "_cutoff_provider", None) is None:
                from claude_code.unreal_exe_provider import UnrealExeProvider
                self._cutoff_provider = UnrealExeProvider(
                    exe_path=exe_path, port=int(port),
                    own_plane_id=1, enemy_plane_id=0,
                    ownship_force_side=int(force_own), target_force_side=int(force_tgt),
                    cwd=str(ROOT), step_timeout_sec=float(timeout), quiet=True)
            return self._cutoff_provider

        def pool_init(self, state_dict, mean, var, count, weights, pool_max, seed,
                      bt_dll="", bt_rule="", bt_index=0, n_bt=0,
                      mpc_root="", mpc_config="", n_mpc=0,
                      exp_states=(), exp_means=(), exp_vars=(), exp_counts=(), n_exp=0,
                      cutoff_exe="", cutoff_port=0, cutoff_force_own=1,
                      cutoff_force_tgt=2, cutoff_timeout=0.5, n_cut=0):
            """opponent pool 초기화. target provider 를 pool 로 교체.

            이 워커의 **로컬** pool = [배정된 BT 1개(있으면)] + [MPC(있으면, 모든 워커 공통)]
            + [CUTOFF exe(있으면, 워커마다 1개)] + [exploiter n_exp개(모든 워커 공통, 고정)] +
            [snapshot...]. bt_index = 이 BT 의 **글로벌** 슬롯(0..n_bt-1), n_bt = 전체 BT 수,
            n_mpc = MPC 수(0/1, 글로벌 슬롯 n_bt), n_cut = 컷오프 수(0/1, 글로벌 슬롯 n_bt+n_mpc),
            n_exp = exploiter 수(글로벌 슬롯 n_bt+n_mpc+n_cut..). 로컬 슬롯을 글로벌 슬롯으로
            매핑해(collect 참고) driver 가 상대별 EMA 를 분리 집계한다. BT·MPC·CUTOFF·exploiter
            슬롯은 고정(절대 evict 안 됨).
            """
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            self._bt_index = int(bt_index)
            self._n_bt = int(n_bt)
            self._n_mpc = int(n_mpc)
            self._n_cut = int(n_cut)
            self._n_exp = int(n_exp)
            provs = []
            self._bt_slots = 0
            if bt_dll:
                provs.append(self._get_bt_provider(bt_dll, bt_rule))
                self._bt_slots = 1
            self._mpc_slots = 0
            if mpc_root and n_mpc:
                provs.append(self._get_mpc_provider(mpc_root, mpc_config))
                self._mpc_slots = 1
            self._cut_slots = 0
            if cutoff_exe and n_cut:
                provs.append(self._get_cutoff_provider(
                    cutoff_exe, cutoff_port, cutoff_force_own, cutoff_force_tgt, cutoff_timeout))
                self._cut_slots = 1
            self._exp_slots = 0
            for st, mn, vr, ct in zip(exp_states, exp_means, exp_vars, exp_counts):
                provs.append(self._build_opp_provider(st, mn, vr, ct))
                self._exp_slots += 1
            provs.append(self._build_opp_provider(state_dict, mean, var, count))
            self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        def pool_add(self, state_dict, mean, var, count):
            """현재 정책 frozen copy 를 pool 에 추가(초과 시 oldest **snapshot** 제거) + env 리셋.

            고정 슬롯(BT _bt_slots + MPC _mpc_slots + exploiter _exp_slots) 다음 index 부터
            제거하므로 BT·MPC·exploiter 는 유지된다.
            """
            prov = self._build_opp_provider(state_dict, mean, var, count)
            providers = list(self._pool_provider.providers)
            providers.append(prov)
            fixed_slots = (int(getattr(self, "_bt_slots", 0)) + int(getattr(self, "_mpc_slots", 0))
                           + int(getattr(self, "_cut_slots", 0)) + int(getattr(self, "_exp_slots", 0)))
            if len(providers) > self._pool_max:
                providers.pop(fixed_slots)
            self._pool_provider.set_pool(providers)
            if self._reset_recon is not None:
                self._reset_recon()
            obs, _ = self.env.reset()
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0

        def pool_add_exploiter(self, state_dict, mean, var, count):
            """exploiter frozen copy 를 pool 의 **고정 exploiter 영역**(MPC 다음·snapshot 앞)에
            추가한다(절대 evict 안 됨). _exp_slots·_n_exp·_pool_max 를 1씩 늘리고, exploiter
            학습 중 frozen 상대로 바뀌었던 target provider 를 pool provider 로 되돌린다 + env 리셋.
            """
            prov = self._build_opp_provider(state_dict, mean, var, count)
            providers = list(self._pool_provider.providers)
            insert_at = int(getattr(self, "_bt_slots", 0)) + int(getattr(self, "_mpc_slots", 0)) \
                + int(getattr(self, "_cut_slots", 0)) + int(getattr(self, "_exp_slots", 0))
            providers.insert(insert_at, prov)   # 기존 exploiter 뒤·snapshot 앞
            self._pool_provider.set_pool(providers)
            self._exp_slots = int(getattr(self, "_exp_slots", 0)) + 1
            self._n_exp = int(getattr(self, "_n_exp", 0)) + 1
            self._pool_max = int(self._pool_max) + 1   # snapshot 정원 유지(고정 슬롯 1↑)
            self.env._target_action_provider = self._pool_provider   # frozen → pool 복귀
            if self._reset_recon is not None:
                self._reset_recon()
            obs, _ = self.env.reset()
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0

        def pool_set_weights(self, weights):
            """opponent 샘플링 가중치 갱신."""
            if getattr(self, "_pool_provider", None) is not None:
                self._pool_provider.set_weights(weights)

        def pool_set_all(self, state_dicts, means, vars_, counts, weights, pool_max, seed,
                         bt_dll="", bt_rule="", bt_index=0, n_bt=0,
                         mpc_root="", mpc_config="", n_mpc=0,
                         exp_states=(), exp_means=(), exp_vars=(), exp_counts=(), n_exp=0,
                         cutoff_exe="", cutoff_port=0, cutoff_force_own=1,
                         cutoff_force_tgt=2, cutoff_timeout=0.5, n_cut=0):
            """checkpoint 의 opponent pool 을 이 워커의 로컬 pool 로 복원.

            state_dicts 는 **snapshot 후보만** 오래된→최신. bt_dll 이 있으면 배정된 BT 1개를,
            mpc_root/n_mpc 가 있으면 MPC 1개를, cutoff_exe/n_cut 가 있으면 컷오프 exe 1개를,
            exp_states(n_exp개)가 있으면 exploiter 를 고정 슬롯으로 넣는다(순서: BT → MPC →
            CUTOFF → exploiter → snapshot). bt_index/n_bt/n_mpc/n_cut/n_exp 는 글로벌 매핑용.
            """
            from claude_code.self_play import PoolSelfPlayProvider
            self._pool_max = max(1, int(pool_max))
            self._bt_index = int(bt_index)
            self._n_bt = int(n_bt)
            self._n_mpc = int(n_mpc)
            self._n_cut = int(n_cut)
            self._n_exp = int(n_exp)
            provs = []
            self._bt_slots = 0
            if bt_dll:
                provs.append(self._get_bt_provider(bt_dll, bt_rule))
                self._bt_slots = 1
            self._mpc_slots = 0
            if mpc_root and n_mpc:
                provs.append(self._get_mpc_provider(mpc_root, mpc_config))
                self._mpc_slots = 1
            self._cut_slots = 0
            if cutoff_exe and n_cut:
                provs.append(self._get_cutoff_provider(
                    cutoff_exe, cutoff_port, cutoff_force_own, cutoff_force_tgt, cutoff_timeout))
                self._cut_slots = 1
            self._exp_slots = 0
            for st, mn, vr, ct in zip(exp_states, exp_means, exp_vars, exp_counts):
                provs.append(self._build_opp_provider(st, mn, vr, ct))
                self._exp_slots += 1
            provs += [self._build_opp_provider(st, mn, vr, ct)
                      for st, mn, vr, ct in zip(state_dicts, means, vars_, counts)]
            self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=int(seed))
            self.env._target_action_provider = self._pool_provider

        def _norm(self, obs):
            if self.obs_rms is None:
                return np.asarray(obs, dtype=np.float32)
            n = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            return np.clip(n, -OBS_CLIP, OBS_CLIP).astype(np.float32)

        def collect(self, n_steps):
            obs_buf = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            raw_buf = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            act_buf = np.zeros((n_steps, self.act_dim), dtype=np.float32)
            logp_buf = np.zeros(n_steps, dtype=np.float32)
            rew_buf = np.zeros(n_steps, dtype=np.float32)
            done_buf = np.zeros(n_steps, dtype=np.float32)
            val_buf = np.zeros(n_steps, dtype=np.float32)
            ep_returns, ep_lengths, ep_components, ep_outcomes = [], [], [], []
            ep_opp_indices = []
            ep_end_conditions = []

            for t in range(n_steps):
                raw = self._next_obs
                norm = self._norm(raw)
                raw_buf[t] = raw
                obs_buf[t] = norm
                done_buf[t] = float(self._next_done)
                with torch.no_grad():
                    a, lp, _, v = self.model.get_action_and_value(
                        torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0))
                a_np = a.squeeze(0).numpy().astype(np.float32)   # 카테고리 index (저장용)
                act_buf[t] = a_np
                logp_buf[t] = float(lp.item())
                val_buf[t] = float(v.item())

                env_action = discrete_indices_to_continuous(a_np, self.model.num_bins)
                # action history: env.step 이 next_obs 를 빌드하기 **전에** 방금 결정한
                # action 을 reconstructor 에 넣어, next_obs 가 이 action 을 포함하게 한다.
                if self._push_action is not None:
                    self._push_action(env_action)
                next_obs, reward, term, trunc, info = self.env.step(env_action)
                done = bool(term or trunc)
                if self._advance_recon is not None:
                    self._advance_recon(self.env._ownship_state, self.env._target_state)
                self._ep_return += float(reward)
                self._ep_len += 1
                rew_buf[t] = float(reward)   # raw 보상 그대로 (스케일링 없음)
                if done:
                    ep_returns.append(self._ep_return)
                    ep_lengths.append(self._ep_len)
                    ep_end_conditions.append(
                        str(info.get("end_condition", "")) if isinstance(info, dict) else "")
                    comp = info.get("ep_reward_components")
                    if isinstance(comp, dict):
                        ep_components.append(dict(comp))
                    oc = rollout_outcome(info)
                    if oc is not None:
                        ep_outcomes.append(oc)
                        prov = getattr(self.env, "_target_action_provider", None)
                        local = int(getattr(prov, "last_index", 0))
                        # 로컬 슬롯([BT?][MPC?][exploiter...][snapshot...]) → 글로벌 슬롯
                        # ([BT n_bt][MPC n_mpc][exploiter n_exp][snap]).
                        #   BT   : 글로벌 bt_index (0..n_bt-1)
                        #   MPC  : 글로벌 n_bt (모든 워커 공통 단일 슬롯)
                        #   exp  : n_bt + n_mpc + (로컬 exploiter 순번) — 모든 워커 공통
                        #   snap : n_bt + n_mpc + n_exp + (로컬 snapshot 순번)
                        bt_slots = int(getattr(self, "_bt_slots", 0))
                        mpc_slots = int(getattr(self, "_mpc_slots", 0))
                        cut_slots = int(getattr(self, "_cut_slots", 0))
                        exp_slots = int(getattr(self, "_exp_slots", 0))
                        n_bt = int(getattr(self, "_n_bt", 0))
                        n_mpc = int(getattr(self, "_n_mpc", 0))
                        n_cut = int(getattr(self, "_n_cut", 0))
                        n_exp = int(getattr(self, "_n_exp", 0))
                        if bt_slots and local < bt_slots:
                            gidx = int(getattr(self, "_bt_index", 0))
                        elif mpc_slots and local < bt_slots + mpc_slots:
                            gidx = n_bt                       # MPC 글로벌 슬롯
                        elif cut_slots and local < bt_slots + mpc_slots + cut_slots:
                            gidx = n_bt + n_mpc               # CUTOFF 글로벌 슬롯
                        elif exp_slots and local < bt_slots + mpc_slots + cut_slots + exp_slots:
                            gidx = n_bt + n_mpc + n_cut + (local - bt_slots - mpc_slots - cut_slots)
                        else:
                            gidx = (n_bt + n_mpc + n_cut + n_exp
                                    + (local - bt_slots - mpc_slots - cut_slots - exp_slots))
                        ep_opp_indices.append(gidx)
                    self._ep_return = 0.0
                    self._ep_len = 0
                    if self._reset_recon is not None:
                        self._reset_recon()
                    next_obs, _ = self.env.reset()
                    # 에피소드 첫 관측은 GPU 학습과 동일하게 fresh recon(advance 안 함).
                # 0-lag: 이번 step advance 뒤 관측을 다시 만든다(reconstruct 사용 시). done 이면
                # 위 reset 된 fresh recon 으로 obs(0) 를 만든다(= env.reset 관측과 동일).
                if self._advance_recon is not None:
                    next_obs = self.env.get_observation()
                self._next_obs = np.asarray(next_obs, dtype=np.float32)
                self._next_done = done

            with torch.no_grad():
                last_v = float(self.model.get_value(
                    torch.as_tensor(self._norm(self._next_obs), dtype=torch.float32).unsqueeze(0)).item())
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, self._next_done,
                                   self.gamma, self.gae_lambda)
            return {
                "obs": obs_buf, "actions": act_buf, "logp": logp_buf,
                "advantages": adv, "returns": ret, "values": val_buf,
                "ep_returns": ep_returns, "ep_lengths": ep_lengths,
                "ep_components": ep_components, "ep_outcomes": ep_outcomes,
                "ep_opp_indices": ep_opp_indices,
                "ep_end_conditions": ep_end_conditions,
                "rms_mean": (raw_buf.mean(0) if self.obs_rms is not None else None),
                "rms_var": (raw_buf.var(0) if self.obs_rms is not None else None),
                "rms_count": n_steps,
            }

        def eval_games(self, opp_state, opp_model_kwargs, opp_rms_dict,
                       cur_mean, cur_var, seeds, stochastic):
            """현재 self.model vs 과거 snapshot(opp_*) 으로 seeds 만큼 평가.

            env 의 상대를 과거 network 로 잠시 교체하고, 끝나면 원래 self-play
            상대를 복원한다. 평가가 진행 중 episode 를 끊으므로 이후 collect 의
            연속성을 위해 env / reconstructor / episode 누적을 새로 리셋한다.
            """
            from claude_code import evaluation
            if self._opp_model is None:
                self._opp_model = make_actor_critic(**opp_model_kwargs)
            self._opp_model.load_state_dict({k: torch.as_tensor(v) for k, v in opp_state.items()})
            self._opp_model.eval()

            prev = self.env._target_action_provider
            self.env._target_action_provider = evaluation.make_opponent(
                self.env, self._opp_model, opp_rms_dict, stochastic)
            try:
                results = evaluation.play_games(
                    self.env, self.model, cur_mean, cur_var, seeds, stochastic,
                    self.reconstruct, "cpu")
            finally:
                self.env._target_action_provider = prev
                # 다음 collect 의 rollout 연속성 복구 (진행 중 episode 폐기)
                if self._reset_recon is not None:
                    self._reset_recon()
                obs, _ = self.env.reset()
                if self._reset_recon is not None:
                    self._reset_recon()
                self._next_obs = np.asarray(obs, dtype=np.float32)
                self._next_done = False
                self._ep_return = 0.0
                self._ep_len = 0
            return results

        def close_providers(self):
            """워커의 opponent provider 들을 닫는다(컷오프 exe 프로세스 종료 등). 학습 종료 시 호출."""
            try:
                prov = getattr(self, "_pool_provider", None)
                if prov is not None:
                    prov.close()          # PoolSelfPlayProvider → 모든 sub-provider.close()
                cut = getattr(self, "_cutoff_provider", None)
                if cut is not None:
                    cut.close()           # exe 프로세스 명시적 종료(중복 close 는 무해)
            except Exception:
                pass

    return RolloutWorker


# ── 병렬 driver ───────────────────────────────────────────────────────────────

class ParallelPPOTrainer:
    """Ray worker 들로 rollout 을 병렬 수집하고, driver 에서 PPO update(=PPOTrainer.update)."""

    def __init__(self, env_kwargs, config: PPOConfig, num_workers: int,
                 self_play: bool, obs_dim: int, act_dim: int,
                 bt_opponents=None, mpc_opponent=None, cutoff_opponent=None):
        import ray
        self.cfg = config
        self.num_workers = max(1, int(num_workers))
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_step = 0
        # MPC(team-share) 고정 상대. (mpc_root, mpc_config) 또는 None. BT 와 달리 프로세스
        # 제약이 없어 모든 워커에 동일하게 넣는다(글로벌 단일 슬롯 n_bt). n_mpc = 0/1.
        self._mpc_root, self._mpc_config = (mpc_opponent or ("", ""))
        self._n_mpc = 1 if self._mpc_root else 0
        # 컷오프 exe 고정 상대(dict 또는 None). MPC 와 같은 never-evict 글로벌 슬롯(n_bt+n_mpc)이나,
        # 워커마다 exe 1개를 상주 실행한다(포트 base_port+worker_id). UnrealExeProvider 사용.
        self._cutoff = dict(cutoff_opponent) if cutoff_opponent else None
        self._n_cut = 1 if self._cutoff else 0
        # opponent pool 에 넣을 BT 목록 [(dll, rule), ...]. 워커를 여기에 round-robin 배정한다.
        # 한 프로세스 = BT rule 1개 제약 때문에, 워커별로 다른 BT 를 고정 배정하고 그 rule 을
        # 워커 프로세스 시작 시점(runtime_env)에 AIP_RULE_XML 로 주입한다(claude_code.bt_rule).
        self._bt_opponents = list(bt_opponents or [])
        self._n_bt = len(self._bt_opponents)
        # 워커 w 의 배정: (bt_dll, bt_rule, bt_index=글로벌 BT 슬롯). n_bt=0 이면 BT 없음.
        self._bt_assign = [
            (self._bt_opponents[i % self._n_bt][0], self._bt_opponents[i % self._n_bt][1],
             i % self._n_bt) if self._n_bt else ("", "", 0)
            for i in range(self.num_workers)
        ]

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.model = make_actor_critic(
            obs_dim, act_dim, hidden=config.hidden, activation=config.activation,
            critic_hidden=config.critic_hidden, critic_activation=config.critic_activation,
            num_bins=config.num_bins).to(config.device)
        # actor / critic 별도 optimizer (완전 분리). PPOTrainer.update 가 이 둘을 사용.
        self.actor_lr0 = config.lr
        self.critic_lr0 = config.critic_lr if config.critic_lr is not None else config.lr
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=self.actor_lr0, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=self.critic_lr0, eps=1e-5)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None

        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(config.hidden),
                            activation=config.activation, num_bins=config.num_bins,
                            critic_hidden=(tuple(config.critic_hidden) if config.critic_hidden else None),
                            critic_activation=config.critic_activation)
        self._model_kwargs = model_kwargs   # scratch exploiter 재생성용
        cfg_dict = dict(gamma=config.gamma, gae_lambda=config.gae_lambda,
                        normalize_obs=config.normalize_obs,
                        reconstruct_state=config.reconstruct_state)

        pythonpath = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
        if not ray.is_initialized():
            # job-level 에는 PYTHONPATH 만. AIP_RULE_XML 은 **워커별로** 주입한다(아래).
            ray.init(num_cpus=self.num_workers, include_dashboard=False,
                     ignore_reinit_error=True, log_to_driver=False,
                     runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        WorkerCls = _make_worker_cls()
        # 워커마다 자기 BT rule 을 AIP_RULE_XML 로 주입(프로세스 시작 시점 = JSBSim 로드 전).
        # BT rule XML 은 worker 프로세스 시작 시점에 들어가 있어야 한다. worker 안에서
        # os.environ 을 바꾸면 JSBSimAIPLib.dll 이 이미 로드된 뒤라 무시된다(claude_code.bt_rule).
        self.workers = []
        for i in range(self.num_workers):
            _dll, _rule, _bti = self._bt_assign[i]
            evars = {"PYTHONPATH": pythonpath}
            if _rule:
                evars[BT_RULE_ENV_KEY] = _rule
            self.workers.append(
                WorkerCls.options(runtime_env={"env_vars": evars}).remote(
                    i, env_kwargs, model_kwargs, cfg_dict, self_play, config.seed + 1 + i))
        if self._n_bt:
            groups = {}
            for i, (_d, _r, _b) in enumerate(self._bt_assign):
                groups.setdefault(_d, []).append(i)
            print(f"[claude_code/PPO] Ray 병렬 수집: workers={self.num_workers}, "
                  f"BT {self._n_bt}종 워커 분할 " + ", ".join(f"{d}:{ws}" for d, ws in groups.items()))
        else:
            print(f"[claude_code/PPO] Ray 병렬 수집: workers={self.num_workers} (BT 없음)")

    def _broadcast(self):
        import ray
        state = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        ref = ray.put(state)
        calls = [w.set_weights.remote(ref) for w in self.workers]
        if self.obs_rms is not None:
            calls += [w.set_obs_rms.remote(self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count)
                      for w in self.workers]
        # 단계 스케줄(gamma·damage 가중치·shaping 배율)을 매 iteration 워커에 반영.
        sr = getattr(self, "_sched_reward", None)
        if sr is not None:
            calls += [w.set_schedule.remote(sr["gamma"], sr["own_damage_weight"], sr["shaping_mult"])
                      for w in self.workers]
        ray.get(calls)

    def collect_rollout(self):
        import ray
        self._broadcast()
        per = max(1, self.cfg.rollout_steps // self.num_workers)
        results = ray.get([w.collect.remote(per) for w in self.workers])

        device = self.cfg.device
        cat = lambda key: np.concatenate([r[key] for r in results], axis=0)
        batch = {
            "obs": torch.as_tensor(cat("obs"), device=device),
            "actions": torch.as_tensor(cat("actions"), device=device),
            "logp": torch.as_tensor(cat("logp"), device=device),
            "advantages": torch.as_tensor(cat("advantages"), device=device),
            "returns": torch.as_tensor(cat("returns"), device=device),
            "values": torch.as_tensor(cat("values"), device=device),
        }
        self.global_step += batch["obs"].shape[0]

        # obs_rms 병합 (각 worker batch 통계를 authoritative obs_rms 에 누적)
        if self.obs_rms is not None:
            for r in results:
                if r["rms_mean"] is not None:
                    self.obs_rms._update_from_moments(
                        np.asarray(r["rms_mean"]), np.asarray(r["rms_var"]), r["rms_count"])

        ep_returns = [x for r in results for x in r["ep_returns"]]
        ep_lengths = [x for r in results for x in r["ep_lengths"]]
        ep_components = [x for r in results for x in r["ep_components"]]
        ep_outcomes = [x for r in results for x in r.get("ep_outcomes", [])]
        ep_opp_indices = [x for r in results for x in r.get("ep_opp_indices", [])]
        ep_end_conditions = [x for r in results for x in r.get("ep_end_conditions", [])]
        return (batch, ep_returns, ep_lengths, ep_components, ep_outcomes,
                ep_opp_indices, ep_end_conditions)

    # 업데이트 로직은 PPOTrainer.update 재사용
    def update(self, batch):
        return PPOTrainer.update(self, batch)

    # ── past-self stochastic 평가 (멀티프로세스: worker 재사용) ───────────────
    def evaluate_vs(self, opp_state, opp_model_kwargs, opp_rms_dict,
                    n_games, stochastic, base_seed):
        """현재 정책(post-update) vs 과거 snapshot 을 worker 들에 분배해 평가."""
        import ray
        from claude_code import evaluation
        self._broadcast()   # worker 에 현재 weights + obs_rms 반영
        mean = self.obs_rms.mean if self.obs_rms is not None else None
        var = self.obs_rms.var if self.obs_rms is not None else None
        seeds = [base_seed + i for i in range(n_games)]
        chunks = [seeds[i::self.num_workers] for i in range(self.num_workers)]
        opp_ref = ray.put(opp_state)
        futs = []
        for w, ch in zip(self.workers, chunks):
            if not ch:
                continue
            futs.append(w.eval_games.remote(
                opp_ref, opp_model_kwargs, opp_rms_dict, mean, var, ch, stochastic))
        results = [r for sub in ray.get(futs) for r in sub]
        return evaluation.summarize(results)

    def set_frozen_opponent(self, state_dict, rms_mean, rms_var, rms_count):
        """모든 worker 의 self-play 상대를 '초기 actor net 고정' 으로 교체(broadcast)."""
        import ray
        ref = ray.put(state_dict)
        ray.get([w.set_frozen_opponent.remote(ref, rms_mean, rms_var, rms_count)
                 for w in self.workers])

    def _current_state_rms(self):
        state = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        mean = self.obs_rms.mean if self.obs_rms is not None else None
        var = self.obs_rms.var if self.obs_rms is not None else None
        count = self.obs_rms.count if self.obs_rms is not None else 0.0
        return state, mean, var, count

    @staticmethod
    def _split_entries(entries):
        """pool 후보 entry 리스트 → (states, means, vars, counts) 로 분해(worker 전달용)."""
        entries = list(entries or [])
        states = [e["state"] for e in entries]
        means = [(e["rms"]["mean"] if e.get("rms") else None) for e in entries]
        vars_ = [(e["rms"]["var"] if e.get("rms") else None) for e in entries]
        counts = [(e["rms"]["count"] if e.get("rms") else 0.0) for e in entries]
        return states, means, vars_, counts

    def _cut_kw(self, i: int) -> dict:
        """워커 i 의 컷오프 exe pool_init/pool_set_all 키워드 인자(포트=base+i). 없으면 n_cut=0."""
        if not self._cutoff:
            return dict(cutoff_exe="", cutoff_port=0, cutoff_force_own=1,
                        cutoff_force_tgt=2, cutoff_timeout=0.5, n_cut=0)
        c = self._cutoff
        return dict(cutoff_exe=str(c["exe_path"]), cutoff_port=int(c["base_port"]) + int(i),
                    cutoff_force_own=int(c["force_own"]), cutoff_force_tgt=int(c["force_tgt"]),
                    cutoff_timeout=float(c["step_timeout_sec"]), n_cut=1)

    def install_opponent_pool(self, pool_max, per_worker_weights, exploiter_entries=None) -> None:
        """모든 worker 의 opponent pool 초기화. broadcast.

        각 워커는 자기에 배정된 BT 1개(self._bt_assign) + MPC + exploiter(있으면) +
        현재 정책 snapshot 1개로 시작한다. per_worker_weights[i] = 워커 i 의 로컬 pool
        ([그 워커 BT, MPC?, exploiter..., snapshot...]) 샘플 가중치.
        """
        import ray
        self._pool_max = max(1, int(pool_max))
        state, mean, var, count = self._current_state_rms()
        ref = ray.put(state)
        es, em, ev, ec = self._split_entries(exploiter_entries)
        n_exp = len(es)
        eref = ray.put(es)
        ray.get([
            w.pool_init.remote(ref, mean, var, count, list(per_worker_weights[i]),
                               self._pool_max, self.cfg.seed + 101 + i,
                               dll, rule, bti, self._n_bt,
                               self._mpc_root, self._mpc_config, self._n_mpc,
                               eref, em, ev, ec, n_exp, **self._cut_kw(i))
            for i, (w, (dll, rule, bti)) in enumerate(zip(self.workers, self._bt_assign))
        ])

    def pool_add_exploiter(self, snapshot) -> None:
        """학습한 exploiter snapshot({state, rms})을 모든 worker 의 pool 고정 exploiter
        영역에 추가(never-evict) + exploiter 학습 중 frozen 상대로 바뀌었던 provider 를
        pool 로 복귀시킨다. run_exploiter 로 main 학습 상태를 이미 복원한 뒤 호출한다.
        """
        import ray
        rms = snapshot.get("rms") or {}
        mean = rms.get("mean"); var = rms.get("var"); count = rms.get("count", 0.0)
        ref = ray.put(snapshot["state"])
        ray.get([w.pool_add_exploiter.remote(ref, mean, var, count) for w in self.workers])

    def pool_add_current(self) -> None:
        """현재 정책 frozen copy 를 모든 worker 의 pool 에 추가(초과 시 oldest 제거)."""
        import ray
        state, mean, var, count = self._current_state_rms()
        ref = ray.put(state)
        ray.get([w.pool_add.remote(ref, mean, var, count) for w in self.workers])

    def pool_set_weights(self, per_worker_weights) -> None:
        """워커별 opponent 샘플링 가중치 갱신(broadcast). per_worker_weights[i] = 워커 i 로컬 가중치."""
        import ray
        ray.get([w.pool_set_weights.remote(list(per_worker_weights[i]))
                 for i, w in enumerate(self.workers)])

    def snapshot_current(self) -> dict:
        """현재 정책 weights + obs_rms 통계를 checkpoint 용 dict 로 반환(numpy)."""
        state, mean, var, count = self._current_state_rms()
        rms = None if self.obs_rms is None else {
            "mean": np.asarray(mean, dtype=np.float64),
            "var": np.asarray(var, dtype=np.float64),
            "count": float(count),
        }
        return {"state": state, "rms": rms}

    def set_opponent_pool(self, entries, per_worker_weights, pool_max,
                          exploiter_entries=None) -> None:
        """checkpoint 의 opponent pool 을 모든 worker 에 복원(broadcast).

        entries 는 **snapshot 후보만**(오래된→최신). exploiter_entries 는 고정 exploiter
        후보들(BT·MPC 다음, snapshot 앞). 각 워커는 자기 BT(self._bt_assign) + MPC +
        exploiter + snapshot 으로 로컬 pool 을 만든다. per_worker_weights[i] = 워커 i 로컬 가중치.
        """
        import ray
        self._pool_max = max(1, int(pool_max))
        states, means, vars_, counts = self._split_entries(entries)
        es, em, ev, ec = self._split_entries(exploiter_entries)
        n_exp = len(es)
        sref = ray.put(states)
        eref = ray.put(es)
        ray.get([
            w.pool_set_all.remote(sref, means, vars_, counts, list(per_worker_weights[i]),
                                  self._pool_max, self.cfg.seed + 101 + i,
                                  dll, rule, bti, self._n_bt,
                                  self._mpc_root, self._mpc_config, self._n_mpc,
                                  eref, em, ev, ec, n_exp, **self._cut_kw(i))
            for i, (w, (dll, rule, bti)) in enumerate(zip(self.workers, self._bt_assign))
        ])

    _EXP_CKPT_FORMAT = "claude_code_exploiter_ckpt"
    _EXP_CKPT_VERSION = 1

    def _exploiter_ckpt_save(self, path, *, main_iteration, next_i, wr, ran,
                             frozen_main, cfg_params) -> None:
        """exploiter 진행 상태를 원자적으로 저장(매 exploiter-iter). 재개용.

        저장: exploiter model/optimizer/obs_rms/global_step + 다음에 실행할 exploiter-iter
        (next_i) + 얼려둔 상대(frozen_main = main snapshot) + 재현용 cfg 파라미터.
        frozen_main 을 함께 저장해, main iteration 을 재실행해 얻은(살짝 다른) 정책이 아니라
        **원래 얼렸던 그 상대**로 exploiter 를 일관되게 이어서 학습한다.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rms = None
        if self.obs_rms is not None:
            rms = {"mean": np.asarray(self.obs_rms.mean, dtype=np.float64),
                   "var": np.asarray(self.obs_rms.var, dtype=np.float64),
                   "count": float(self.obs_rms.count)}
        ckpt = {
            "format": self._EXP_CKPT_FORMAT, "version": self._EXP_CKPT_VERSION,
            "main_iteration": (int(main_iteration) if main_iteration is not None else None),
            "next_i": int(next_i), "wr": float(wr), "ran": int(ran),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "obs_rms": rms, "global_step": int(self.global_step),
            "frozen_main": frozen_main, "cfg_params": dict(cfg_params),
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(ckpt, str(tmp))
        os.replace(str(tmp), str(p))   # 원자적 교체 → 저장 도중 중단돼도 직전 ckpt 보존

    def _exploiter_ckpt_load(self, path, *, main_iteration):
        """exploiter 서브 체크포인트를 로드(없거나 불일치/손상이면 None → 새로 시작)."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[claude_code/PPO] exploiter ckpt 로드 실패({e}) → 새로 시작", flush=True)
            return None
        if ckpt.get("format") != self._EXP_CKPT_FORMAT:
            return None
        ci = ckpt.get("main_iteration")
        if main_iteration is not None and ci is not None and int(ci) != int(main_iteration):
            print(f"[claude_code/PPO] exploiter ckpt main_iteration 불일치"
                  f"({ci}!={main_iteration}) → 무시하고 새로 시작", flush=True)
            return None
        return ckpt

    def run_exploiter(self, main_snapshot, *, max_iters, win_target, rollout_steps,
                      lr, ent_coef, clip_coef, critic_lr=None, seed=0, log_cb=None,
                      ckpt_path=None, save_every=1, main_iteration=None):
        """main agent 를 잠깐 멈추고, main_snapshot 을 유일 상대로 삼아 exploiter 를 학습한다.
        승률≥win_target 또는 max_iters 도달 시 종료.

        학습 대상(self.model)·optimizer·obs_rms·global_step·관련 cfg 를 저장했다가, 종료 후
        원래 main 학습 상태로 완전히 복원한다(exploiter step 은 main global_step 에 반영 안 됨).

        ckpt_path 가 주어지면 **매 exploiter-iter(save_every 주기)마다** 진행 상태를 저장하고,
        시작 시 그 파일이 있으면(=학습 도중 중단됐던 경우) 중단 지점 exploiter-iter 부터 이어서
        학습한다. 정상 종료하면 그 파일을 삭제한다. 파일이 없으면 scratch(random init) 로 시작.

        log_cb: exploiter iter 마다 호출되는 콜백(선택). main 로깅과 완전히 분리된 별도 wandb
                run 에 지표를 남기기 위한 훅. `log_cb(dict)` 형태로 iter 단위 지표를 넘긴다.
        반환: (exploiter_snapshot, final_win_rate, iters_run).
        """
        import time
        # 1) main 학습 상태 저장(복원용). (재개 시엔 재실행된 main iter 의 현재 정책 = 복원 대상)
        saved_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        saved_actor_opt = self.actor_opt.state_dict()
        saved_critic_opt = self.critic_opt.state_dict()
        saved_rms = None
        if self.obs_rms is not None:
            saved_rms = (self.obs_rms.mean.copy(), self.obs_rms.var.copy(), float(self.obs_rms.count))
        saved_step = int(self.global_step)
        saved_cfg = {k: getattr(self.cfg, k) for k in ("rollout_steps", "ent_coef", "clip_coef")}
        saved_sched = (getattr(self, "_sched_base", None), getattr(self, "_sched_phase", None))

        # 재개 여부 판단(있으면 exploiter 서브 체크포인트에서 상태 복원).
        resume = self._exploiter_ckpt_load(ckpt_path, main_iteration=main_iteration) \
            if ckpt_path is not None else None
        cfg_params = {"rollout_steps": rollout_steps, "lr": lr, "ent_coef": ent_coef,
                      "clip_coef": clip_coef, "critic_lr": critic_lr, "seed": seed,
                      "max_iters": max_iters, "win_target": win_target}
        clr = lr if critic_lr is None else critic_lr

        # 2) 모든 worker 의 상대를 frozen main 으로 교체. 재개면 원래 얼렸던 상대를 그대로 사용.
        if resume is not None:
            frozen_main = resume["frozen_main"]
        else:
            frozen_main = main_snapshot
        fmr = frozen_main.get("rms") or {}
        self.set_frozen_opponent(frozen_main["state"], fmr.get("mean"),
                                 fmr.get("var"), fmr.get("count", 0.0))

        # 3) exploiter 초기화. 재개면 저장된 model/optimizer/obs_rms/step 복원, 아니면 scratch.
        self.cfg.rollout_steps = int(rollout_steps)
        self.cfg.ent_coef = float(ent_coef)
        self.cfg.clip_coef = float(clip_coef)
        if resume is not None:
            self.model.load_state_dict({k: torch.as_tensor(v)
                                        for k, v in resume["model_state"].items()})
            self.model.to(self.cfg.device)
            self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=lr, eps=1e-5)
            self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)
            self.actor_opt.load_state_dict(resume["actor_opt"])
            self.critic_opt.load_state_dict(resume["critic_opt"])
            if self.obs_rms is not None and resume.get("obs_rms"):
                r = resume["obs_rms"]
                self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
                self.obs_rms.mean = np.asarray(r["mean"], dtype=np.float64)
                self.obs_rms.var = np.asarray(r["var"], dtype=np.float64)
                self.obs_rms.count = float(r["count"])
            self.global_step = int(resume.get("global_step", self.global_step))
            start_i = int(resume["next_i"]); wr = float(resume.get("wr", 0.0))
            ran = int(resume.get("ran", start_i - 1))
            print(f"[claude_code/PPO] === exploiter 재개(vs main): iter {start_i}부터 "
                  f"(직전 wr {wr:.3f}, max_iters={max_iters}, target_wr={win_target:.2f}) ===",
                  flush=True)
        else:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            fresh = make_actor_critic(**self._model_kwargs).to(self.cfg.device)
            self.model.load_state_dict(fresh.state_dict())
            self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=lr, eps=1e-5)
            self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)
            if self.obs_rms is not None:
                self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
            start_i = 1; wr = 0.0; ran = 0
            print(f"[claude_code/PPO] === exploiter 학습 시작(vs main): max_iters={max_iters}, "
                  f"target_wr={win_target:.2f}, rollout={rollout_steps}, lr={lr:.1e}, "
                  f"ent={ent_coef:.1e}, clip={clip_coef} ===", flush=True)

        # 4) exploiter 학습 루프(승률은 exploiter=ownship 관점 outcome 으로 계산).
        for i in range(int(start_i), int(max_iters) + 1):
            t0 = time.time()
            batch, ep_returns, ep_lengths, _, ep_outcomes, _, _ = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)
            ran = i
            oc = _outcome_counts(ep_outcomes)
            w, l, d = int(oc.get("win", 0)), int(oc.get("loss", 0)), int(oc.get("draw", 0))
            dec = w + l + d
            wr = (w / dec) if dec else 0.0
            mret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mlen = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            elapsed = time.time() - t0
            print(f"[claude_code/PPO]   exploiter iter {i:3d} | vs-main W/L/D {w}/{l}/{d} "
                  f"wr {wr:.3f} | ret {mret:8.3f} | {elapsed:.1f}s", flush=True)
            if log_cb is not None:
                try:
                    log_cb({
                        "iter": i, "win_rate": wr, "wins": w, "losses": l, "draws": d,
                        "decided": dec, "mean_return": mret, "mean_length": mlen,
                        "completed_episodes": len(ep_returns),
                        "policy_loss": pl, "value_loss": vl, "entropy": ent,
                        "approx_kl": kl, "explained_variance": ev,
                        "rollout_steps": int(rollout_steps), "lr": float(lr),
                        "ent_coef": float(ent_coef), "clip_coef": float(clip_coef),
                        "elapsed_sec": elapsed,
                    })
                except Exception as e:
                    print(f"[claude_code/PPO]   exploiter log_cb 실패({e})", flush=True)
            # 진행 상태 저장(매 exploiter-iter). next_i=i+1 → 중단 시 다음 iter 부터 재개.
            if ckpt_path is not None and (int(save_every) <= 1 or i % int(save_every) == 0):
                try:
                    self._exploiter_ckpt_save(ckpt_path, main_iteration=main_iteration,
                                              next_i=i + 1, wr=wr, ran=ran,
                                              frozen_main=frozen_main, cfg_params=cfg_params)
                except Exception as e:
                    print(f"[claude_code/PPO]   exploiter ckpt 저장 실패({e})", flush=True)
            if dec > 0 and wr >= win_target:
                print(f"[claude_code/PPO]   exploiter 승률 목표 달성(wr {wr:.3f} ≥ {win_target:.2f}) "
                      f"@ iter {i}", flush=True)
                break
        else:
            print(f"[claude_code/PPO]   exploiter max_iters({max_iters}) 도달, 최종 wr {wr:.3f}",
                  flush=True)

        # 5) exploiter snapshot 확보(복원 전).
        exploiter_snapshot = self.snapshot_current()
        # exploiter 정상 종료 → 서브 체크포인트 삭제(다음 exploiter 는 새로 시작).
        if ckpt_path is not None:
            try:
                if Path(ckpt_path).exists():
                    os.remove(str(ckpt_path))
            except Exception as e:
                print(f"[claude_code/PPO] exploiter ckpt 삭제 실패({e})", flush=True)

        # 6) main 학습 상태 완전 복원.
        self.model.load_state_dict(saved_model)
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=self.actor_lr0, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=self.critic_lr0, eps=1e-5)
        self.actor_opt.load_state_dict(saved_actor_opt)     # lr·momentum 포함 복원
        self.critic_opt.load_state_dict(saved_critic_opt)
        if self.obs_rms is not None and saved_rms is not None:
            self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = (
                saved_rms[0], saved_rms[1], saved_rms[2])
        self.global_step = saved_step
        for k, v in saved_cfg.items():
            setattr(self.cfg, k, v)
        if saved_sched[0] is not None:
            self._sched_base = saved_sched[0]
        if saved_sched[1] is not None:
            self._sched_phase = saved_sched[1]
        return exploiter_snapshot, wr, ran

    def train(self, on_iteration=None, start_iteration: int = 1):
        import time
        history = []
        it = int(start_iteration)
        while self.cfg.total_iterations <= 0 or it <= self.cfg.total_iterations:
            PPOTrainer._apply_iteration_schedule(self, it)
            t0 = time.time()
            (batch, ep_returns, ep_lengths, ep_components,
             ep_outcomes, ep_opp_indices, ep_end_conditions) = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            comp_means = {}
            if ep_components:
                for key in ("pursuit", "damage", "shaping", "terminal", "safety", "step"):
                    comp_means[key] = float(np.mean([c.get(key, 0.0) for c in ep_components]))
            comp_means.update(_outcome_counts(ep_outcomes))
            comp_means["per_opp"] = _outcome_counts_by_opp(ep_opp_indices, ep_outcomes)
            comp_means["alt_term"] = _count_altitude_terms(ep_end_conditions)
            comp_means["update_epochs"] = int(self.last_update_epochs)
            comp_means["critic_epochs"] = int(self.last_critic_epochs)
            comp_means["update_early_stop"] = int(self.last_update_early_stop)
            stats = IterationStats(
                iteration=it, global_step=self.global_step, mean_return=mean_ret,
                mean_length=mean_len, completed_episodes=len(ep_returns),
                policy_loss=pl, value_loss=vl, entropy=ent, approx_kl=kl,
                explained_variance=ev, elapsed_sec=time.time() - t0, extra=comp_means)
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats)
            it += 1
        return history

    def close(self):
        try:
            import ray
            # 워커의 opponent provider(특히 컷오프 exe 프로세스)를 먼저 정리한 뒤 Ray 종료.
            if self._n_cut and getattr(self, "workers", None):
                try:
                    ray.get([w.close_providers.remote() for w in self.workers])
                except Exception:
                    pass
            ray.shutdown()
        except Exception:
            pass


__all__ = ["physical_cpu_count", "ParallelPPOTrainer"]
