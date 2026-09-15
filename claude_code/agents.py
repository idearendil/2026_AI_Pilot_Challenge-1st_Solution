# -*- coding: utf-8 -*-
"""ownship/target 슬롯에 넣을 agent 로딩 + provider 생성 공용 유틸.

power_test / final_power_test / run_local_dogfight 가 공유한다. 한 "side"(ownship 또는
target)는 아래 소스 중 하나로 지정한다(agent spec 문자열):

  - ``bundle:<경로>``  CPU/CUDA 학습 번들(metadata.json + policy_weights.pkl.gz).
  - ``ckpt:<경로>``    CUDA 학습 체크포인트(runs/*.pt, PPOGPUTrainer.save 포맷).
  - ``bt:<이름>``      baselines/ 의 BT DLL(Jeon_BT1/Jeon_BT2/Lee_BT1/Shin_BT_best/Shin_BT_def).
  - ``release_mpc``    baselines/Release_MPC_team_share (team-share MPC, 60Hz).
  - ``stable_mpc``     baselines/Stable_MPC_team_share (safe MPC, 60Hz).
  - ``unreal_exe``     baselines/unreal_bt_client.exe (외부 BT UDP 클라이언트, 60Hz).

두 학습 pipeline(CPU claude_code.train / CUDA cuda_fdm.train_gpu) 모두 관측이
claude164r(my_observation)라 bundle·ckpt 를 슬롯 무관하게 서로 붙일 수 있다. 신경망
소스는 각자 독립 reconstructor 를 갖는 SelfPlayProvider(10Hz) 또는 HighRateProvider(60Hz)
로 굴려(전역 _RECON 오염 없음, 양측 공정 mirror), argmax(deterministic)/stochastic 을 고른다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from claude_code.bt_rule import ENV_KEY, rule_for  # noqa: F401 (leaf 모듈)

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"

# 학습 관측 모듈(두 pipeline 공통). nn side 가 있으면 env 를 이 모듈로 만든다.
OBS_MODULE = "claude_code.my_observation"

BT_DLLS = ["Jeon_BT1.dll", "Jeon_BT2.dll", "Lee_BT1.dll",
           "Shin_BT_best.dll", "Shin_BT_def.dll"]
RELEASE_MPC_ROOT = str(BASELINES / "Release_MPC_team_share")
STABLE_MPC_ROOT = str(BASELINES / "Stable_MPC_team_share")
UNREAL_EXE = str(BASELINES / "unreal_bt_client.exe")
UNREAL_BASE_PORT = 9600

NN_KINDS = {"bundle", "ckpt"}
BASELINE_KINDS = {"bt", "release_mpc", "stable_mpc", "unreal_exe"}


# ── agent spec 파싱 ───────────────────────────────────────────────────────────
def parse_agent_spec(s: str) -> dict:
    """agent spec 문자열 → dict(kind, ...). 위 docstring 의 형식을 받는다."""
    s = (s or "").strip()
    head, _, rest = s.partition(":")
    head = head.lower()
    rest = rest.strip()
    if head == "bundle":
        if not rest:
            raise ValueError("bundle:<경로> 형식으로 번들 디렉토리를 지정하세요")
        return {"kind": "bundle", "name": Path(rest).name, "path": rest}
    if head == "ckpt":
        if not rest:
            raise ValueError("ckpt:<경로> 형식으로 체크포인트(.pt)를 지정하세요")
        return {"kind": "ckpt", "name": Path(rest).stem, "path": rest}
    if head == "bt":
        name = rest or "Lee_BT1"
        if not name.endswith(".dll"):
            dll = name + ".dll"
        else:
            dll, name = name, name[:-4]
        return {"kind": "bt", "name": name, "dll": f"baselines/{dll}",
                "rule": f"./baselines/{name}.xml"}
    if head in ("release_mpc", "team_mpc"):
        return {"kind": "release_mpc", "name": "Release_MPC"}
    if head in ("stable_mpc", "safe_mpc"):
        return {"kind": "stable_mpc", "name": "Stable_MPC"}
    if head in ("unreal_exe", "unreal", "exe"):
        return {"kind": "unreal_exe", "name": "unreal_bt"}
    raise ValueError(
        f"알 수 없는 agent spec: {s!r}. "
        "bundle:<경로> / ckpt:<경로> / bt:<이름> / release_mpc / stable_mpc / unreal_exe")


def is_nn(spec: dict) -> bool:
    return spec["kind"] in NN_KINDS


def list_baselines() -> list[dict]:
    """설치된(파일이 실제로 있는) baseline 상대 전체 목록(final_power_test 순회용)."""
    out: list[dict] = []
    for dll in BT_DLLS:
        if (BASELINES / dll).is_file():
            name = dll[:-4]
            out.append({"kind": "bt", "name": name, "dll": f"baselines/{dll}",
                        "rule": f"./baselines/{name}.xml"})
    if Path(RELEASE_MPC_ROOT).is_dir():
        out.append({"kind": "release_mpc", "name": "Release_MPC"})
    if (Path(STABLE_MPC_ROOT) / "src" / "safe_mpc").is_dir():
        out.append({"kind": "stable_mpc", "name": "Stable_MPC"})
    if Path(UNREAL_EXE).is_file():
        out.append({"kind": "unreal_exe", "name": "unreal_bt"})
    return out


def bt_rule_for_specs(*specs: dict) -> str:
    """side 스펙들 중 BT 의 rule XML(Ray worker 프로세스에 주입할 값)을 고른다.

    한 프로세스에서 BT DLL 은 전역 AIP_RULE_XML 을 **한 번만** 읽으므로, 양측이 서로 다른
    rule 의 BT 이면 동시 지정이 불가능하다 → 명확히 실패시킨다."""
    rules = {s["rule"] for s in specs if s.get("kind") == "bt"}
    if len(rules) > 1:
        raise ValueError(
            "ownship·target 이 서로 다른 rule 의 BT 입니다. 한 프로세스에서 BT DLL 은 "
            "전역 AIP_RULE_XML 을 한 번만 읽어 동시 지정이 불가능합니다(같은 BT 는 가능).")
    return next(iter(rules)) if rules else ""


# ── 신경망 정책 로딩/ provider ────────────────────────────────────────────────
def load_policy(spec: dict, device: str = "cpu"):
    """bundle/ckpt spec → (model, obs_rms, obs_mode). 둘 다 claude164r 관측을 요구한다."""
    from claude_code.normalizers import RunningMeanStd
    if spec["kind"] == "bundle":
        from claude_code.model import load_bundle
        model, meta = load_bundle(spec["path"], device=device)
        if meta.get("observation_module") != OBS_MODULE:
            raise ValueError(
                f"claude164r({OBS_MODULE}) 번들만 지원합니다: {spec['path']} "
                f"(observation_module={meta.get('observation_module')!r}). 두 학습 "
                "pipeline 모두 이 관측으로 학습합니다.")
        rms = (RunningMeanStd.from_state_dict(meta["obs_normalization"])
               if meta.get("obs_normalization") else None)
        return model, rms, meta.get("observation_mode", "claude164r")
    if spec["kind"] == "ckpt":
        from cuda_fdm.gpu_ckpt_to_bundle import load_ckpt_as_model
        model, obs_norm, obs_mode = load_ckpt_as_model(spec["path"], device=device,
                                                       observation_module=OBS_MODULE)
        rms = (RunningMeanStd.from_state_dict(obs_norm) if obs_norm else None)
        return model, rms, obs_mode
    raise ValueError(f"신경망 소스가 아님: {spec['kind']!r}")


def make_policy_provider(env, model, obs_rms, step_ratio: int,
                         hz: int = 10, deterministic: bool = True):
    """로드된 정책 → provider. hz=10 → SelfPlayProvider, hz=60 → HighRateProvider.
    deterministic=True → argmax, False → stochastic(정책 분포 샘플링)."""
    explore = not deterministic
    if int(hz) == 60:
        from claude_code.high_rate import high_rate_from_model
        return high_rate_from_model(model, obs_rms, step_ratio=step_ratio,
                                    device="cpu", explore=explore)
    from claude_code.self_play import SelfPlayProvider
    return SelfPlayProvider(model, obs_rms, env._observation_fn, env._observation_mode,
                            step_ratio, "cpu", explore=explore)


# ── team-share MPC 계열 target 래퍼 ───────────────────────────────────────────
class _MPCTargetWrapper:
    """env 상태(pqr/시간 없음)를 각속도 추정으로 보강해 team-share MPC 계열 provider
    (Release MPCActionProvider / Stable SafeMPCActionProvider)를 60Hz 로 감싼다.

    env 는 provider 에 [0:9](NED pos·euler·body vel)만 채운 상태를 준다. MPC 는 pqr(deg)@[9:12],
    시간@[41] 을 기대하므로 각속도 추정기(60Hz)와 substep 카운터로 51D state 를 만든다(대회 서버
    방식). 내부 action_repeat=6 이 10Hz replan / 60Hz 출력을 만든다. 의존 클래스는 mpc 패키지
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


# ── baseline provider ─────────────────────────────────────────────────────────
def make_baseline_provider(env, spec: dict, step_ratio: int, *,
                           wid: int = 0, root=None, base_port: int = UNREAL_BASE_PORT):
    """baseline spec(bt/release_mpc/stable_mpc/unreal_exe) → provider.
    provider 는 env context(ownship_state=자기 기체)를 받으므로 슬롯 무관하게 동작한다."""
    kind = spec["kind"]
    if kind == "bt":
        from claude_code.self_play import make_bt_provider
        return make_bt_provider(spec["dll"], spec.get("rule", ""))
    if kind == "release_mpc":
        from claude_code.self_play import make_mpc_provider
        from claude_code.my_observation import SIM_HZ
        from dogfight.ai.action_provider import ActionContext, ActionResult
        base = make_mpc_provider(RELEASE_MPC_ROOT, "")
        from mpc.transforms import AngularRateEstimator   # make_mpc_provider 뒤라야 import 가능
        return _MPCTargetWrapper(base, AngularRateEstimator, SIM_HZ,
                                 ActionContext, ActionResult, "release_mpc")
    if kind == "stable_mpc":
        import sys as _sys
        from claude_code.my_observation import SIM_HZ
        from dogfight.ai.action_provider import ActionContext, ActionResult
        sroot = Path(STABLE_MPC_ROOT).resolve()
        ssrc = str(sroot / "src")
        if ssrc not in _sys.path:
            _sys.path.append(ssrc)
        from mpc.config import load_config
        from mpc.transforms import AngularRateEstimator
        from safe_mpc import SafeMPCActionProvider, load_safe_mpc_config
        base = SafeMPCActionProvider(
            sroot, load_config(str(sroot / "configs" / "mpc.yaml")),
            load_safe_mpc_config(str(sroot / "configs" / "safe_mpc.yaml")))
        return _MPCTargetWrapper(base, AngularRateEstimator, SIM_HZ,
                                 ActionContext, ActionResult, "stable_mpc")
    if kind == "unreal_exe":
        from claude_code.unreal_exe_provider import UnrealExeProvider
        return UnrealExeProvider(
            exe_path=UNREAL_EXE, port=int(base_port) + int(wid),
            own_plane_id=1, enemy_plane_id=0, ownship_force_side=1, target_force_side=2,
            cwd=str(root or ROOT), step_timeout_sec=0.5, quiet=True)
    raise ValueError(f"알 수 없는 baseline kind: {kind!r}")


def make_side_provider(env, spec: dict, step_ratio: int, *, hz: int = 10,
                       deterministic: bool = True, wid: int = 0, root=None,
                       base_port: int = UNREAL_BASE_PORT, device: str = "cpu"):
    """spec → provider. 신경망은 load_policy→make_policy_provider(hz·deterministic 반영),
    baseline 은 make_baseline_provider(자체 고정 제어주기)."""
    if is_nn(spec):
        model, rms, _ = load_policy(spec, device=device)
        return make_policy_provider(env, model, rms, step_ratio,
                                    hz=hz, deterministic=deterministic)
    return make_baseline_provider(env, spec, step_ratio, wid=wid, root=root,
                                  base_port=base_port)


__all__ = [
    "ROOT", "BASELINES", "OBS_MODULE", "BT_DLLS", "RELEASE_MPC_ROOT", "STABLE_MPC_ROOT",
    "UNREAL_EXE", "UNREAL_BASE_PORT", "NN_KINDS", "BASELINE_KINDS", "ENV_KEY",
    "parse_agent_spec", "is_nn", "list_baselines", "bt_rule_for_specs",
    "load_policy", "make_policy_provider", "make_baseline_provider", "make_side_provider",
]
