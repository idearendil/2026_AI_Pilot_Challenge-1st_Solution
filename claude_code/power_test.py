"""두 모델(rl 번들 / baseline BT DLL)을 N판 붙여 어느 쪽이 강한지 통계로 판정한다.

`run_local_dogfight.py` 와 **동일한 env·동일한 제어 주기**로 싸우되, 리플레이 로그는
안 만들고 Ray 로 여러 판을 동시에 돌린다. 판정은 승/패/무 집계 + 이항검정이라
"우연히 몇 판 이겼다" 와 "실제로 더 강하다" 를 구분할 수 있다.

  - **rl 은 항상 stochastic**(학습 때와 동일하게 정책 분포에서 샘플링).
  - **판마다 시드가 랜덤**(--seed 를 주면 그 시드에서 판별 시드를 파생해 재현 가능).
  - **시작 위치는 판마다 좌우 교대**(정확히 50/50). 시작 위치 유불리를 상쇄한다.
    한쪽 위치로 고정하려면 --fixed-side.

예시 (학습 번들 vs baseline BT, 100판):
  python claude_code/power_test.py \
    --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic \
    --target-backend bt --target-bt-dll Lee_BT1.dll --games 100

예시 (번들 A vs 번들 B):
  python claude_code/power_test.py \
    --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic \
    --target-backend rl --target-bundle-dir artifacts/models/team01/basic_old --games 100

예시 (내 rl 모델 vs 팀원 model2_gylee agent, 100판):
  python claude_code/power_test.py \
    --ownship-backend rl --ownship-bundle-dir artifacts/models/team01/basic \
    --target-backend gylee --games 100
  (다른 snapshot 을 붙이려면 --target-gylee-snapshot <경로>, stochastic 상대면
   --target-gylee-explore. 내 모델(ownship)은 항상 stochastic.)
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

# 콘솔 인코딩(cp949 등)에 없는 문자(em-dash, 화살표 등)를 print 해도 UnicodeEncodeError 로
# 죽지 않게 한다(train_redq.py 와 동일 관례). 인코딩은 유지하고 불가 문자만 안전 대체.
try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

# ── BT rule XML: 반드시 claude_code 의 다른 import 보다 먼저 ────────────────────
# AIP_RULE_XML 은 JSBSimAIPLib.dll 로드 시점(= claude_code.env_utils import 체인)에 한 번만
# 캐싱된다. 늦게 세팅하면 DLL 이 Rule_forTraining.xml(Task_Empty)로 폴백해 BT 가 조종을
# 전혀 안 한다 → 가짜 승률. 그래서 argv 를 미리 훑어 여기서 세팅한다(claude_code.bt_rule).
from claude_code.bt_rule import BT_RULE_DEFAULTS, ENV_KEY  # noqa: E402  (leaf 모듈)

_DEF_OWNSHIP_BT = "AIP_DCS_ownship.dll"
_DEF_TARGET_BT = "Lee_BT1.dll"   # 기존 baseline. Jeon_BT1 / Jeon_BT2 / Shin_BT1.dll 로도 지정 가능
# 팀원(gyLee) 패키지의 기본 opponent snapshot = 문서 계보의 iter_1415(동봉 SHA 일치·검증됨).
# 다른 snapshot(예: iter_2350)을 --target-gylee-snapshot 으로 지정하면 SHA 가 달라
# checksum 이 안 맞으므로 provider 생성 시 verify_checksum=False 로 로드한다.
_DEF_GYLEE_SNAPSHOT = str(ROOT / "model2_gylee" / "model" / "iter_1415.pt")


def _resolve_bt_rule(ns) -> str | None:
    """이번 대결에 쓸 rule XML(없으면 None = DLL 기본값)."""
    if getattr(ns, "bt_rule_xml", ""):
        return ns.bt_rule_xml
    dlls = []
    if ns.ownship_backend == "bt":
        dlls.append(ns.ownship_bt_dll)
    if ns.target_backend == "bt":
        dlls.append(ns.target_bt_dll)
    rules = {BT_RULE_DEFAULTS.get(Path(d).name) for d in dlls}
    rules.discard(None)
    if len(rules) > 1:
        # AIP_RULE_XML 은 프로세스 전역이라 한 프로세스(=한 게임)에 rule 1개만 로드된다.
        # DLL 파일을 따로 복사해도 전부 이 전역 값을 읽으므로, 서로 다른 두 BT 를 동시에
        # 붙일 수 없다(BT vs BT 불가). BT 강도 비교는 공통 상대(RL)에 각 BT 를 따로 붙여서 한다.
        raise ValueError(
            f"ownship/target BT 가 서로 다른 rule 을 요구합니다: {sorted(rules)}.\n"
            "  한 프로세스 = BT rule 1개(AIP_RULE_XML 전역) 제약 때문에 서로 다른 두 BT 를\n"
            "  같은 게임에 붙일 수 없습니다(BT vs BT 불가). BT 강도 비교는 '공통 RL 상대 vs\n"
            "  각 BT' 를 따로 돌려 승률을 비교하세요. (--bt-rule-xml 로 강제 지정하면 양쪽이\n"
            "  같은 rule 로 도는 거울 대결이 됩니다.)"
        )
    return rules.pop() if rules else None


def _apply_bt_rule_env() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--bt-rule-xml", default="")
    pre.add_argument("--ownship-backend", default="rl")
    pre.add_argument("--target-backend", default="bt")
    pre.add_argument("--ownship-bt-dll", default=_DEF_OWNSHIP_BT)
    pre.add_argument("--target-bt-dll", default=_DEF_TARGET_BT)
    known, _ = pre.parse_known_args()
    rule = _resolve_bt_rule(known)
    if rule:
        os.environ[ENV_KEY] = rule
        print(f"[power_test] BT rule XML = {rule} ({ENV_KEY})")


_apply_bt_rule_env()   # ← 반드시 아래 import 들보다 먼저!

import numpy as np  # noqa: E402

from claude_code.env_utils import STANDARD_ENV_CONFIG  # noqa: E402
from claude_code.parallel import physical_cpu_count  # noqa: E402


# ── 승패 판정 ────────────────────────────────────────────────────────────────
# ownship 관점. 대칭 규칙이라 "어느 쪽이 강한가" 측정에 적합하다.
#   격추/피격추, 추락(고도 하락)은 그대로 승/패.  (프레임워크 _classify_outcome 은 표적
#   추락을 draw 로 두는데, 여기선 상대가 땅에 박은 것이므로 승으로 센다.)
#   그 외(타임아웃 등)는 최종 체력 비교, 같으면 무승부.
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
    """이항 비율의 Wilson 95% 신뢰구간(정규근사보다 소표본에서 정확)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_test_two_sided(k: int, n: int) -> float:
    """H0: p=0.5 양측 이항검정 exact p-value (n 이 100 규모라 정확 계산으로 충분)."""
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
        """env + provider 를 한 번만 만들고 여러 판을 순차로 돌린다(JSBSim init 이 비쌈)."""

        def __init__(self, spec: dict):
            os.chdir(spec["root"])
            import torch

            from claude_code.env_utils import make_env

            torch.set_num_threads(1)
            self.spec = spec
            self.step_ratio = int(STANDARD_ENV_CONFIG.get("step_ratio", 6))

            # ownship 번들의 관측 모듈을 env 에 주입(학습/제출과 동일 관측 재구성).
            self.env = make_env(overrides=spec["overrides"],
                                reward_module="",
                                observation_module=spec["obs_module"],
                                runner_index=f"pt{spec['wid']}")

            # ownship rl: MLPActionProvider(항상 stochastic) + action_repeat=step_ratio (10Hz).
            self.own_provider = None
            if spec["ownship_backend"] == "rl":
                from claude_code.action_provider import MLPActionProvider
                from claude_code.evaluate import _ActionRepeatProvider
                inner = MLPActionProvider(bundle_dir=spec["ownship_bundle_dir"],
                                          stochastic=True)
                self.own_provider = _ActionRepeatProvider(inner, self.step_ratio)
                self.env._ownship_action_provider = self.own_provider
            elif spec["ownship_backend"] == "altguard":
                # 복합 에이전트는 매 substep 호출을 받아 내부에서 제어 주기를 맞추므로
                # _ActionRepeatProvider 로 감싸지 않고 직접 주입한다(basic 10Hz / MPC 60Hz).
                from claude_code.altguard_provider import AltGuardMPCProvider
                self.own_provider = AltGuardMPCProvider(
                    spec["ownship_bundle_dir"], mpc_root=spec["altguard_mpc_root"],
                    mpc_config_path=(spec["altguard_mpc_config"] or None),
                    step_ratio=self.step_ratio, device="cpu", stochastic=True,
                    guard_altitude_ft=spec["altguard_alt_ft"])
                self.env._ownship_action_provider = self.own_provider

            # target rl: SelfPlayProvider(자체 reconstructor). MLPActionProvider 는 전역
            # _RECON 싱글톤을 쓰는데 ownship 도 같은 걸 쓰므로 상대 관점 HP 가 깨진다.
            self.tgt_provider = None
            if spec["target_backend"] == "bt" and spec["target_bt_mode"] == "provider":
                # 학습 opponent pool 과 **완전히 같은** BT 호출 경로(BTActionProvider →
                # sim.step). env 내장 behavior_tree 경로(sim.step_behavior)와는 throttle
                # 인자가 미세하게 달라 결과가 달라질 수 있어 따로 고를 수 있게 뒀다.
                from claude_code.self_play import make_bt_provider
                self.tgt_provider = make_bt_provider(spec["target_bt_dll"], spec["bt_rule"])
                self.env._target_action_provider = self.tgt_provider
            elif spec["target_backend"] == "rl":
                # 기본은 stochastic(pool 다양성·기존 동작). --target-rl-deterministic 면 argmax
                # (결정론 상대. 예: BT 를 복제한 clone 을 원본 BT 처럼 결정론으로 평가할 때).
                tgt_explore = bool(spec.get("target_rl_explore", True))
                # --target-rl-high-rate 면 target rl(예: exe 를 복제한 clone)만 60Hz(매 substep)로
                # 굴린다. 60Hz exe 를 대신하는 clone 을 원본처럼 60Hz 반응성으로 쓰기 위함.
                if spec.get("target_rl_high_rate"):
                    from claude_code.high_rate import high_rate_from_bundle
                    self.tgt_provider = high_rate_from_bundle(
                        spec["target_bundle_dir"], step_ratio=self.step_ratio,
                        device="cpu", explore=tgt_explore)
                else:
                    from claude_code.model import load_bundle
                    from claude_code.normalizers import RunningMeanStd
                    from claude_code.self_play import SelfPlayProvider
                    m, meta = load_bundle(spec["target_bundle_dir"], device="cpu")
                    rms = (RunningMeanStd.from_state_dict(meta["obs_normalization"])
                           if meta.get("obs_normalization") else None)
                    self.tgt_provider = SelfPlayProvider(
                        m, rms, self.env._observation_fn, self.env._observation_mode,
                        self.step_ratio, "cpu", explore=tgt_explore)
                self.env._target_action_provider = self.tgt_provider
            elif spec["target_backend"] == "gylee":
                # 팀원 model2_gylee 패키지의 self-contained opponent(10Hz). 자체 47D 관측·RMS·
                # reconstructor 를 가지므로 env 관측 모듈과 무관하게 동작한다(제공 계약대로
                # context.ownship_state=상대 자신, target_state=본 기체를 받는다).
                from model2_gylee import make_opponent_provider
                self.tgt_provider = make_opponent_provider(
                    snapshot_path=spec["gylee_snapshot"], step_ratio=self.step_ratio,
                    device="cpu", explore=bool(spec["gylee_explore"]),
                    verify_checksum=False)
                self.env._target_action_provider = self.tgt_provider

        def play(self, jobs):
            import torch

            out = []
            zero = np.zeros(4, dtype=np.float32)
            for seed, swap, head_swap in jobs:
                # 정책 샘플링은 torch RNG → 판마다 다른 시드로 고정(재현 가능한 랜덤).
                torch.manual_seed(int(seed))
                # 북남(swap)·방향(head_swap) 배정을 여기서 정확히 제어(randomize_start_side 는 꺼둠).
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
                    # provider 사용 시 action 인자는 무시되지만 시그니처상 필요.
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
                })
            return out

        def close(self):
            try:
                self.env.close()
            except Exception:
                pass

    return GameWorker


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="두 모델을 N판 붙여 어느 쪽이 강한지 통계로 판정(리플레이 없음)")
    p.add_argument("--ownship-backend", choices=["rl", "bt", "altguard"], default="rl",
                   help="ownship 종류. altguard = 평상시 basic 번들(10Hz stochastic) + 고도 "
                        "임계값 이하에서 team-share MPC(60Hz) 로 자동 전환하는 복합 에이전트")
    p.add_argument("--target-backend",
                   choices=["rl", "bt", "gylee", "loiter", "fixed", "autopilot"],
                   default="bt",
                   help="상대 종류. gylee = 팀원 model2_gylee 패키지의 고정 opponent(47D·19bin)")
    p.add_argument("--ownship-bundle-dir", help="ownship rl 일 때 claude 번들 경로")
    p.add_argument("--target-bundle-dir", help="target rl 일 때 claude 번들 경로")
    p.add_argument("--target-rl-deterministic", action="store_true",
                   help="target rl 을 argmax(결정론)로 굴린다. 기본은 stochastic(샘플링). "
                        "BT 를 복제한 clone 을 원본처럼 결정론으로 평가할 때 사용.")
    p.add_argument("--target-rl-high-rate", action="store_true",
                   help="target rl(예: 60Hz exe 를 복제한 exe_clone)만 **매 substep(60Hz)** 로 "
                        "결정시킨다. action-history 는 학습대로 0.1s 간격 유지(60Hz 큐 subsample). "
                        "ownship/기타 신경망 모델은 10Hz 그대로. clone 을 원본 exe(60Hz)처럼 쓸 때.")
    p.add_argument("--target-gylee-snapshot", default=_DEF_GYLEE_SNAPSHOT,
                   help=f"gylee opponent snapshot .pt (기본 {Path(_DEF_GYLEE_SNAPSHOT).name})")
    p.add_argument("--target-gylee-explore", action="store_true",
                   help="gylee opponent 를 stochastic(sample)으로. 기본은 deterministic(argmax) "
                        "— 고정 비교 상대로는 결정론이 해석하기 쉬움(model card §6).")
    p.add_argument("--ownship-bt-dll", default=_DEF_OWNSHIP_BT)
    p.add_argument("--target-bt-dll", default=_DEF_TARGET_BT)
    # ── altguard(고도 안전망) ownship 옵션 ──
    p.add_argument("--ownship-guard-altitude-ft", type=float, default=3000.0,
                   help="altguard: 이 고도(ft) 이하로 내려가면 MPC 가 대신 조종(기본 3000)")
    p.add_argument("--ownship-mpc-root", default=str(ROOT / "Release_MPC_team_share"),
                   help="altguard 이 쓸 team-share MPC 폴더(기본 Release_MPC_team_share)")
    p.add_argument("--ownship-mpc-config", default="",
                   help="altguard MPC config yaml(생략 시 <mpc-root>/configs/mpc.yaml)")
    p.add_argument("--bt-rule-xml", default="",
                   help="BT rule XML(기본: DLL 별 자동 선택. Lee_BT1.dll/Jeon_BT1.dll/"
                        "Jeon_BT2.dll/Shin_BT1.dll → 각 동명 .xml)")
    p.add_argument("--target-bt-mode", choices=["behavior_tree", "provider"],
                   default="behavior_tree",
                   help="target bt 를 부르는 경로. behavior_tree(기본) = env 내장 경로"
                        "(run_local_dogfight/리플레이와 동일), provider = 학습 opponent "
                        "pool 과 동일한 BTActionProvider 경로")
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
    p.add_argument("--min-altitude", type=float, default=None,
                   help="생략하면 env 기본값")
    p.add_argument("--csv", default="", help="판별 결과를 CSV 로 저장할 경로(선택)")
    return p.parse_args()


def _bt_to_behavior_tree(mode: str) -> str:
    return "behavior_tree" if mode == "bt" else mode


def main():
    args = parse_args()

    # rule XML 은 import 전에 세팅됐다. 전체 args 로 다시 계산해 어긋나면 실패시킨다
    # (조용히 Task_Empty BT 와 싸워서 가짜 승률을 보는 사고 방지).
    bt_rule = _resolve_bt_rule(args) or ""
    if bt_rule != os.environ.get(ENV_KEY, ""):
        raise RuntimeError(
            f"BT rule XML 불일치: 최종={bt_rule!r} / import 시점={os.environ.get(ENV_KEY)!r}. "
            "DLL 은 import 시점 값을 캐싱하므로 --bt-rule-xml 을 명시해 주세요.")

    if args.ownship_backend in ("rl", "altguard") and not args.ownship_bundle_dir:
        raise ValueError(f"--ownship-backend {args.ownship_backend} 이면 "
                         "--ownship-bundle-dir(basic 번들) 이 필요합니다.")
    if args.ownship_backend == "altguard" and not Path(args.ownship_mpc_root).is_dir():
        raise ValueError(f"altguard MPC 폴더를 찾을 수 없습니다: {args.ownship_mpc_root}")
    if args.target_backend == "rl" and not args.target_bundle_dir:
        raise ValueError("--target-backend rl 이면 --target-bundle-dir 이 필요합니다.")
    if args.target_backend == "gylee" and not Path(args.target_gylee_snapshot).is_file():
        raise ValueError(
            f"--target-backend gylee 의 snapshot 을 찾을 수 없습니다: {args.target_gylee_snapshot}")

    # 관측 모듈: env 는 하나뿐이라 양쪽 rl 번들이 같은 모듈을 써야 한다.
    from claude_code.model import load_bundle

    obs_modules = {}
    for side, d in (("ownship", args.ownship_bundle_dir
                     if args.ownship_backend in ("rl", "altguard") else None),
                    ("target", args.target_bundle_dir if args.target_backend == "rl" else None)):
        if d:
            _, meta = load_bundle(d, device="cpu")
            obs_modules[side] = meta.get("observation_module", "") or ""
    if len(set(obs_modules.values())) > 1:
        raise ValueError(f"두 번들의 observation_module 이 다릅니다: {obs_modules}. "
                         "같은 관측 모듈로 학습한 번들끼리만 비교할 수 있습니다.")
    obs_module = next(iter(obs_modules.values()), "")

    # provider 로 조종하는 상대(rl / bt-provider)는 env 가 자체 AI 를 만들지 않도록 fixed.
    provider_target = (args.target_backend in ("rl", "gylee")
                       or (args.target_backend == "bt"
                           and args.target_bt_mode == "provider"))
    overrides = {
        # target 이 rl/bt 가 아니면 스크립트 상대(loiter/fixed/autopilot).
        "target_mode": ("fixed" if provider_target
                        else _bt_to_behavior_tree(args.target_backend)),
        "target_behavior_dll": args.target_bt_dll,
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "randomize_start_side": False,     # 아래에서 정확히 50/50 로 직접 교대
    }
    if args.min_altitude is not None:
        overrides["min_altitude"] = args.min_altitude
    if args.ownship_backend == "bt":
        overrides["ownship_control_mode"] = "behavior_tree"
        overrides["ownship_behavior_dll"] = args.ownship_bt_dll

    games = max(1, int(args.games))
    n_workers = args.num_workers if args.num_workers > 0 else physical_cpu_count()
    n_workers = max(1, min(int(n_workers), games))

    master_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(master_seed)
    seeds = [int(x) for x in rng.integers(0, 2 ** 31 - 1, size=games)]
    # 북남(swap)·방향(head_swap)을 독립 교대: fixed_side 면 둘 다 고정(0),
    # 아니면 4가지 조합(북/남 × 90/270)이 균형 있게 돌게 한다(swap=i%2, head=(i//2)%2).
    swaps = [0] * games if args.fixed_side else [i % 2 for i in range(games)]
    head_swaps = [0] * games if args.fixed_side else [(i // 2) % 2 for i in range(games)]

    if args.ownship_backend == "rl":
        own_desc = f"rl({args.ownship_bundle_dir})"
    elif args.ownship_backend == "altguard":
        own_desc = (f"altguard(basic={args.ownship_bundle_dir} 10Hz / "
                    f"MPC<{args.ownship_guard_altitude_ft:.0f}ft 60Hz)")
    else:
        own_desc = f"bt({args.ownship_bt_dll})"
    if args.target_backend == "rl":
        tgt_desc = f"rl({args.target_bundle_dir})"
    elif args.target_backend == "bt":
        tgt_desc = f"bt({args.target_bt_dll}, {args.target_bt_mode})"
    elif args.target_backend == "gylee":
        tgt_desc = (f"gylee({Path(args.target_gylee_snapshot).name}, "
                    f"{'stochastic' if args.target_gylee_explore else 'deterministic'})")
    else:
        tgt_desc = args.target_backend
    print(f"[power_test] ownship = {own_desc}")
    print(f"[power_test] target  = {tgt_desc}")
    print(f"[power_test] games={games} workers={n_workers} master_seed={master_seed} "
          f"rl=stochastic obs_module={obs_module or '(env default)'}")
    print(f"[power_test] max_engage={args.max_engage_time}s step_limit={args.episode_step_limit} "
          f"시작위치={'고정' if args.fixed_side else '좌우 교대(50/50)'}")

    import ray

    if not ray.is_initialized():
        pythonpath = os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
        env_vars = {"PYTHONPATH": pythonpath}
        rule_env = os.environ.get(ENV_KEY, "")
        if rule_env:
            # worker **프로세스 시작 시점**에 들어가 있어야 DLL 이 읽는다(bt_rule 참고).
            env_vars[ENV_KEY] = rule_env
        ray.init(num_cpus=n_workers, include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=False, runtime_env={"env_vars": env_vars})

    spec = {
        "root": str(ROOT), "overrides": overrides, "obs_module": obs_module,
        "ownship_backend": args.ownship_backend, "target_backend": args.target_backend,
        "ownship_bundle_dir": args.ownship_bundle_dir,
        "target_bundle_dir": args.target_bundle_dir,
        "target_bt_mode": args.target_bt_mode, "target_bt_dll": args.target_bt_dll,
        "bt_rule": args.bt_rule_xml,
        "gylee_snapshot": args.target_gylee_snapshot,
        "gylee_explore": args.target_gylee_explore,
        "target_rl_explore": (not args.target_rl_deterministic),
        "target_rl_high_rate": bool(args.target_rl_high_rate),
        "altguard_mpc_root": str(args.ownship_mpc_root),
        "altguard_mpc_config": str(args.ownship_mpc_config),
        "altguard_alt_ft": float(args.ownship_guard_altitude_ft),
    }
    WorkerCls = _make_worker_cls()
    workers = [WorkerCls.remote({**spec, "wid": i}) for i in range(n_workers)]

    # 판을 worker 에 라운드로빈 분배(좌우 교대가 worker 별로 치우치지 않게).
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
        print(f"  이항검정 p-value    = {p:.3g}  (H0: 두 모델 실력 동일)")
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
