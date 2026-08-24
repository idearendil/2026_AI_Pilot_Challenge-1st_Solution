# -*- coding: utf-8 -*-
"""[claude_code] 경진대회 제출 진입점 — config.json 기반 단일 실행 파일.

이 스크립트는 소스로도(`python claude_code/submission_client.py --config <dir>/config.json`),
PyInstaller 로 얼린 exe 로도 동작한다. 얼린 exe 는 **자기 옆(상대경로)** 의
`config.json` 을 읽어 모델 번들·MPC 자원을 로드한 뒤 대회 서버에 바로 접속한다.

제출 패키지 레이아웃(onedir)
---------------------------
  DogfightSubmission/
    DogfightSubmission.exe      ← 이 스크립트를 얼린 실행 파일
    _internal/                  ← PyInstaller 런타임(파이썬/torch 등)
    config.json                 ← 서버 IP/포트, 팀명, 모드, 자원 상대경로
    model/                      ← basic 번들(metadata.json + policy_weights.pkl.gz)
    Release_MPC_team_share/     ← altguard 저고도 MPC(예측기 DLL + f16 에셋 + configs)

주최측은 exe 만 실행하면 된다. side(ownship/target)는 지정하지 않는다 —
서버가 MT_SetPlaneID 로 조종할 기체를 배정하고, 클라이언트는 배정된 기체를
기준으로 관측을 만들어 어느 편에 붙든 동일하게 동작한다.

config.json 필드
----------------
  server_ip        : 대회 서버 IP (예: "221.151.77.208")
  server_port      : 서버 UDP 포트 (기본 9999)
  team_name        : 참가 팀 이름
  mode             : "altguard" | "basic"
  bundle_dir       : basic 번들 경로(상대=config 기준). 두 모드 공통 사용.
  mpc_root         : altguard 전용. Release_MPC_team_share 폴더 경로(상대).
  guard_altitude_ft: altguard 저고도 전환 임계(기본 3000).
  action_repeat    : 정책 재호출 주기(생략 시 altguard=1/60Hz, basic=6/10Hz).
  heartbeat_sec / recv_timeout_sec / command_delay_sec : 연결 튜닝(선택).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── 실행 위치·경로 해석 ────────────────────────────────────────────────────
def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _default_base() -> Path:
    """config.json 과 자원이 놓이는 기준 폴더."""
    if _frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _ensure_source_paths() -> None:
    """소스 실행 모드에서만 저장소 루트/ src 를 import 경로에 추가."""
    if _frozen():
        return
    root = Path(__file__).resolve().parents[1]
    for p in (root, root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


# ── 관측 더미(altguard 는 자체 관측을 만들므로 정책에 넘길 관측은 무시됨) ──
_DUMMY_OBS = np.zeros(1, dtype=np.float32)


def _dummy_obs(*_args, **_kwargs) -> np.ndarray:
    return _DUMMY_OBS


# ── provider 구성 ──────────────────────────────────────────────────────────
def _build_altguard(cfg: dict, base: Path):
    from claude_code.altguard_provider import make_altguard_provider

    bundle_dir = _resolve(base, cfg["bundle_dir"])
    mpc_root = _resolve(base, cfg.get("mpc_root", "Release_MPC_team_share"))
    if not bundle_dir.exists():
        raise FileNotFoundError(f"basic 번들을 찾을 수 없습니다: {bundle_dir}")
    if not mpc_root.exists():
        raise FileNotFoundError(f"MPC 자원 폴더를 찾을 수 없습니다: {mpc_root}")

    provider = make_altguard_provider(
        bundle_dir=str(bundle_dir),
        mpc_root=str(mpc_root),
        mpc_config_path=str(mpc_root / "configs" / "mpc.yaml"),
        step_ratio=int(cfg.get("step_ratio", 6)),
        device="cpu",
        stochastic=True,
        guard_altitude_ft=float(cfg.get("guard_altitude_ft", 3000.0)),
    )
    # altguard 는 매 substep(60Hz) 호출을 받아 내부에서 basic(10Hz)/MPC(60Hz)를
    # 스스로 분할한다 → 정책 쪽 action_repeat 는 1 이어야 한다.
    action_repeat = int(cfg.get("action_repeat", 1))
    print(f"[{cfg['team_name']}] 모드: altguard "
          f"(저고도 {cfg.get('guard_altitude_ft', 3000.0)}ft↓ → team-share MPC 60Hz)")
    return provider, "tactical16", _dummy_obs, action_repeat


def _build_basic(cfg: dict, base: Path):
    from claude_code.action_provider import MLPActionProvider
    from dogfight.ai.student_hooks import load_observation_hook

    bundle_dir = _resolve(base, cfg["bundle_dir"])
    if not bundle_dir.exists():
        raise FileNotFoundError(f"basic 번들을 찾을 수 없습니다: {bundle_dir}")

    provider = MLPActionProvider(bundle_dir=str(bundle_dir), stochastic=True)
    obs_module = provider.metadata.get("observation_module", "") or ""
    hook = load_observation_hook(obs_module) if obs_module else None
    obs_mode = hook["mode"] if hook else \
        (provider.metadata.get("observation_mode") or "tactical16")
    obs_fn = hook["build_observation"] if hook else None
    action_repeat = int(cfg.get("action_repeat", 6))
    print(f"[{cfg['team_name']}] 모드: basic (PPO MLP, 관측={obs_mode}"
          f"{', custom:' + obs_module if obs_module else ''})")
    return provider, obs_mode, obs_fn, action_repeat


# ── 메인 ───────────────────────────────────────────────────────────────────
def load_config(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"config.json 을 찾을 수 없습니다: {cfg_path}\n"
            f"exe 와 같은 폴더에 config.json 이 있어야 합니다."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("server_ip", "team_name", "mode", "bundle_dir"):
        if not cfg.get(key):
            raise ValueError(f"config.json 에 필수 항목 '{key}' 이(가) 없습니다: {cfg_path}")
    return cfg


def main() -> None:
    _ensure_source_paths()

    parser = argparse.ArgumentParser(description="claude_code 경진대회 제출 클라이언트")
    parser.add_argument("--config", default=None,
                        help="config.json 경로(생략 시 실행 파일 옆의 config.json)")
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config).resolve()
        base = cfg_path.parent
    else:
        base = _default_base()
        cfg_path = base / "config.json"

    cfg = load_config(cfg_path)

    from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient

    server_ip = str(cfg["server_ip"])
    server_port = int(cfg.get("server_port", 9999))
    team_name = str(cfg["team_name"])
    mode = str(cfg["mode"]).lower()

    print(f"=== [claude_code] {team_name} 경진대회 클라이언트 시작 ===")
    print(f"서버: {server_ip}:{server_port}  (config: {cfg_path})")

    if mode == "altguard":
        provider, obs_mode, obs_fn, action_repeat = _build_altguard(cfg, base)
    elif mode == "basic":
        provider, obs_mode, obs_fn, action_repeat = _build_basic(cfg, base)
    else:
        raise ValueError(f"알 수 없는 mode: {mode!r} (altguard 또는 basic)")

    command_policy = ProviderCommandPolicy(
        action_provider=provider,
        observation_mode=obs_mode,
        observation_fn=obs_fn,
        ownship_force_side=1,     # 고정 라벨일 뿐 — 진영은 서버가 배정한다.
        target_force_side=2,
        action_repeat=action_repeat,
        debug_action_repeat=False,
    )

    client = UnrealAIPilotUDPClient(
        command_policy=command_policy,
        server_ip=server_ip,
        server_port=server_port,
        team_name=team_name,
        ai_type=AIType.ReinforcementLearning,
        heartbeat_interval_sec=float(cfg.get("heartbeat_sec", 1.0)),
        command_delay_sec=float(cfg.get("command_delay_sec", 0.0)),
        recv_timeout_sec=float(cfg.get("recv_timeout_sec", 0.2)),
        enable_terminal_monitor=bool(cfg.get("terminal_monitor", True)),
    )

    try:
        client.run()
    finally:
        try:
            provider.close()
        except Exception:
            pass
        print(f"[{team_name}] 클라이언트 종료")


if __name__ == "__main__":
    main()
