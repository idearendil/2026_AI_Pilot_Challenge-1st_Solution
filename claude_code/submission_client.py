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
    config.json                 ← 서버 IP/포트, 팀명, 제어주기·argmax 여부, 번들 상대경로
    model/                      ← 학습 번들(metadata.json + policy_weights.pkl.gz)

CPU 학습 번들·CUDA 학습 번들 모두 같은 포맷이라 그대로 넣으면 된다. 주최측은 exe 만
실행하면 된다. side(ownship/target)는 지정하지 않는다 — 서버가 MT_SetPlaneID 로 조종할
기체를 배정하고, 클라이언트는 배정된 기체 기준으로 관측을 만들어 어느 편에 붙든 동일하게
동작한다.

config.json 필드
----------------
  server_ip        : 대회 서버 IP (예: "127.0.0.1" 로 로컬 BattleServer)
  server_port      : 서버 UDP 포트 (기본 9999)
  team_name        : 참가 팀 이름
  bundle_dir       : 학습 번들 경로(상대=config 기준).
  control_hz       : 10(action_repeat=6) | 60(action_repeat=1). 기본 10.
  deterministic    : true=argmax | false=stochastic(정책 분포 샘플링). 기본 false.
  action_repeat    : (선택) 정책 재호출 주기 직접 지정(생략 시 control_hz 로 결정).
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


# ── provider 구성 (순수 학습모델) ────────────────────────────────────────────
def _build_model(cfg: dict, base: Path):
    """CPU/CUDA 학습 번들 → provider. config 로 제어주기(10/60Hz)와 argmax/stochastic 지정.

    config.json 필드:
      bundle_dir     : 학습 번들 경로(상대=config 기준). CPU·CUDA 번들 모두 동일 포맷.
      control_hz     : 10(action_repeat=6) | 60(action_repeat=1). 기본 10.
      deterministic  : true=argmax | false=stochastic(정책 분포 샘플링). 기본 false.
    """
    from dogfight.ai.student_hooks import load_observation_hook

    bundle_dir = _resolve(base, cfg["bundle_dir"])
    if not bundle_dir.exists():
        raise FileNotFoundError(f"번들을 찾을 수 없습니다: {bundle_dir}")

    stochastic = not bool(cfg.get("deterministic", False))
    hz = int(cfg.get("control_hz", 10))
    # action_repeat: config 우선, 없으면 hz 로 결정(10Hz=6, 60Hz=1).
    action_repeat = int(cfg.get("action_repeat", 1 if hz == 60 else 6))

    if hz == 60:
        # 매 substep(60Hz) 재결정. HighRateProvider 가 context 상태로 스스로 관측을
        # 재구성하므로 ProviderCommandPolicy 의 obs 는 무시된다(claude164r 번들만 지원).
        from claude_code.high_rate import high_rate_from_bundle
        provider = high_rate_from_bundle(
            str(bundle_dir), step_ratio=int(cfg.get("step_ratio", 6)),
            device="cpu", explore=stochastic)
        obs_module = "claude_code.my_observation"
    else:
        from claude_code.action_provider import MLPActionProvider
        provider = MLPActionProvider(bundle_dir=str(bundle_dir), stochastic=stochastic)
        obs_module = provider.metadata.get("observation_module", "") or ""

    hook = load_observation_hook(obs_module) if obs_module else None
    obs_mode = hook["mode"] if hook else "claude164r"
    obs_fn = hook["build_observation"] if hook else None
    print(f"[{cfg['team_name']}] 순수 학습모델 (PPO MLP, 관측={obs_mode}, "
          f"{hz}Hz, {'argmax' if not stochastic else 'stochastic'})")
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
    for key in ("server_ip", "team_name", "bundle_dir"):
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

    print(f"=== [claude_code] {team_name} 경진대회 클라이언트 시작 ===")
    print(f"서버: {server_ip}:{server_port}  (config: {cfg_path})")

    provider, obs_mode, obs_fn, action_repeat = _build_model(cfg, base)

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
