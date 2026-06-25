"""[claude_code] 경진대회 제출 — Unreal 대결 서버 연결.

원본 `student/my_submission.py` 와 동일한 UDP 클라이언트 경로
(`UnrealAIPilotUDPClient` + `ProviderCommandPolicy`)를 사용하되, RLlib 번들 대신
claude_code PPO 번들(`metadata.json` + `policy_weights.pkl.gz`)을 로드하는
`MLPActionProvider` 를 연결한다. 대결 서버 입장에서는 원본 RL 제출과 완전히
동일하게 동작한다.

사용법
------
1) 먼저 학습:
     python claude_code/train.py --output-name team01 --output-tag ppo_mlp_v1
2) 아래 BUNDLE_DIR / TEAM_NAME / SERVER_IP 설정 후 실행:
     python claude_code/submission.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient

from claude_code.action_provider import MLPActionProvider


# =============================================================================
# TODO: 아래 설정을 팀에 맞게 수정하세요.
# =============================================================================

TEAM_NAME = "team01"                                   # TODO: 팀 이름
SERVER_IP = "221.151.77.208"                           # TODO: 경진대회 서버 IP
SERVER_PORT = 9999

# 학습으로 만든 claude_code PPO 번들 경로 (metadata.json + policy_weights.pkl.gz)
BUNDLE_DIR = "artifacts/models/team01/ppo_mlp_v1"      # TODO: 학습된 모델 경로
OBSERVATION_MODE = "tactical16"                        # 학습 시 관측 모드와 동일해야 함

# 연결 설정 (원본 my_submission.py 와 동일한 기본값)
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # 학습 step_ratio=6 과 맞춰 6개 PlaneInfo pair 마다 새 policy 호출
DEBUG_ACTION_REPEAT = False


# =============================================================================
# 로컬 검증 (제출 전 권장) — 원본 run_local_dogfight.py 는 RLlib 번들 backend 만
# 직접 지원하므로, claude_code 번들은 claude_code/evaluate.py 로 로컬 검증한다:
#   python claude_code/evaluate.py --bundle-dir artifacts/models/team01/ppo_mlp_v1
# =============================================================================


def main():
    print(f"=== [claude_code] {TEAM_NAME} 경진대회 클라이언트 시작 ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}")
    print(f"모드: claude_code PPO (MLP)")

    bundle_path = ROOT / BUNDLE_DIR if not Path(BUNDLE_DIR).is_absolute() else Path(BUNDLE_DIR)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}\n"
            f"먼저 claude_code/train.py 로 학습을 완료하고 BUNDLE_DIR 경로를 확인하세요."
        )

    print(f"[{TEAM_NAME}] PPO 모델 로드: {bundle_path}")
    action_provider = MLPActionProvider(bundle_dir=str(bundle_path))

    # 학습 때 custom 관측을 썼다면 동일 모듈을 로드해 서버 추론에도 같은 관측을 사용.
    obs_module = action_provider.metadata.get("observation_module", "") or ""
    observation_hook = load_observation_hook(obs_module) if obs_module else None
    obs_mode = observation_hook["mode"] if observation_hook else \
        (action_provider.metadata.get("observation_mode") or OBSERVATION_MODE)
    print(f"[{TEAM_NAME}] 관측: {obs_mode}"
          f"{' (custom: ' + obs_module + ')' if obs_module else ''}")

    command_policy = ProviderCommandPolicy(
        action_provider=action_provider,
        observation_mode=obs_mode,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=ACTION_REPEAT,
        debug_action_repeat=DEBUG_ACTION_REPEAT,
    )

    client = UnrealAIPilotUDPClient(
        command_policy=command_policy,
        server_ip=SERVER_IP,
        server_port=SERVER_PORT,
        team_name=TEAM_NAME,
        ai_type=AI_TYPE,
        heartbeat_interval_sec=HEARTBEAT_SEC,
        command_delay_sec=COMMAND_DELAY_SEC,
        recv_timeout_sec=RECV_TIMEOUT_SEC,
        enable_terminal_monitor=True,
    )

    try:
        client.run()
    finally:
        action_provider.close()
        print(f"[{TEAM_NAME}] 클라이언트 종료")


if __name__ == "__main__":
    main()
