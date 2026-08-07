"""[claude_code] 경진대회 제출 — Unreal 대결 서버 연결 (인스턴스 A).

로컬 DogFightViewer 대전용 2인스턴스 중 하나. submission_b.py 와 TEAM_NAME 만 다르고
같은 모델 번들을 쓴다(self-play 로컬 테스트). 대회 제출 시 SERVER_IP 를 원격 서버로 바꾼다.

사용법
------
1) 먼저 학습해서 번들 생성:
     python claude_code/train.py --output-name team01 --output-tag basic
2) DogFightViewer.exe 실행 → "서버 오픈" → 아래 두 명령을 각각 실행 → "시작":
     python claude_code/submission_a.py
     python claude_code/submission_b.py
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
# 설정 (인스턴스 A)
# =============================================================================

TEAM_NAME = "team01"                                   # 인스턴스 A 팀 이름 (B 와 달라야 함)
SERVER_IP = "127.0.0.1"                                # 로컬 DogFightViewer. 대회 땐 원격 IP(221.151.77.208) 로 교체.
SERVER_PORT = 9999

# 학습으로 만든 claude_code PPO 번들 경로 (metadata.json + policy_weights.pkl.gz)
BUNDLE_DIR = "artifacts/models/team01/basic"           # A·B 공용 모델 번들
OBSERVATION_MODE = "tactical16"                        # 학습 시 관측 모드와 동일해야 함

# 연결 설정 (원본 my_submission.py 와 동일한 기본값)
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # 학습 step_ratio=6 과 맞춰 6개 PlaneInfo pair 마다 새 policy 호출
DEBUG_ACTION_REPEAT = False
DEBUG_OBS = False           # 관측 규약 진단 로깅(초반 프레임 raw state/파생값). 진단 끝나면 False.


# =============================================================================
# 로컬 검증 (제출 전 권장) — 원본 run_local_dogfight.py 는 RLlib 번들 backend 만
# 직접 지원하므로, claude_code 번들은 claude_code/evaluate.py 로 로컬 검증한다:
#   python claude_code/evaluate.py --bundle-dir artifacts/models/team01/basic
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
    action_provider = MLPActionProvider(bundle_dir=str(bundle_path), debug_obs=DEBUG_OBS)

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
