"""[claude_code] neural-MPC 제출 — submission_a/b 와 같은 번들 + neural MPC (인스턴스 C).

submission_a.py 와 **같은 모델 번들**(BUNDLE_DIR)을 쓰되, 단일 MLP 정책 대신
`MPCActionProvider` 로 obs-WM 1~2초 lookahead 를 굴려 first-action 을 고른다.
submission_d.py 와 TEAM_NAME 만 다르다(로컬 self-play 테스트용 페어).

a/b 와 차이:
  - actor/critic 는 같은 번들에서 in-memory 로 로드(별도 *_ac.pt export 불필요).
  - obs-WM(wm_model_obs.pt) 로 H 스텝 rollout → critic value 최대 first-action 선택.
  - GPU(CUDA) 필요. 매 RL-step 재계획(plan_fast, CUDA-graph).

전제:
  - artifacts/models/team01/basic (a/b 공용 번들) + claude_code/models/wm/wm_model_obs.pt
  - 번들은 관측 정규화(obs_normalization)와 함께 저장돼 있어야 함(MPC 필수).

사용법: DogFightViewer.exe → "서버 오픈" → 아래 두 명령 각각 실행 → "시작":
     python claude_code/submission_c.py
     python claude_code/submission_d.py
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

from claude_code.mpc_action_provider import MPCActionProvider


# =============================================================================
# 설정 (인스턴스 C)
# =============================================================================

TEAM_NAME = "team01"                                   # 인스턴스 C 팀 이름 (D 와 달라야 함)
SERVER_IP = "127.0.0.1"                                # 로컬 DogFightViewer. 대회 땐 원격 IP(221.151.77.208) 로 교체.
SERVER_PORT = 9999

# a/b 와 **같은** PPO 번들 (metadata.json + policy_weights.pkl.gz)
BUNDLE_DIR = "artifacts/models/team01/basic"           # C·D 공용 모델 번들 (submission_a/b 와 동일)
OBSERVATION_MODE = "claude164r"                        # 번들 metadata 우선, 없으면 이 값

# neural-MPC 설정
WM_CKPT = str(ROOT / "claude_code/models/wm/wm_model_obs.pt")   # 추론호환 obs-WM
DEVICE = "cuda"                                        # plan_fast(CUDA-graph)로 실시간 추론
H = 5                                                  # lookahead 스텝(5=0.5초). 매 RL-step 재계획(서버 동시 구동 delay 완화 위해 10→5)
K, M = 8, 1                                             # 후보/상대샘플 수(B=K*M rollout). delay 완화 위해 12·8→8·1(B 96→8)
DECIDE_EVERY = 1                                       # actor 재결정 주기(WM 스텝). 2 로 올리면 빨라짐

# 연결 설정 (submission_a/b 와 동일한 기본값)
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # 학습 step_ratio=6 과 맞춰 6개 PlaneInfo pair 마다 새 plan 호출


def main():
    print(f"=== [claude_code] {TEAM_NAME} neural-MPC 제출 시작 (인스턴스 C) ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}  | H={H}(={H*0.1:.1f}s) K={K} M={M} device={DEVICE}")
    print(f"모드: claude_code neural-MPC (obs-WM + 번들 actor/critic)")

    bundle_path = ROOT / BUNDLE_DIR if not Path(BUNDLE_DIR).is_absolute() else Path(BUNDLE_DIR)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}\n"
            f"먼저 claude_code/train.py 로 학습을 완료하고 BUNDLE_DIR 경로를 확인하세요.")
    if not Path(WM_CKPT).exists():
        raise FileNotFoundError(f"world model 체크포인트 없음: {WM_CKPT}")

    print(f"[{TEAM_NAME}] 번들 actor/critic + obs-WM 로드: {bundle_path}")
    action_provider = MPCActionProvider(
        WM_CKPT, bundle_dir=str(bundle_path), device=DEVICE,
        K=K, M=M, H=H, decide_every=DECIDE_EVERY)

    # 번들 metadata 의 관측 모듈을 그대로 사용(a/b 와 동일 번들 → 동일 관측).
    import json
    meta = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    obs_module = meta.get("observation_module", "") or "claude_code.my_observation"
    observation_hook = load_observation_hook(obs_module) if obs_module else None
    obs_mode = observation_hook["mode"] if observation_hook else \
        (meta.get("observation_mode") or OBSERVATION_MODE)
    print(f"[{TEAM_NAME}] 관측: {obs_mode}"
          f"{' (custom: ' + obs_module + ')' if obs_module else ''}")

    command_policy = ProviderCommandPolicy(
        action_provider=action_provider,
        observation_mode=obs_mode,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=ACTION_REPEAT,
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
