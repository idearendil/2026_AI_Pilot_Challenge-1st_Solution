"""[claude_code] neural-MPC 제출 — 1~2초 lookahead(obs-WM + team01/basic actor/critic).

submission.py(단일 MLP 정책)와 동일한 UDP 경로를 쓰되, ActionProvider 를
`MPCActionProvider` 로 교체한다. 매 RL-step 에 obs-WM 로 H 스텝(기본 20=2초) rollout 을
굴려 critic value 가 가장 좋은 first-action 을 고른다.

전제:
  - GPU(CUDA) 필요 — plan_fast(CUDA-graph)로 실시간 추론.
  - wm_model_obs.pt(추론호환 obs-WM), team01_basic_ac.pt(actor/critic) 가 있어야 함.
    team01_basic_ac.pt 생성:
      python - <<'PY'
      from claude_code.model import load_bundle; import torch,numpy as np
      m,meta=load_bundle("artifacts/models/team01/basic")
      mm=meta["model"]
      mk=dict(obs_dim=int(meta["observation_size"]),act_dim=int(meta.get("action_size",4)),
              hidden=tuple(mm["hidden"]),activation=mm.get("activation","tanh"),
              critic_hidden=tuple(mm["critic_hidden"]),critic_activation=mm.get("critic_activation"),
              num_bins=int(mm["num_bins"]))
      on=meta["obs_normalization"]
      torch.save({"model_kwargs":mk,"state_dict":{k:v.cpu().numpy() for k,v in m.state_dict().items()},
                  "obs_rms":{"mean":np.asarray(on["mean"]),"var":np.asarray(on["var"])}},
                 "claude_code/models/wm/team01_basic_ac.pt")
      PY

사용법: BUNDLE_DIR/TEAM_NAME/SERVER_IP/H 설정 후  python claude_code/submission_mpc.py
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
# TODO: 아래 설정을 팀에 맞게 수정하세요.
# =============================================================================
TEAM_NAME = "team01"
SERVER_IP = "221.151.77.208"
SERVER_PORT = 9999

WM_CKPT = str(ROOT / "claude_code/models/wm/wm_model_obs.pt")
AC_CKPT = str(ROOT / "claude_code/models/wm/team01_basic_ac.pt")
DEVICE = "cuda"                 # plan_fast(CUDA-graph)로 실시간 추론
H = 10                          # lookahead 스텝(10=1초). 매 RL-step 현재 state 로 재계획
K, M = 12, 8                    # 후보/상대샘플 수(B=K*M rollout)
DECIDE_EVERY = 1                # actor 재결정 주기(WM 스텝). 2 로 올리면 빨라짐(1/60s 목표 시)

OBSERVATION_MODE = "claude164r"
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6              # step_ratio=6 과 맞춤(6 sim frame 마다 새 plan 호출)


def main():
    print(f"=== [claude_code] {TEAM_NAME} neural-MPC 제출 시작 ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}  | H={H}(={H*0.1:.1f}s) K={K} M={M} device={DEVICE}")

    for p in (WM_CKPT, AC_CKPT):
        if not Path(p).exists():
            raise FileNotFoundError(f"체크포인트 없음: {p}")

    action_provider = MPCActionProvider(WM_CKPT, AC_CKPT, device=DEVICE, K=K, M=M, H=H,
                                        decide_every=DECIDE_EVERY)

    obs_module = "claude_code.my_observation"
    observation_hook = load_observation_hook(obs_module)
    obs_mode = observation_hook["mode"] if observation_hook else OBSERVATION_MODE

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
