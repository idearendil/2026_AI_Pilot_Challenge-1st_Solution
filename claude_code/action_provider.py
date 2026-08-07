"""claude_code MLP 정책을 원본 ActionProvider 계약으로 감싼다.

원본 `RLActionProvider` 와 동일한 인터페이스(`reset`, `compute_action`, `close`)를
구현하므로, 원본 `ProviderCommandPolicy` / `UnrealAIPilotUDPClient` 와 그대로
연결된다. 즉 대결 서버 입장에서는 원본 RL 제출과 완전히 동일하게 동작한다.

관측은 `ProviderCommandPolicy` 가 `build_observation("tactical16", ...)` 로
만들어 `context.observation` 에 넣어주며, 이는 학습 환경의 관측 파이프라인과
동일한 함수다.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

from claude_code.model import (
    load_bundle, make_obs_normalizer, policy_action_to_command, discrete_indices_to_continuous,
)


class MLPActionProvider(ActionProvider):
    def __init__(self, bundle_dir: str | Path, device: str = "cpu", confidence: float = 0.9,
                 stochastic: bool = False, debug_obs: bool | None = None):
        self.bundle_dir = str(bundle_dir)
        self.device = device
        self.confidence = confidence
        # True 면 학습 때와 동일하게 정책 분포에서 샘플링(리플레이 다양성용).
        # 제출/평가 기본값은 False(argmax).
        self.stochastic = bool(stochastic)
        self.model, self.metadata = load_bundle(bundle_dir, device=device)
        self.obs_dim = int(self.metadata.get("observation_size", 16))
        # 학습 때와 동일한 관측 정규화 적용 (없으면 항등).
        self._normalize_obs = make_obs_normalizer(self.metadata.get("obs_normalization"))
        # claude_code.my_observation 을 쓴 번들이면 HP/damage 재구성을 RL-step 당 1회
        # 갱신한다 (다른 관측 모듈에는 영향 없음).
        self._reconstruct = self.metadata.get("observation_module") == "claude_code.my_observation"
        if self._reconstruct:
            from claude_code.my_observation import (reset_reconstructor, advance_reconstructor,
                                                    push_action_reconstructor)
            self._reset_recon = reset_reconstructor
            self._advance_recon = advance_reconstructor
            self._push_action = push_action_reconstructor

        # ── 진단 로깅 ─────────────────────────────────────────────────────────
        # 서버 관측 규약(속도 좌표계·각도 단위·단위계)이 학습과 맞는지 첫 프레임들에서
        # raw state + 파생값을 찍는다. debug_obs=None 이면 환경변수 CLAUDE_OBS_DEBUG 로 켠다.
        if debug_obs is None:
            debug_obs = os.environ.get("CLAUDE_OBS_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
        self._debug_obs = bool(debug_obs)
        self._dbg_n = 0                                    # 지금까지 찍은 policy 프레임 수
        self._dbg_head = int(os.environ.get("CLAUDE_OBS_DEBUG_HEAD", "20"))   # 초반 매 프레임
        self._dbg_every = int(os.environ.get("CLAUDE_OBS_DEBUG_EVERY", "30")) # 이후 N프레임마다
        if self._debug_obs:
            print(f"[OBS_DEBUG] 활성화: obs_dim={self.obs_dim} "
                  f"module={self.metadata.get('observation_module')} "
                  f"mode={self.metadata.get('observation_mode')} "
                  f"(head={self._dbg_head}, every={self._dbg_every})", flush=True)

    def reset(self, context: ActionContext | None = None) -> None:
        # MLP 정책은 recurrent state 가 없으므로 reset 시 별도 처리 불필요.
        if self._reconstruct:
            self._reset_recon()
        if self._debug_obs:
            # reset 이 매 게임 시작마다 실제로 불리는지 확인용(안 불리면 t_sec 누적 → 규약 어긋남).
            print(f"[OBS_DEBUG] === reset() 호출됨 (게임 시작, reconstructor t_sec=0 으로 초기화) ===",
                  flush=True)
            self._dbg_n = 0

    def compute_action(self, context: ActionContext) -> ActionResult:
        # HP/damage 재구성을 RL-step 당 1회 갱신 (관측 빌드 전에 호출되어도 1-step lag 로
        # 학습 경로와 동일). context 에 양측 state 가 채워져 있을 때만.
        if (self._reconstruct and context.ownship_state is not None
                and context.target_state is not None):
            self._advance_recon(context.ownship_state, context.target_state)

        observation = context.observation
        if observation is None:
            raise ValueError(
                "MLPActionProvider 는 context.observation 이 필요합니다 "
                "(ProviderCommandPolicy 가 tactical16 관측을 채워줍니다)."
            )
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"관측 차원 불일치: got {obs.shape[0]}, expected {self.obs_dim}"
            )

        obs_prenorm = obs.copy()                          # 정규화 전 원본(진단용)
        obs = self._normalize_obs(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_fn = (self.model.act_stochastic if self.stochastic
                  else self.model.act_deterministic)
        raw = act_fn(obs_tensor).squeeze(0).cpu().numpy()
        # 이산 정책이면 카테고리 index → 연속값으로 변환.
        if hasattr(self.model, "num_bins"):
            raw = discrete_indices_to_continuous(raw, self.model.num_bins)
        # action history: 방금 결정한 action([-1,1]^4)을 reconstructor 에 push → 다음 관측이
        # 이 action 을 포함(학습 경로와 동일 규약). command 변환 전 raw 를 넣는다.
        if self._reconstruct:
            self._push_action(raw)
        command = policy_action_to_command(raw)  # [roll,pitch,rudder]∈[-1,1], throttle∈[0,1]

        if self._debug_obs:
            self._debug_dump(context, obs_prenorm, raw, command)

        return ActionResult(
            action=command,
            source="claude_code_ppo",
            confidence=self.confidence,
            info={"bundle_dir": self.bundle_dir},
        )

    def _debug_dump(self, context, obs_prenorm, raw_action, command) -> None:
        """서버 관측 규약 진단: raw state + 파생값(속도 좌표계·각도 단위·단위계)을 찍는다.

        초반 self._dbg_head 프레임은 매 프레임, 이후 self._dbg_every 프레임마다 1회.
        모두 ASCII 로만 출력(콘솔 cp949 인코딩 에러 방지).
        """
        n = self._dbg_n
        self._dbg_n += 1
        if not (n < self._dbg_head or (self._dbg_every > 0 and n % self._dbg_every == 0)):
            return

        own = np.asarray(context.ownship_state, dtype=np.float64)
        tgt = np.asarray(context.target_state, dtype=np.float64)

        def _frame_line(tag, s):
            nn, ee, dd = s[0], s[1], s[2]
            roll, pitch, yaw = s[3], s[4], s[5]
            u, v, w = s[6], s[7], s[8]
            alt = -dd
            spd = math.sqrt(u * u + v * v + w * w)
            sph = math.hypot(u, v)                 # 수평 성분
            # body-frame 가정 파생값
            aoa = math.degrees(math.atan2(w, u)) if spd > 1.0 else 0.0
            aos = math.degrees(math.atan2(v, math.hypot(u, w))) if spd > 1.0 else 0.0
            # NED 가정이면 (vel_n,vel_e) ~= sph*(cos hdg, sin hdg)
            hdg = math.radians(yaw)
            ned_pred = (sph * math.cos(hdg), sph * math.sin(hdg))
            print(f"[OBS_DEBUG]  {tag}: pos(N,E,D)=({nn:.1f},{ee:.1f},{dd:.1f}) alt=-D={alt:.1f}m | "
                  f"rot(roll,pit,yaw)=({roll:.2f},{pitch:.2f},{yaw:.2f}) | "
                  f"vel[6:9]=({u:.2f},{v:.2f},{w:.2f}) |vel|={spd:.1f}", flush=True)
            print(f"[OBS_DEBUG]      body가정 -> AOA={aoa:.1f}deg AOS={aos:.1f}deg "
                  f"(둘 다 작아야 정상; |v|,|w| 큼={abs(v):.1f},{abs(w):.1f}) | "
                  f"NED였다면 vel~=({ned_pred[0]:.1f},{ned_pred[1]:.1f})", flush=True)

        # 각도 단위 힌트: |roll|,|pitch|,|yaw| 가 <=~3.2 면 라디안 의심.
        att_max = float(np.max(np.abs(own[3:6])))
        deg_hint = "deg(정상범위)" if att_max > 6.5 else "RAD 의심(<=6.5)"
        dist = float(np.linalg.norm(own[:3] - tgt[:3]))

        info = context.info or {}
        print(f"[OBS_DEBUG] ---- frame#{info.get('frame_index','?')} policy_call={n} "
              f"my_id={info.get('my_plane_id','?')} tgt_id={info.get('target_plane_id','?')} "
              f"dist={dist:.1f}m att_scale={att_max:.2f}({deg_hint}) ----", flush=True)
        _frame_line("OWN ", own)
        _frame_line("ENMY", tgt)

        # reconstructor 상태(있으면): t_sec 가 프레임마다 +0.1 로 증가하는지, HP 가 살아있는지.
        if self._reconstruct:
            try:
                from claude_code.my_observation import get_reconstructor
                rec = get_reconstructor()
                print(f"[OBS_DEBUG]      recon: t_sec={float(rec.t_sec):.2f} "
                      f"hp_own={float(rec.hp_own):.3f} hp_tgt={float(rec.hp_tgt):.3f} "
                      f"pqr_own={np.round(np.asarray(rec.own_pqr_est, float), 3).tolist()}", flush=True)
            except Exception as e:
                print(f"[OBS_DEBUG]      recon 조회 실패: {e}", flush=True)

        # 관측 벡터 sanity: 포화(|x|>5)·NaN 개수, 범위.
        o = np.asarray(obs_prenorm, dtype=np.float64)
        n_sat = int(np.sum(np.abs(o) > 5.0))
        n_nan = int(np.sum(~np.isfinite(o)))
        print(f"[OBS_DEBUG]      obs(prenorm) dim={o.size} min={o.min():.2f} max={o.max():.2f} "
              f"mean={o.mean():.3f} |x|>5={n_sat} nan/inf={n_nan}", flush=True)
        print(f"[OBS_DEBUG]      action raw[roll,pit,rud,thr]={np.round(np.asarray(raw_action,float),3).tolist()} "
              f"-> cmd={np.round(np.asarray(command,float),3).tolist()}", flush=True)

    def close(self) -> None:
        return None


__all__ = ["MLPActionProvider"]
