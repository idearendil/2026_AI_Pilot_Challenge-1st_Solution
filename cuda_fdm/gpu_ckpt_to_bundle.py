# -*- coding: utf-8 -*-
"""CUDA(GPU) PPO 체크포인트(.pt) → 제출용 2-파일 번들(metadata.json + policy_weights.pkl.gz).

cuda_fdm/train_gpu.py 가 저장하는 체크포인트(PPOGPUTrainer.save)는 claude_code 의
snapshot 포맷과 두 군데가 다르다:

  ① 컨테이너: {"model","actor_opt","critic_opt","norm","pool","cfg",...} 로,
     claude_code.evaluation.load_snapshot 이 기대하는 {"state_dict","model_kwargs","obs_rms"}
     와 키가 다르다 → snapshot_to_bundle.py 로 바로 못 읽는다.
  ② state_dict 키 이름: CUDA ActorCritic 은 body(Sequential)+head(단독 Linear)로 분리돼
     `actor_body.*`/`actor_logits`/`critic_body.*`/`critic_head` 를 쓰고, 제출용
     MLPDiscreteActorCritic 은 출력층까지 한 Sequential 이라 `actor_logits.{0,2,..,2L}`/
     `critic.{0,2,..,2L}` 를 쓴다.

두 네트워크는 레이어 수·shape·활성화·연산이 100% 동일하고 파라미터가 1:1 대응하므로,
키 이름만 remap 하면 수치적으로 동일한(bit-identical) 정책이 제출 번들 포맷으로 나온다.
관측도 CUDA env 가 claude_code 와 같은 claude164r(OBS_SIZE=184)라 추론 측과 그대로 맞는다.

예:
  python -m cuda_fdm.gpu_ckpt_to_bundle \
      --ckpt runs/gpu.pt \
      --output-dir artifacts/models/team01/gpu_ppo_final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code.model import make_lstm_actor, save_bundle


def remap_state_dict(cuda_sd: dict) -> dict:
    """CUDA ActorCritic(LSTM) state_dict → 추론용 LSTMDiscreteActor 키로 변환(값·shape 불변).

    두 모듈은 actor trunk 파라미터 이름이 **완전히 동일**하다(`actor_lstm.*`, `actor_logits.*`).
    critic(critic_lstm/critic_head)·aux head(actor_aux_head/critic_aux_head)는 추론(제출)에서
    전혀 안 쓰므로 버린다. → actor_lstm.*/actor_logits.* 만 그대로 복사한다."""
    out: dict = {}
    for k, v in cuda_sd.items():
        if k.startswith("actor_lstm.") or k.startswith("actor_logits."):
            out[k] = v
        elif (k.startswith("critic_lstm.") or k.startswith("critic_head.")
              or k.startswith("actor_aux_head.") or k.startswith("critic_aux_head.")):
            continue                                 # critic·aux: 추론(제출) 미사용 → 버림
        else:
            raise KeyError(f"예상치 못한 CUDA state_dict 키: {k!r} (actor_lstm/actor_logits/"
                           "critic_lstm/critic_head/*_aux_head 만 지원)")
    return out


def obs_norm_from_ckpt(norm: dict | None) -> dict | None:
    """CUDA RunningNorm state_dict(torch mean/var/count) → 번들 obs_normalization dict.
    번들의 make_obs_normalizer 는 mean/var 를, rms_from_dict 는 count 까지 읽는다."""
    if not norm:
        return None
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return {
        "mean": _np(norm["mean"]).astype(np.float64).tolist(),
        "var": _np(norm["var"]).astype(np.float64).tolist(),
        "count": float(_np(norm["count"])),
    }


def _detect_obs_mode(observation_module: str) -> str:
    """관측 모듈에서 mode 문자열을 읽어 메타에 정확히 기록(실패 시 claude164r 폴백)."""
    if not observation_module:
        return "claude164r"
    try:
        from dogfight.ai.student_hooks import load_observation_hook
        return load_observation_hook(observation_module)["mode"]
    except Exception:
        try:
            import importlib
            m = importlib.import_module(observation_module)
            return getattr(m, "OBSERVATION_MODE", "claude164r")
        except Exception:
            return "claude164r"


def parse_args():
    p = argparse.ArgumentParser(description="CUDA PPO 체크포인트(.pt) → 제출 번들 변환")
    p.add_argument("--ckpt", required=True, help="변환할 CUDA 체크포인트(.pt) 경로")
    p.add_argument("--output-dir", required=True, help="번들 저장 디렉토리")
    p.add_argument("--observation-module", default="claude_code.my_observation",
                   help="학습 때 쓴 관측 모듈 경로(메타 기록; 추론이 동일 관측 재구성)")
    p.add_argument("--reward-module", default="claude_code.my_reward",
                   help="학습 때 쓴 보상 모듈 경로(메타 기록용)")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {ckpt_path}")

    # 자체 체크포인트(optimizer/cfg/pool 포함)라 weights_only=False (신뢰 소스). CPU 로 로드.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{ckpt_path} 는 CUDA PPOGPUTrainer 체크포인트가 아님('model' 키 없음)")
    cuda_sd = ckpt["model"]
    cfg = ckpt.get("cfg", {}) or {}

    # 구조 파라미터: obs_dim/act_dim/num_bins/lstm 은 가중치 shape 에서 역산(신뢰 가능).
    #   actor_lstm.weight_ih_l0: (4*H, obs_dim)  → obs_dim = shape[1], H = shape[0]//4
    #   actor_logits.weight:     (act_dim*num_bins, H)
    #   lstm_layers = weight_ih_l{k} 개수
    obs_dim = int(cuda_sd["actor_lstm.weight_ih_l0"].shape[1])
    lstm_hidden = int(cuda_sd["actor_lstm.weight_ih_l0"].shape[0] // 4)
    lstm_layers = sum(1 for k in cuda_sd if k.startswith("actor_lstm.weight_ih_l"))
    num_bins = int(cfg.get("num_bins", cuda_sd["actor_logits.weight"].shape[0] // 4))
    act_dim = int(cuda_sd["actor_logits.weight"].shape[0] // num_bins)

    remapped = remap_state_dict(cuda_sd)

    model = make_lstm_actor(obs_dim=obs_dim, act_dim=act_dim, num_bins=num_bins,
                            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)
    model.load_state_dict(remapped, strict=True)   # strict → 키/shape 완전 일치 검증(불일치 시 예외)
    model.eval()

    obs_norm = obs_norm_from_ckpt(ckpt.get("norm"))
    obs_mode = _detect_obs_mode(args.observation_module)
    it = int(ckpt.get("iteration", -1))

    out = Path(args.output_dir)
    save_bundle(
        model, out, obs_norm=obs_norm,
        extra_metadata={
            "reward_module": args.reward_module,
            "observation_module": args.observation_module,
            "observation_mode": obs_mode,
            "selected_iteration": it,
            "source": f"gpu_ckpt_to_bundle:{ckpt_path.name}",
            "trainer": "cuda_fdm.PPOGPUTrainer",
        },
    )
    print(f"[gpu_ckpt_to_bundle] iter {it} → 번들 저장 완료: {out}")
    print(f"  - from: {ckpt_path}")
    print(f"  - obs_size={obs_dim} act={act_dim} num_bins={num_bins} "
          f"lstm={lstm_layers}x{lstm_hidden} obs_norm={'yes' if obs_norm else 'no'}")
    print(f"  - observation_mode={obs_mode}")
    print("  - metadata.json / policy_weights.pkl.gz")


if __name__ == "__main__":
    main()
