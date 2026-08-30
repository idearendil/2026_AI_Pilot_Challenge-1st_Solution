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

from claude_code.model import make_actor_critic, save_bundle


def remap_state_dict(cuda_sd: dict, num_hidden: int) -> dict:
    """CUDA ActorCritic state_dict → MLPDiscreteActorCritic 키로 변환(값·shape 불변).

    CUDA body Sequential 의 Linear 는 index 0,2,..,2(L-1) 에 있고, 출력 Linear 는
    별도 attribute(actor_logits/critic_head). 제출용은 출력층이 같은 Sequential 의
    index 2L 에 온다. → body 는 prefix 만 교체, head 는 index 2L 로 이동.
      actor_body.k.*   -> actor_logits.k.*      (k 그대로)
      actor_logits.*   -> actor_logits.{2L}.*   (단독 head → Sequential 끝)
      critic_body.k.*  -> critic.k.*
      critic_head.*    -> critic.{2L}.*
    """
    head_idx = 2 * num_hidden
    out: dict = {}
    for k, v in cuda_sd.items():
        if k.startswith("actor_body."):
            out["actor_logits." + k[len("actor_body."):]] = v
        elif k.startswith("actor_logits."):          # 단독 head(weight/bias)
            out[f"actor_logits.{head_idx}." + k[len("actor_logits."):]] = v
        elif k.startswith("critic_body."):
            out["critic." + k[len("critic_body."):]] = v
        elif k.startswith("critic_head."):
            out[f"critic.{head_idx}." + k[len("critic_head."):]] = v
        else:
            raise KeyError(f"예상치 못한 CUDA state_dict 키: {k!r} "
                           f"(actor_body/actor_logits/critic_body/critic_head 만 지원)")
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

    # 구조 파라미터: cfg 우선, obs_dim/act_dim 은 가중치 shape 에서 역산(신뢰 가능).
    hidden = tuple(cfg.get("hidden", (256, 256)))
    activation = cfg.get("activation", "tanh")
    num_bins = int(cfg.get("num_bins", cuda_sd["actor_logits.weight"].shape[0] // 4))
    obs_dim = int(cuda_sd["actor_body.0.weight"].shape[1])
    act_dim = int(cuda_sd["actor_logits.weight"].shape[0] // num_bins)

    remapped = remap_state_dict(cuda_sd, num_hidden=len(hidden))

    model = make_actor_critic(obs_dim=obs_dim, act_dim=act_dim, num_bins=num_bins,
                              hidden=hidden, activation=activation)
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
          f"hidden={hidden} act_fn={activation} obs_norm={'yes' if obs_norm else 'no'}")
    print(f"  - observation_mode={obs_mode}")
    print("  - metadata.json / policy_weights.pkl.gz")


if __name__ == "__main__":
    main()
