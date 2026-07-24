# -*- coding: utf-8 -*-
"""claude_code REDQ 트랙 — discrete SAC + REDQ 스타일 off-policy 학습.

기존 PPO 파이프라인(claude_code/train.py, ppo.py, parallel.py)과 **완전히 독립**된
병행 트랙이다. env/관측/보상/self-play pool/번들 export 등 공용 인프라는 재사용하되
(claude_code.env_utils, self_play, model.save_bundle 등), 학습 알고리즘만 새로 얹는다.

엔트리포인트: claude_code/train_redq.py

이 트랙은 "env-step 은 비싸고(CPU 6코어가 천장) GPU 는 논다"는 비대칭을 노린다.
UTD(update-to-data) ratio 를 올려 같은 env-step 예산에서 gradient step 을 더 많이
밟는 게 핵심 레버다. GPU 환경(aip_gpu, torch cu13x)에서 돌리는 것을 전제로 한다.
"""
from __future__ import annotations

__all__ = ["config", "networks", "replay", "sac"]
