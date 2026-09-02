"""Iteration snapshot 저장 + 'past-self' stochastic 평가.

매 iteration 의 actor network 를 가벼운 .pt snapshot 으로 저장하고(iter 0 = 학습
시작 직전의 초기 네트워크 포함), 5 iter 마다 **현재 정책 vs 5 iter 전 정책** 을
stochastic 으로 여러 판 대결시켜 진척을 측정한다.

- 평가 게임은 (가능하면) 멀티프로세스에서 돈다: 병렬 학습 모드면 기존 Ray
  rollout worker 를 재사용(`RolloutWorker.eval_games`), 단일 프로세스 모드면
  driver 에서 순차 실행된다.
- 상대는 과거 snapshot 의 network 로 `SelfPlayProvider` 를 통해 조종된다.
- 현재 정책과 상대 모두 stochastic(분포 샘플)으로 행동한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from claude_code.env_utils import STANDARD_ENV_CONFIG
from claude_code.model import discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd
from claude_code.self_play import SelfPlayProvider

OBS_CLIP = 10.0


# ── snapshot 직렬화 ──────────────────────────────────────────────────────────

def save_snapshot(path, model: MLPActorCritic, obs_rms, model_kwargs: dict) -> Path:
    """actor network(전체 state) + obs_rms 통계 + 구조를 .pt 로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_kwargs": dict(model_kwargs),
        "obs_rms": None if obs_rms is None else {
            "mean": np.asarray(obs_rms.mean, dtype=np.float64),
            "var": np.asarray(obs_rms.var, dtype=np.float64),
            "count": float(obs_rms.count),
        },
    }
    torch.save(data, str(path))
    return path


def load_snapshot(path):
    """snapshot 을 (state_dict, model_kwargs, obs_rms_dict) 로 읽는다."""
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    return data["state_dict"], data["model_kwargs"], data.get("obs_rms")


def rms_from_dict(rms_dict):
    """snapshot 의 obs_rms dict → RunningMeanStd 객체 (없으면 None)."""
    if rms_dict is None:
        return None
    mean = np.asarray(rms_dict["mean"], dtype=np.float64)
    r = RunningMeanStd(shape=mean.shape)
    r.mean = mean
    r.var = np.asarray(rms_dict["var"], dtype=np.float64)
    r.count = float(rms_dict["count"])
    return r


# ── 게임 플레이 / 집계 (단일·병렬 worker 공용) ───────────────────────────────

def _normalize(obs, mean, var):
    if mean is None:
        return np.asarray(obs, dtype=np.float32)
    n = (np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + 1e-8)
    return np.clip(n, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def make_opponent(env, opp_model, opp_rms_dict, stochastic: bool) -> SelfPlayProvider:
    """과거 snapshot network 로 상대를 조종하는 SelfPlayProvider 생성."""
    sr = int(STANDARD_ENV_CONFIG["step_ratio"])
    return SelfPlayProvider(
        opp_model, rms_from_dict(opp_rms_dict), env._observation_fn,
        env._observation_mode, sr, "cpu", explore=stochastic)


def play_games(env, model, cur_rms_mean, cur_rms_var, seeds, stochastic: bool,
               reconstruct: bool, device: str = "cpu"):
    """env(상대는 외부에서 이미 설정됨)에서 `model` 로 seeds 만큼 게임을 돈다.

    각 게임의 raw return / 길이 / terminal 보상 성분 + 최종 양측 체력(승패 판정용)을 반환.
    """
    if reconstruct:
        from claude_code.my_observation import (reset_reconstructor, advance_reconstructor,
                                                push_action_reconstructor)
    results = []
    for seed in seeds:
        if reconstruct:
            reset_reconstructor()
        o, _ = env.reset(seed=int(seed))
        if reconstruct:
            reset_reconstructor()
        # 에피소드 첫 관측은 GPU 학습과 동일하게 fresh recon(advance 안 함).
        done, ret, steps, terminal = False, 0.0, 0, 0.0
        own_hp, tgt_hp = 1.0, 1.0
        while not done:
            obs_n = _normalize(o, cur_rms_mean, cur_rms_var)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if stochastic:
                    a, _, _, _ = model.get_action_and_value(obs_t)
                    a = a.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    a = model.act_deterministic(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
            # 이산 정책이면 카테고리 index → 연속값으로 변환 후 env.step.
            env_a = discrete_indices_to_continuous(a, model.num_bins) if hasattr(model, "num_bins") else a
            if reconstruct:
                push_action_reconstructor(env_a)   # next obs 빌드(env.step) 전에 push
            o, r, term, trunc, info = env.step(env_a)
            if reconstruct:
                advance_reconstructor(env._ownship_state, env._target_state)
                o = env.get_observation()   # 0-lag: advance 뒤 관측 재빌드
            ret += float(r)
            steps += 1
            done = bool(term or trunc)
            if isinstance(info, dict):
                comp = info.get("ep_reward_components")
                if isinstance(comp, dict):
                    terminal = float(comp.get("terminal", terminal))
                # 게임 종료 시점의 양측 체력(체력 비교 승패 판정용).
                own_hp = float(info.get("ownship_health", own_hp))
                tgt_hp = float(info.get("target_health", tgt_hp))
        results.append({"return": ret, "length": steps, "terminal": terminal,
                        "own_hp": own_hp, "tgt_hp": tgt_hp})
    return results


def _game_outcome(r) -> str:
    """한 게임의 승패 판정 (ownship 관점).

    1) terminal 보상 성분(±10: 격추/추락 등 명확한 종료)이 있으면 그 부호로 판정.
    2) terminal=0(timeout 등 무승부)이면 **최종 체력 비교**: 체력이 더 많이 닳은 쪽이 패배.
       (격추는 HP=0 이라 1)에서 이미 처리되고, 추락은 HP 가 멀쩡해도 1)에서 패배로
       잡히므로, 체력 비교는 양쪽이 살아서 timeout 된 경우의 tiebreak 로만 쓰인다.)
    """
    if r["terminal"] > 1e-6:
        return "win"
    if r["terminal"] < -1e-6:
        return "loss"
    own_hp = r.get("own_hp")
    tgt_hp = r.get("tgt_hp")
    if own_hp is not None and tgt_hp is not None:
        if own_hp > tgt_hp + 1e-9:
            return "win"
        if own_hp < tgt_hp - 1e-9:
            return "loss"
    return "draw"


def summarize(results) -> dict:
    """게임 결과 리스트 → 평균 return / 승·패·무 / 승률.

    승패: terminal 부호 우선, 무승부면 최종 체력 비교(_game_outcome).
    """
    if not results:
        return {"mean_return": float("nan"), "mean_length": float("nan"),
                "win": 0, "loss": 0, "draw": 0, "win_rate": float("nan"), "n": 0}
    rets = [r["return"] for r in results]
    lens = [r["length"] for r in results]
    outcomes = [_game_outcome(r) for r in results]
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    n = len(results)
    return {
        "mean_return": float(np.mean(rets)),
        "mean_length": float(np.mean(lens)),
        "win": wins, "loss": losses, "draw": n - wins - losses,
        "win_rate": wins / n, "n": n,
    }


__all__ = [
    "save_snapshot", "load_snapshot", "rms_from_dict",
    "make_opponent", "play_games", "summarize",
]
