"""순수 PyTorch PPO (RLlib 의존 없음).

단일 DogFightWrapper 환경에서 on-policy rollout 을 모으고, GAE 로 advantage 를
계산한 뒤 clipped surrogate objective 로 정책을 업데이트한다. 학습 진행 상황을
iteration 단위로 출력하고(평균 episode return 등), 종료 시 2-파일 번들로 저장한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from claude_code.model import make_actor_critic, discrete_indices_to_continuous
from claude_code.normalizers import RunningMeanStd

OBS_CLIP = 10.0


def compute_gae(rewards, values, dones, last_value, last_done, gamma, gae_lambda):
    """GAE advantage / return 계산 (단일/병렬 수집 공용)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - float(last_done)
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


def rollout_outcome(info) -> Optional[str]:
    """rollout 의 done-step info 로 승패 판정(ownship 관점). info 없으면 None.

    evaluation._game_outcome 과 동일한 규칙:
      1) terminal 보상 성분(±10: 격추/추락) 부호로 우선 판정.
      2) terminal=0(timeout 등) 이면 최종 체력 비교로 tiebreak.
    """
    if not isinstance(info, dict):
        return None
    comp = info.get("ep_reward_components")
    terminal = float(comp.get("terminal", 0.0)) if isinstance(comp, dict) else 0.0
    if terminal > 1e-6:
        return "win"
    if terminal < -1e-6:
        return "loss"
    own_hp = float(info.get("ownship_health", 1.0))
    tgt_hp = float(info.get("target_health", 1.0))
    if own_hp > tgt_hp + 1e-9:
        return "win"
    if own_hp < tgt_hp - 1e-9:
        return "loss"
    return "draw"


def _outcome_counts(outcomes) -> dict:
    """승패 리스트 → {win, loss, draw, decided, raw_win_rate} (IterationStats.extra 용)."""
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")
    n = wins + losses + draws
    return {
        "win": wins, "loss": losses, "draw": draws, "decided": n,
        "raw_win_rate": (wins / n) if n > 0 else float("nan"),
    }


ALT_TERM_OWNSHIP = "ownship altitude below min"   # 우리 기체 고도 하락 종료 end_condition


def _count_altitude_terms(end_conditions) -> int:
    """완료 episode 의 end_condition 목록에서 '우리 기체 고도 하락' 종료 횟수."""
    return sum(1 for c in end_conditions if c == ALT_TERM_OWNSHIP)


def _outcome_counts_by_opp(indices, outcomes) -> dict:
    """(opponent index, 승패) → {index: {win, loss, draw, decided, raw_win_rate}}.

    opponent pool 각 후보별로 이번 iteration rollout 게임 승패를 집계한다. index 는
    샘플 당시 pool 위치(0=가장 오래된 후보). 병렬 수집에서도 모든 worker 가 동일
    순서의 pool 을 broadcast 받으므로 index 의미가 일치한다.
    """
    by: dict = {}
    for idx, oc in zip(indices, outcomes):
        d = by.setdefault(int(idx), {"win": 0, "loss": 0, "draw": 0})
        if oc in d:
            d[oc] += 1
    for d in by.values():
        n = d["win"] + d["loss"] + d["draw"]
        d["decided"] = n
        d["raw_win_rate"] = (d["win"] / n) if n > 0 else float("nan")
    return by


@dataclass
class PPOConfig:
    total_iterations: int = 50
    rollout_steps: int = 2048      # iteration 당 환경 step 수
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 256
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05   # 초과 시 epoch 조기 종료 (None 이면 비활성)
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    log_std_init: float = -0.5    # (이산 정책에서는 미사용)
    num_bins: int = 7             # 각 행동 채널의 이산 카테고리 수
    # critic 을 actor 와 완전히 분리된 네트워크로: 구조/학습률 독립 설정 (None=actor 와 동일)
    critic_hidden: Optional[tuple] = None
    critic_activation: Optional[str] = None
    critic_lr: Optional[float] = None
    normalize_obs: bool = True          # 관측 running mean/std 정규화
    reconstruct_state: bool = False     # claude_code.my_observation HP 재구성 갱신
    seed: int = 0
    device: str = "cpu"


@dataclass
class IterationStats:
    iteration: int
    global_step: int
    mean_return: float
    mean_length: float
    completed_episodes: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    explained_variance: float
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


class PPOTrainer:
    def __init__(self, env, config: PPOConfig):
        self.env = env
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        self.model = make_actor_critic(
            obs_dim, act_dim,
            hidden=config.hidden,
            activation=config.activation,
            critic_hidden=config.critic_hidden,
            critic_activation=config.critic_activation,
            num_bins=config.num_bins,
        ).to(config.device)
        # actor 와 critic 을 각각 별도 optimizer 로 (완전 분리). critic_lr=None 이면 lr 공유.
        self.actor_lr0 = config.lr
        self.critic_lr0 = config.critic_lr if config.critic_lr is not None else config.lr
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=self.actor_lr0, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=self.critic_lr0, eps=1e-5)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_step = 0
        # 재개(resume)/pool 후보 재생성 시 동일 구조로 모델을 만들기 위한 kwargs.
        self._model_kwargs = dict(
            obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(config.hidden),
            activation=config.activation, num_bins=config.num_bins,
            critic_hidden=(tuple(config.critic_hidden) if config.critic_hidden else None),
            critic_activation=config.critic_activation)

        # 정규화기 (관측만 — 보상 스케일링 없음)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if config.normalize_obs else None

        # 상태 재구성 (HP 누적) 갱신 함수 (claude_code.my_observation 사용 시)
        self._reset_recon = self._advance_recon = None
        if config.reconstruct_state:
            from claude_code.my_observation import reset_reconstructor, advance_reconstructor
            self._reset_recon = reset_reconstructor
            self._advance_recon = advance_reconstructor
            self._reset_recon()

        # rollout 가로지르며 유지되는 환경 상태
        obs, _ = env.reset(seed=config.seed)
        if self._reset_recon is not None:
            self._reset_recon()
        self._next_obs = np.asarray(obs, dtype=np.float32)
        self._next_done = False
        self._ep_return = 0.0
        self._ep_len = 0

    def _normalize_obs(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        if update:
            self.obs_rms.update(obs)
        norm = (np.asarray(obs, dtype=np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(norm, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    # ── rollout 수집 ─────────────────────────────────────────────────────────
    def collect_rollout(self):
        cfg = self.cfg
        T = cfg.rollout_steps
        device = cfg.device

        obs_buf = np.zeros((T, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, self.act_dim), dtype=np.float32)
        logp_buf = np.zeros(T, dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)
        val_buf = np.zeros(T, dtype=np.float32)

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_components: list[dict] = []
        ep_outcomes: list[str] = []   # 각 완료 episode 의 승패("win"/"loss"/"draw")
        ep_opp_indices: list[int] = []  # 각 완료 episode 가 쓴 opponent 의 pool index
        ep_end_conditions: list[str] = []  # 각 완료 episode 의 종료 사유(end_condition)

        for t in range(T):
            norm_obs = self._normalize_obs(self._next_obs, update=True)
            obs_buf[t] = norm_obs
            done_buf[t] = float(self._next_done)

            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, _, value = self.model.get_action_and_value(obs_tensor)
            # action = 카테고리 index(저장용). env.step 에는 연속값으로 변환해 전달.
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            act_buf[t] = action_np
            logp_buf[t] = float(logp.item())
            val_buf[t] = float(value.item())

            env_action = discrete_indices_to_continuous(action_np, self.model.num_bins)
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = bool(terminated or truncated)
            # HP 재구성 갱신: 이번 RL-step 결과 state 로 1회 advance (obs 는 step 안에서
            # advance 전 HP 를 읽었으므로 추론 경로와 동일한 1-step lag).
            if self._advance_recon is not None:
                self._advance_recon(self.env._ownship_state, self.env._target_state)
            self.global_step += 1
            self._ep_return += float(reward)
            self._ep_len += 1
            rew_buf[t] = float(reward)   # raw 보상 그대로 GAE 에 사용 (스케일링 없음)

            if done:
                ep_returns.append(self._ep_return)
                ep_lengths.append(self._ep_len)
                ep_end_conditions.append(
                    str(info.get("end_condition", "")) if isinstance(info, dict) else "")
                comp = info.get("ep_reward_components")
                if isinstance(comp, dict):
                    ep_components.append(dict(comp))
                oc = rollout_outcome(info)
                if oc is not None:
                    ep_outcomes.append(oc)
                    prov = getattr(self.env, "_target_action_provider", None)
                    ep_opp_indices.append(int(getattr(prov, "last_index", 0)))
                self._ep_return = 0.0
                self._ep_len = 0
                # 새 에피소드 전에 HP=1 로 리셋 → env.reset 의 관측이 올바른 HP 로 빌드됨.
                if self._reset_recon is not None:
                    self._reset_recon()
                next_obs, _ = self.env.reset()

            self._next_obs = np.asarray(next_obs, dtype=np.float32)
            self._next_done = done

        # bootstrap value (통계 갱신 없이 정규화)
        with torch.no_grad():
            last_value = float(
                self.model.get_value(
                    torch.as_tensor(
                        self._normalize_obs(self._next_obs, update=False),
                        dtype=torch.float32, device=device,
                    ).unsqueeze(0)
                ).item()
            )

        adv_buf, ret_buf = self._compute_gae(rew_buf, val_buf, done_buf, last_value, self._next_done)

        batch = {
            "obs": torch.as_tensor(obs_buf, device=device),
            "actions": torch.as_tensor(act_buf, device=device),
            "logp": torch.as_tensor(logp_buf, device=device),
            "advantages": torch.as_tensor(adv_buf, device=device),
            "returns": torch.as_tensor(ret_buf, device=device),
            "values": torch.as_tensor(val_buf, device=device),
        }
        return (batch, ep_returns, ep_lengths, ep_components, ep_outcomes,
                ep_opp_indices, ep_end_conditions)

    def _compute_gae(self, rewards, values, dones, last_value, last_done):
        return compute_gae(rewards, values, dones, last_value, last_done,
                           self.cfg.gamma, self.cfg.gae_lambda)

    # ── 정책 업데이트 ────────────────────────────────────────────────────────
    def update(self, batch):
        cfg = self.cfg
        T = batch["obs"].shape[0]
        idx = np.arange(T)

        advantages = batch["advantages"]
        returns = batch["returns"]
        old_logp = batch["logp"]
        old_values = batch["values"]

        clip = cfg.clip_coef
        last_pl = last_vl = last_ent = last_kl = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            approx_kls = []
            for start in range(0, T, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_obs = batch["obs"][mb]
                mb_act = batch["actions"][mb]

                _, new_logp, entropy, new_value = self.model.get_action_and_value(mb_obs, mb_act)
                log_ratio = new_logp - old_logp[mb]
                ratio = log_ratio.exp()

                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (clip 없이 단순 MSE — value 가 큰 오차를 빠르게 따라가도록)
                value_loss = 0.5 * ((new_value - returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                # actor / critic 손실을 분리해 각자의 optimizer 로 독립 업데이트.
                actor_loss = policy_loss - cfg.ent_coef * entropy_loss
                critic_loss = cfg.vf_coef * value_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                (actor_loss + critic_loss).backward()   # 파라미터가 분리돼 각 net 에만 grad
                nn.utils.clip_grad_norm_(self.model.actor_parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.critic_parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())

                last_pl = float(policy_loss.item())
                last_vl = float(value_loss.item())
                last_ent = float(entropy_loss.item())

            last_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
            if cfg.target_kl is not None and last_kl > cfg.target_kl:
                break

        # explained variance
        y_pred = old_values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

        return last_pl, last_vl, last_ent, last_kl, explained_var

    # ── gated self-play: opponent pool (frozen deep-copy 후보들) ───────────────
    def _make_opp_provider(self, model, rms, explore: bool = True):
        """현재 env 설정에 맞는 frozen opponent SelfPlayProvider 생성."""
        from claude_code.self_play import SelfPlayProvider
        from claude_code.env_utils import STANDARD_ENV_CONFIG
        sr = int(STANDARD_ENV_CONFIG["step_ratio"])
        return SelfPlayProvider(
            model, rms, self.env._observation_fn, self.env._observation_mode,
            sr, self.cfg.device, explore=explore)

    def install_opponent_pool(self, pool_max: int = 5, explore: bool = True) -> None:
        """opponent pool 초기화: 후보 1개(= 현재 정책의 frozen deep-copy)로 시작.

        학습 시작 시(초기 후보=iter0) 1회 호출. 이후 pool_add_current 로 후보를 추가한다.
        env 의 target provider 를 PoolSelfPlayProvider 로 교체한다.
        """
        import copy
        from claude_code.self_play import PoolSelfPlayProvider
        self._pool_max = max(1, int(pool_max))
        self._opp_explore = bool(explore)
        m = copy.deepcopy(self.model).eval()
        rms = copy.deepcopy(self.obs_rms) if self.obs_rms is not None else None
        prov = self._make_opp_provider(m, rms, explore)
        self._pool_provider = PoolSelfPlayProvider([prov], [1.0], seed=self.cfg.seed)
        self.env._target_action_provider = self._pool_provider

    def pool_set_weights(self, weights) -> None:
        """opponent 샘플링 가중치 갱신(다음 episode reset 부터 적용). EMA 낮을수록 크게."""
        if getattr(self, "_pool_provider", None) is not None:
            self._pool_provider.set_weights(weights)

    def snapshot_current(self) -> dict:
        """현재 정책 weights + obs_rms 통계를 checkpoint 용 dict 로 반환(numpy)."""
        state = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        rms = None if self.obs_rms is None else {
            "mean": np.asarray(self.obs_rms.mean, dtype=np.float64),
            "var": np.asarray(self.obs_rms.var, dtype=np.float64),
            "count": float(self.obs_rms.count),
        }
        return {"state": state, "rms": rms}

    def _make_opp_provider_from(self, state, rms_dict, explore=True):
        """checkpoint 의 (state_dict, rms) 로 frozen opponent SelfPlayProvider 재생성."""
        m = make_actor_critic(**self._model_kwargs)
        m.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        m.eval()
        rms = None
        if rms_dict is not None and self.obs_rms is not None:
            rms = RunningMeanStd(shape=(self.obs_dim,))
            rms.mean = np.asarray(rms_dict["mean"], dtype=np.float64)
            rms.var = np.asarray(rms_dict["var"], dtype=np.float64)
            rms.count = float(rms_dict["count"])
        return self._make_opp_provider(m, rms, explore)

    def set_opponent_pool(self, entries, weights, pool_max, explore=True) -> None:
        """checkpoint 의 opponent pool 전체를 그대로 복원(단일 프로세스).

        entries: [{"state": state_dict(np), "rms": {mean,var,count}|None}, ...] (오래된→최신).
        """
        from claude_code.self_play import PoolSelfPlayProvider
        self._pool_max = max(1, int(pool_max))
        self._opp_explore = bool(explore)
        provs = [self._make_opp_provider_from(e["state"], e.get("rms"), explore) for e in entries]
        if not provs:  # 방어: 최소 1개(현재 정책)
            snap = self.snapshot_current()
            provs = [self._make_opp_provider_from(snap["state"], snap["rms"], explore)]
        self._pool_provider = PoolSelfPlayProvider(provs, weights, seed=self.cfg.seed)
        self.env._target_action_provider = self._pool_provider

    def pool_add_current(self) -> int:
        """현재 정책+obs_rms 의 frozen deep-copy 를 pool 에 추가(초과 시 가장 오래된 후보 제거).

        pool 구성이 바뀌므로 진행 중이던 rollout episode 를 폐기하고 env 를 리셋해
        다음 episode 부터 새 구성으로 opponent 를 샘플/귀속하게 한다.
        """
        import copy
        m = copy.deepcopy(self.model).eval()
        rms = copy.deepcopy(self.obs_rms) if self.obs_rms is not None else None
        prov = self._make_opp_provider(m, rms, getattr(self, "_opp_explore", True))
        providers = list(self._pool_provider.providers)
        providers.append(prov)
        if len(providers) > self._pool_max:
            providers.pop(0)
        self._pool_provider.set_pool(providers)
        self._reset_rollout_env()
        return len(providers)

    def _reset_rollout_env(self) -> None:
        if self._reset_recon is not None:
            self._reset_recon()
        obs, _ = self.env.reset()
        if self._reset_recon is not None:
            self._reset_recon()
        self._next_obs = np.asarray(obs, dtype=np.float32)
        self._next_done = False
        self._ep_return = 0.0
        self._ep_len = 0

    # ── past-self stochastic 평가 (단일 프로세스) ────────────────────────────
    def evaluate_vs(self, env, opp_state, opp_model_kwargs, opp_rms_dict,
                    n_games, stochastic, base_seed):
        """`env` 에서 현재 정책 vs 과거 snapshot(opp_*) 으로 n_games 판 평가.

        병렬 모드가 아닐 때 driver 에서 순차 실행된다. `env` 의 상대를 과거
        network 로 교체해 게임을 돌리고 끝나면 원래 상대를 복원한다.
        """
        from claude_code import evaluation
        opp_model = make_actor_critic(**opp_model_kwargs)
        opp_model.load_state_dict({k: torch.as_tensor(v) for k, v in opp_state.items()})
        opp_model.eval()

        prev = getattr(env, "_target_action_provider", None)
        env._target_action_provider = evaluation.make_opponent(
            env, opp_model, opp_rms_dict, stochastic)
        mean = self.obs_rms.mean if self.obs_rms is not None else None
        var = self.obs_rms.var if self.obs_rms is not None else None
        seeds = [base_seed + i for i in range(n_games)]
        try:
            results = evaluation.play_games(
                env, self.model, mean, var, seeds, stochastic,
                self.cfg.reconstruct_state, self.cfg.device)
        finally:
            env._target_action_provider = prev
            # 평가는 rollout env 와 공유되는 reconstructor singleton 을 건드리므로,
            # rollout 연속성을 위해 singleton + rollout env 상태를 새로 리셋한다
            # (진행 중이던 rollout episode 는 폐기 — eval 주기마다 1회).
            if self._reset_recon is not None:
                self._reset_recon()
            obs, _ = self.env.reset()
            if self._reset_recon is not None:
                self._reset_recon()
            self._next_obs = np.asarray(obs, dtype=np.float32)
            self._next_done = False
            self._ep_return = 0.0
            self._ep_len = 0
        return evaluation.summarize(results)

    # ── 메인 루프 ────────────────────────────────────────────────────────────
    def train(self, on_iteration: Optional[Callable[[IterationStats], None]] = None,
              start_iteration: int = 1):
        history: list[IterationStats] = []
        for it in range(int(start_iteration), self.cfg.total_iterations + 1):
            t0 = time.time()
            (batch, ep_returns, ep_lengths, ep_components,
             ep_outcomes, ep_opp_indices, ep_end_conditions) = self.collect_rollout()
            pl, vl, ent, kl, ev = self.update(batch)

            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
            comp_means: dict = {}
            if ep_components:
                for key in ("pursuit", "damage", "distance", "aim", "terminal", "safety", "step"):
                    vals = [c.get(key, 0.0) for c in ep_components]
                    comp_means[key] = float(np.mean(vals))
            comp_means.update(_outcome_counts(ep_outcomes))
            comp_means["per_opp"] = _outcome_counts_by_opp(ep_opp_indices, ep_outcomes)
            comp_means["alt_term"] = _count_altitude_terms(ep_end_conditions)
            stats = IterationStats(
                iteration=it,
                global_step=self.global_step,
                mean_return=mean_ret,
                mean_length=mean_len,
                completed_episodes=len(ep_returns),
                policy_loss=pl,
                value_loss=vl,
                entropy=ent,
                approx_kl=kl,
                explained_variance=ev,
                elapsed_sec=time.time() - t0,
                extra=comp_means,
            )
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats)
        return history


__all__ = ["PPOConfig", "PPOTrainer", "IterationStats", "compute_gae",
           "rollout_outcome", "_outcome_counts", "_outcome_counts_by_opp",
           "_count_altitude_terms"]
