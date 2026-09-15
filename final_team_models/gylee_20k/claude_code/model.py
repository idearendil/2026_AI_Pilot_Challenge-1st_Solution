"""MLP actor-critic 정책과 2-파일 번들 직렬화.

번들 형식 (원본과 동일한 2-파일 규약):
  <bundle_dir>/
  ├── metadata.json          # 관측 모드, 구조 메타, throttle 변환 규칙
  └── policy_weights.pkl.gz  # gzip 압축된 정책 state_dict

원본 RLlib 번들은 RLModule state 를 담지만, claude_code 번들은 아래
`MLPActorCritic.state_dict()` 를 담는다. claude_code/submission.py 가 이
형식을 직접 읽으므로 대결 서버 연결에는 RLlib 가 전혀 필요 없다.
"""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

WEIGHTS_FILENAME = "policy_weights.pkl.gz"
METADATA_FILENAME = "metadata.json"

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(act())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """분리형 actor/critic MLP + 상태 독립 log_std 를 갖는 가우시안 정책.

    action 차원은 4 (roll, pitch, rudder, throttle), 모두 R 출력이며
    환경/추론 단계에서 roll/pitch/rudder 는 [-1, 1] 로 clip, throttle 은
    (a+1)/2 로 [0, 1] 변환된다 (DogFightEnv._to_sim_action 과 동일).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str = "tanh",
        log_std_init: float = -0.5,
        critic_hidden: Sequence[int] | None = None,
        critic_activation: str | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = list(hidden)
        self.activation = activation
        # critic 은 actor 와 완전히 독립된 네트워크(파라미터·구조 모두 별개).
        # critic_hidden/critic_activation 미지정 시 actor 와 같은 구조를 쓴다.
        self.critic_hidden = list(critic_hidden) if critic_hidden is not None else list(hidden)
        self.critic_activation = critic_activation if critic_activation is not None else activation
        self.actor_mean = _mlp(obs_dim, hidden, act_dim, activation)
        self.critic = _mlp(obs_dim, self.critic_hidden, 1, self.critic_activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.apply(self._init_weights)

    def actor_parameters(self):
        """actor(정책) 파라미터: actor_mean MLP + log_std."""
        return list(self.actor_mean.parameters()) + [self.log_std]

    def critic_parameters(self):
        """critic(가치) 파라미터: critic MLP (actor 와 공유 없음)."""
        return list(self.critic.parameters())

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        mean = self.actor_mean(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    # 탐험 표준편차가 발산/소멸하지 않도록 log_std 범위를 제한한다.
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def _dist(self, mean: torch.Tensor) -> torch.distributions.Normal:
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        """**actor 전용** forward: critic 을 태우지 않고 (log_prob, entropy) 만 계산한다.

        actor/critic 업데이트 루프가 분리돼 있어(ppo.py `update()`), actor 루프에서
        불필요한 critic forward/grad 를 만들지 않기 위한 경로다.
        """
        mean = self.actor_mean(obs)
        dist = self._dist(mean)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """샘플(or 평가)된 action, log_prob, entropy, value 반환."""
        mean, value = self.forward(obs)
        dist = self._dist(mean)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 분포 평균을 그대로 사용 (탐험 noise 없음)."""
        mean, _ = self.forward(obs)
        return mean

    @torch.no_grad()
    def act_stochastic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 학습 때와 동일하게 정책 분포에서 샘플링."""
        mean, _ = self.forward(obs)
        return self._dist(mean).sample()


# ── 이산(discrete) 행동 공간 ─────────────────────────────────────────────────
# roll/pitch/yaw/throttle 각 채널을 num_bins(기본 7)개의 균등 분할 카테고리로
# 이산화한다. 카테고리 index 는 make_action_grid 로 [-1,1] 의 연속값에 매핑되고,
# env.step / 추론 경로의 throttle 변환((a+1)/2)은 기존과 동일하게 적용된다.
# 홀수 bins 여야 가운데 index=(num_bins-1)/2 가 정확히 0.0(조종간 중립)에 떨어진다.
ACTION_BINS = 21


def make_action_grid(num_bins: int = ACTION_BINS, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """[low, high] 를 num_bins 개로 균등 분할한 격자값(양 끝 포함)."""
    return np.linspace(low, high, int(num_bins)).astype(np.float32)


def discrete_indices_to_continuous(action_idx, num_bins: int = ACTION_BINS) -> np.ndarray:
    """카테고리 index(0..num_bins-1) → [-1,1] 연속 행동값."""
    grid = make_action_grid(num_bins)
    idx = np.clip(np.asarray(action_idx).round().astype(np.int64), 0, int(num_bins) - 1)
    return grid[idx].astype(np.float32)


class MLPDiscreteActorCritic(nn.Module):
    """채널별 독립 Categorical 정책 + 완전 분리형 critic(가치) 네트워크.

    actor 는 (act_dim × num_bins) 로짓을 출력하고, 각 행동 채널을 독립적인
    Categorical 분포로 본다(전체 행동 = 4개 카테고리의 곱). log_prob/entropy 는
    채널별 값을 합산한다. critic 은 actor 와 파라미터를 공유하지 않는 별도 MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,
        num_bins: int = ACTION_BINS,
        hidden: Sequence[int] = (256, 256),
        activation: str = "tanh",
        critic_hidden: Sequence[int] | None = None,
        critic_activation: str | None = None,
        **_ignored,   # log_std_init 등 연속용 kwargs 를 무해하게 흡수
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_bins = int(num_bins)
        self.hidden = list(hidden)
        self.activation = activation
        self.critic_hidden = list(critic_hidden) if critic_hidden is not None else list(hidden)
        self.critic_activation = critic_activation if critic_activation is not None else activation
        self.actor_logits = _mlp(obs_dim, hidden, self.act_dim * self.num_bins, activation)
        self.critic = _mlp(obs_dim, self.critic_hidden, 1, self.critic_activation)
        self.apply(MLPActorCritic._init_weights)

    def actor_parameters(self):
        return list(self.actor_logits.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor_logits(obs).view(-1, self.act_dim, self.num_bins)
        return torch.distributions.Categorical(logits=logits)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        """**actor 전용** forward: critic 을 태우지 않고 (log_prob, entropy) 만 계산한다.

        actor/critic 업데이트 루프가 분리돼 있어(ppo.py `update()`), actor 루프에서
        불필요한 critic forward/grad 를 만들지 않기 위한 경로다. action 은 카테고리 index.
        """
        dist = self._dist(obs)
        action = action.long()
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """샘플(or 평가)된 카테고리 index, 합산 log_prob, 합산 entropy, value 반환.

        반환 action 은 **카테고리 index**(float 캐스팅). env.step 에 넣기 전에
        discrete_indices_to_continuous 로 연속값으로 변환해야 한다.
        """
        dist = self._dist(obs)
        if action is None:
            action = dist.sample()              # (B, act_dim) long
        else:
            action = action.long()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return action.float(), log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 채널별 argmax 카테고리 index 반환 (탐험 없음)."""
        logits = self.actor_logits(obs).view(-1, self.act_dim, self.num_bins)
        return logits.argmax(-1).float()        # (B, act_dim) index

    @torch.no_grad()
    def act_stochastic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 학습 때와 동일하게 채널별 Categorical 에서 샘플링."""
        return self._dist(obs).sample().float()  # (B, act_dim) index


class GRUDiscreteActorCritic(nn.Module):
    """Independent recurrent actor/critic: 512→512→GRU(256)→512."""

    is_recurrent = True

    def __init__(self, obs_dim: int, act_dim: int = 4, num_bins: int = ACTION_BINS,
                 hidden: Sequence[int] = (512, 512, 512), activation: str = "tanh",
                 critic_hidden: Sequence[int] | None = None,
                 critic_activation: str | None = None, gru_size: int = 256,
                 encoder_depth: int = 2,
                 **_ignored):
        super().__init__()
        self.obs_dim, self.act_dim, self.num_bins = int(obs_dim), int(act_dim), int(num_bins)
        self.hidden, self.activation = list(hidden), activation
        self.gru_size = int(gru_size)
        self.critic_hidden = list(critic_hidden) if critic_hidden is not None else list(hidden)
        self.critic_activation = critic_activation if critic_activation is not None else activation
        self.encoder_depth = int(encoder_depth)
        if self.encoder_depth not in (1, 2):
            raise ValueError("encoder_depth must be 1 or 2")
        encoder_widths = list(hidden[:self.encoder_depth])
        post_width = int(hidden[2]) if len(hidden) > 2 else 512
        critic_widths = list(self.critic_hidden[:self.encoder_depth])
        critic_post_width = (int(self.critic_hidden[2])
                             if len(self.critic_hidden) > 2 else 512)
        act_cls = _ACTIVATIONS[activation]
        encoder_layers: list[nn.Module] = []
        last = self.obs_dim
        for width in encoder_widths:
            encoder_layers.extend((nn.Linear(last, int(width)), act_cls()))
            last = int(width)
        self.actor_encoder = nn.Sequential(*encoder_layers)
        encoder_out = last
        self.actor_gru = nn.GRU(encoder_out, self.gru_size, batch_first=True)
        self.actor_post = nn.Sequential(nn.Linear(self.gru_size, post_width), act_cls())
        self.actor_head = nn.Linear(post_width, self.act_dim * self.num_bins)
        critic_layers: list[nn.Module] = []
        last = self.obs_dim
        critic_act = _ACTIVATIONS[self.critic_activation]
        for width in critic_widths:
            critic_layers.extend((nn.Linear(last, int(width)), critic_act()))
            last = int(width)
        self.critic_encoder = nn.Sequential(*critic_layers)
        self.critic_gru = nn.GRU(last, self.gru_size, batch_first=True)
        self.critic_post = nn.Sequential(
            nn.Linear(self.gru_size, critic_post_width), critic_act())
        self.critic_head = nn.Linear(critic_post_width, 1)
        self.apply(MLPActorCritic._init_weights)
        for recurrent in (self.actor_gru, self.critic_gru):
            for name, param in recurrent.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def actor_parameters(self):
        return (list(self.actor_encoder.parameters()) + list(self.actor_gru.parameters())
                + list(self.actor_post.parameters()) + list(self.actor_head.parameters()))

    def critic_parameters(self):
        return (list(self.critic_encoder.parameters()) + list(self.critic_gru.parameters())
                + list(self.critic_post.parameters()) + list(self.critic_head.parameters()))

    def initial_state(self, batch_size: int, device=None):
        device = device if device is not None else next(self.parameters()).device
        z = torch.zeros(1, int(batch_size), self.gru_size, device=device)
        return z, z.clone()

    @staticmethod
    def _mask_state(state, episode_start):
        if episode_start is None:
            return state
        keep = (1.0 - episode_start.float()).view(1, -1, 1)
        return state[0] * keep, state[1] * keep

    def _step_logits(self, obs, state=None, episode_start=None):
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        actor_h, critic_h = state
        if episode_start is not None:
            keep = (1.0 - episode_start.float()).view(1, -1, 1)
            actor_h = actor_h * keep
        encoded = self.actor_encoder(obs).unsqueeze(1)
        output, new_actor_h = self.actor_gru(encoded, actor_h)
        logits = self.actor_head(self.actor_post(output[:, 0])).view(-1, self.act_dim, self.num_bins)
        return logits, (new_actor_h, critic_h)

    def value_step(self, obs, state=None, episode_start=None):
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        actor_h, critic_h = state
        if episode_start is not None:
            keep = (1.0 - episode_start.float()).view(1, -1, 1)
            critic_h = critic_h * keep
        encoded = self.critic_encoder(obs).unsqueeze(1)
        output, new_critic_h = self.critic_gru(encoded, critic_h)
        value = self.critic_head(self.critic_post(output[:, 0])).squeeze(-1)
        return value, (actor_h, new_critic_h)

    def actor_step(self, obs, state=None, episode_start=None, action=None,
                   deterministic: bool = False):
        logits, new_state = self._step_logits(obs, state, episode_start)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if deterministic else dist.sample()
        else:
            action = action.long()
        return (action.float(), dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1), new_state)

    def evaluate_actions_sequence(self, obs, action, initial_state, episode_starts):
        # PPO 쪽에서 chunk를 episode 경계에 맞춰 자르므로 내부 reset이 없으면 GRU가
        # 전체 time axis를 한 번에 처리할 수 있다. 예전 Python 32-step loop는 GPU
        # kernel launch를 지나치게 많이 만들어 recurrent update의 주 병목이었다.
        if obs.shape[1] <= 1 or not bool(torch.any(episode_starts[:, 1:]).item()):
            actor_h, _ = initial_state
            if episode_starts.shape[1]:
                actor_h = actor_h * (1.0 - episode_starts[:, 0].float()).view(1, -1, 1)
            encoded = self.actor_encoder(obs)
            output, _ = self.actor_gru(encoded, actor_h)
            logits = self.actor_head(self.actor_post(output)).view(
                obs.shape[0], obs.shape[1], self.act_dim, self.num_bins)
            dist = torch.distributions.Categorical(logits=logits)
            action = action.long()
            return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)
        state, logps, entropies = initial_state, [], []
        for t in range(obs.shape[1]):
            _, lp, ent, state = self.actor_step(
                obs[:, t], state, episode_starts[:, t], action[:, t])
            logps.append(lp)
            entropies.append(ent)
        return torch.stack(logps, 1), torch.stack(entropies, 1)

    def evaluate_actions_packed_sequence(self, obs, action, actor_h, episode_start):
        """Evaluate an episode-aligned padded sequence without a host sync.

        The optimized PPO packer guarantees that no valid sequence contains an
        episode boundary after timestep zero.  Padding follows all valid steps,
        so running the GRU through padded zeros cannot affect a valid output.
        Keeping this contract explicit avoids the ``torch.any(...).item()``
        branch used by the compatibility path above.
        """
        if episode_start is not None:
            actor_h = actor_h * (1.0 - episode_start.float()).view(1, -1, 1)
        encoded = self.actor_encoder(obs)
        output, _ = self.actor_gru(encoded, actor_h)
        logits = self.actor_head(self.actor_post(output)).view(
            obs.shape[0], obs.shape[1], self.act_dim, self.num_bins)
        dist = torch.distributions.Categorical(logits=logits)
        action = action.long()
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def evaluate_actions_stepwise_sequence(self, obs, action, actor_h, episode_starts):
        """Explicit compatibility path for the legacy padded-reset calculation.

        Unlike :meth:`evaluate_actions_sequence`, the caller has already
        determined on the CPU that a step-wise path is required, so this avoids
        a per-minibatch ``torch.any(...).item()`` synchronization.
        """
        dummy_critic_h = actor_h
        state, logps, entropies = (actor_h, dummy_critic_h), [], []
        for t in range(obs.shape[1]):
            _, lp, ent, state = self.actor_step(
                obs[:, t], state, episode_starts[:, t], action[:, t])
            logps.append(lp)
            entropies.append(ent)
        return torch.stack(logps, 1), torch.stack(entropies, 1)

    def evaluate_values_sequence(self, obs, initial_state, episode_starts):
        if obs.shape[1] <= 1 or not bool(torch.any(episode_starts[:, 1:]).item()):
            _, critic_h = initial_state
            if episode_starts.shape[1]:
                critic_h = critic_h * (1.0 - episode_starts[:, 0].float()).view(1, -1, 1)
            encoded = self.critic_encoder(obs)
            output, _ = self.critic_gru(encoded, critic_h)
            return self.critic_head(self.critic_post(output)).squeeze(-1)
        state, values = initial_state, []
        for t in range(obs.shape[1]):
            value, state = self.value_step(obs[:, t], state, episode_starts[:, t])
            values.append(value)
        return torch.stack(values, 1)

    def evaluate_values_packed_sequence(self, obs, critic_h, episode_start):
        """Value counterpart of :meth:`evaluate_actions_packed_sequence`."""
        if episode_start is not None:
            critic_h = critic_h * (1.0 - episode_start.float()).view(1, -1, 1)
        encoded = self.critic_encoder(obs)
        output, _ = self.critic_gru(encoded, critic_h)
        return self.critic_head(self.critic_post(output)).squeeze(-1)

    def evaluate_values_stepwise_sequence(self, obs, critic_h, episode_starts):
        """Explicit compatibility path for legacy padded critic sequences."""
        dummy_actor_h = critic_h
        state, values = (dummy_actor_h, critic_h), []
        for t in range(obs.shape[1]):
            value, state = self.value_step(obs[:, t], state, episode_starts[:, t])
            values.append(value)
        return torch.stack(values, 1)

    def get_value(self, obs):
        return self.value_step(obs)[0]

    def get_action_and_value(self, obs, action=None,
                             recurrent_state=None, episode_start=None):
        a, lp, ent, state = self.actor_step(obs, recurrent_state, episode_start, action)
        value, state = self.value_step(obs, state, episode_start)
        return a, lp, ent, value, state

    def evaluate_actions(self, obs, action):
        _, lp, ent, _ = self.actor_step(obs, action=action)
        return lp, ent

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor_step(obs, deterministic=True)[0]

    @torch.no_grad()
    def act_stochastic(self, obs):
        return self.actor_step(obs)[0]


def make_actor_critic(
    obs_dim: int,
    act_dim: int,
    hidden: Sequence[int] = (256, 256),
    activation: str = "tanh",
    critic_hidden: Sequence[int] | None = None,
    critic_activation: str | None = None,
    num_bins: int = ACTION_BINS,
    gru_size: int = 0,
    encoder_depth: int = 2,
    lstm_size: int = 0,
    **_ignored,
) -> nn.Module:
    """학습/추론 공용 정책 팩토리 (현재 기본 = 이산 행동 정책)."""
    if int(gru_size) > 0 or int(lstm_size) > 0:
        return GRUDiscreteActorCritic(
            obs_dim, act_dim, num_bins=num_bins, hidden=hidden, activation=activation,
            critic_hidden=critic_hidden, critic_activation=critic_activation,
            gru_size=gru_size or lstm_size, encoder_depth=encoder_depth)
    return MLPDiscreteActorCritic(
        obs_dim, act_dim, num_bins=num_bins, hidden=hidden, activation=activation,
        critic_hidden=critic_hidden, critic_activation=critic_activation)


class MLPDiscreteActor(nn.Module):
    """가치망 없는 **정책 전용** factorized categorical actor (REDQ/discrete SAC 용).

    MLPDiscreteActorCritic 과 동일한 정책 구조(채널 act_dim 개 × num_bins 카테고리,
    채널별 독립 softmax)를 갖지만 critic(가치망)이 없다. REDQ 트랙은 Q 앙상블을 별도
    모듈(claude_code.redq.networks)로 관리하므로 정책 번들에는 actor 만 담는다.

    추론 계약(act_deterministic / act_stochastic / num_bins)은 MLPDiscreteActorCritic
    과 동일해서, 기존 MLPActionProvider / load_bundle 이 그대로 이 actor 를 로드·구동한다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,
        num_bins: int = ACTION_BINS,
        hidden: Sequence[int] = (256, 256),
        activation: str = "tanh",
        **_ignored,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_bins = int(num_bins)
        self.hidden = list(hidden)
        self.activation = activation
        # save_bundle 메타 호환: critic_* 필드를 갖되 None 으로 둔다(가치망 없음).
        self.critic_hidden = None
        self.critic_activation = None
        self.actor_logits = _mlp(obs_dim, hidden, self.act_dim * self.num_bins, activation)
        self.apply(MLPActorCritic._init_weights)

    def actor_parameters(self):
        return list(self.actor_logits.parameters())

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, act_dim, num_bins) 로짓."""
        return self.actor_logits(obs).view(-1, self.act_dim, self.num_bins)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(obs))

    def probs_log_probs(self, obs: torch.Tensor):
        """discrete SAC 용: (probs, log_probs) 각각 (B, act_dim, num_bins).

        전체 카테고리 기대값 Σ_a π(a|s)(α logπ − Q) 을 채널별로 계산할 때 쓴다.
        log_softmax 로 수치 안정적으로 구한다.
        """
        logits = self.logits(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 채널별 argmax 카테고리 index (탐험 없음)."""
        return self.logits(obs).argmax(-1).float()

    @torch.no_grad()
    def act_stochastic(self, obs: torch.Tensor) -> torch.Tensor:
        """추론용: 채널별 Categorical 에서 샘플링."""
        return self._dist(obs).sample().float()

    @torch.no_grad()
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """SelfPlayProvider(explore) 호환 shim. 가치망이 없으므로 value=None.

        SelfPlayProvider 는 첫 반환값(카테고리 index)만 쓰므로 나머지는 None 이어도 된다.
        이 덕분에 REDQ actor(MLPDiscreteActor)를 기존 self-play pool 상대로 그대로 쓸 수 있다.
        """
        idx = self._dist(obs).sample().float() if action is None else action.float()
        return idx, None, None, None


# ── 환경/서버로 보낼 action 변환 ──────────────────────────────────────────────
# 학습 환경 DogFightEnv._to_sim_action 과 동일한 변환을 추론에서도 적용해
# train/inference action 의미를 일치시킨다.
SIM_ACTION_LOW = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
SIM_ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def make_obs_normalizer(obs_norm: dict | None, clip: float = 10.0):
    """번들 메타의 obs_normalization 으로 관측 정규화 함수를 만든다.

    None 이면 항등 함수를 반환한다. 학습 시 `ppo._normalize_obs` 와 동일한 식.
    """
    if not obs_norm:
        return lambda obs: np.asarray(obs, dtype=np.float32)
    mean = np.asarray(obs_norm["mean"], dtype=np.float64)
    var = np.asarray(obs_norm["var"], dtype=np.float64)
    inv_std = 1.0 / np.sqrt(var + 1e-8)

    def _normalize(obs: np.ndarray) -> np.ndarray:
        norm = (np.asarray(obs, dtype=np.float64) - mean) * inv_std
        return np.clip(norm, -clip, clip).astype(np.float32)

    return _normalize


def policy_action_to_command(raw_action: np.ndarray) -> np.ndarray:
    """정책 raw 출력(R^4) → 서버 CMD([roll,pitch,rudder]∈[-1,1], throttle∈[0,1])."""
    a = np.asarray(raw_action, dtype=np.float32).copy()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, -1.0, 1.0)
    a[3] = (a[3] + 1.0) / 2.0  # throttle [-1,1] → [0,1]
    return np.clip(a, SIM_ACTION_LOW, SIM_ACTION_HIGH)


# ── 번들 저장 / 로드 ─────────────────────────────────────────────────────────

def save_bundle(
    model: MLPActorCritic,
    output_dir: str | Path,
    obs_norm: dict | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(model, MLPDiscreteActor):
        model_type = "mlp_discrete_actor"          # 정책 전용(REDQ/SAC). 가치망 없음.
    elif isinstance(model, GRUDiscreteActorCritic):
        model_type = "gru_discrete_actor_critic"
    elif isinstance(model, MLPDiscreteActorCritic):
        model_type = "mlp_discrete_actor_critic"
    else:
        model_type = "mlp_actor_critic"
    model_meta = {
        "type": model_type,
        "hidden": model.hidden,
        "activation": model.activation,
        "critic_hidden": model.critic_hidden,
        "critic_activation": model.critic_activation,
    }
    if model_type != "mlp_actor_critic":
        model_meta["num_bins"] = model.num_bins
    if model_type == "gru_discrete_actor_critic":
        model_meta["gru_size"] = model.gru_size
        model_meta["encoder_depth"] = model.encoder_depth
    metadata = {
        "framework": "claude_code_ppo",
        "algorithm": "PPO",
        "observation_mode": "tactical16",
        "observation_size": model.obs_dim,
        "action_size": model.act_dim,
        "recurrent": bool(getattr(model, "is_recurrent", False)),
        "recurrent_type": ("gru" if isinstance(model, GRUDiscreteActorCritic) else None),
        "model": model_meta,
        "throttle_remap": "(a+1)/2",
        "action_clip": {"low": SIM_ACTION_LOW.tolist(), "high": SIM_ACTION_HIGH.tolist()},
        # 관측 정규화 통계 (추론에서 동일하게 적용). None 이면 정규화 미사용.
        "obs_normalization": obs_norm,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    (out / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    with gzip.open(out / WEIGHTS_FILENAME, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_bundle(bundle_dir: str | Path, device: str = "cpu") -> tuple[MLPActorCritic, dict]:
    bundle = Path(bundle_dir)
    metadata = json.loads((bundle / METADATA_FILENAME).read_text(encoding="utf-8"))
    with gzip.open(bundle / WEIGHTS_FILENAME, "rb") as fh:
        state = pickle.load(fh)

    model_meta = metadata.get("model", {})
    crit_hidden = model_meta.get("critic_hidden")
    common = dict(
        obs_dim=int(metadata.get("observation_size", 16)),
        act_dim=int(metadata.get("action_size", 4)),
        hidden=tuple(model_meta.get("hidden", (256, 256))),
        activation=model_meta.get("activation", "tanh"),
        critic_hidden=tuple(crit_hidden) if crit_hidden is not None else None,
        critic_activation=model_meta.get("critic_activation"),
    )
    mtype = model_meta.get("type")
    if mtype == "mlp_discrete_actor":
        # 정책 전용 actor(REDQ/SAC). critic 관련 kwargs 는 무시.
        model = MLPDiscreteActor(
            obs_dim=common["obs_dim"], act_dim=common["act_dim"],
            num_bins=int(model_meta.get("num_bins", ACTION_BINS)),
            hidden=common["hidden"], activation=common["activation"])
    elif mtype == "gru_discrete_actor_critic":
        model = GRUDiscreteActorCritic(
            num_bins=int(model_meta.get("num_bins", ACTION_BINS)),
            gru_size=int(model_meta.get("gru_size", 256)),
            encoder_depth=int(model_meta.get("encoder_depth", 2)),
            **common)
    elif mtype == "mlp_discrete_actor_critic":
        model = MLPDiscreteActorCritic(num_bins=int(model_meta.get("num_bins", ACTION_BINS)), **common)
    else:
        model = MLPActorCritic(**common)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, metadata


__all__ = [
    "MLPActorCritic",
    "MLPDiscreteActorCritic",
    "GRUDiscreteActorCritic",
    "MLPDiscreteActor",
    "make_actor_critic",
    "ACTION_BINS",
    "make_action_grid",
    "discrete_indices_to_continuous",
    "policy_action_to_command",
    "make_obs_normalizer",
    "save_bundle",
    "load_bundle",
    "WEIGHTS_FILENAME",
    "METADATA_FILENAME",
]
