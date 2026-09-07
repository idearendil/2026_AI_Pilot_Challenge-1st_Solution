# -*- coding: utf-8 -*-
"""GPU 네이티브 벡터화 PPO (GpuDogfightVecEnv 전용) + opponent pool 자기대전.

CPU 단일-env 용 claude_code/ppo.py 와 **동일한 학습 규약**을 GPU 벡터 env 위에서
전 과정 GPU 텐서로 재현한다(롤아웃·GAE·업데이트에 numpy 왕복이나 per-step .item()
동기화 없음; 로깅 sync 는 iteration 당 1회).

원본과 동일한 부분:
  - **관측** = claude164r(my_observation, 214dim; 가속도 30 추가) — 커널이 build_observation 과 비트일치.
  - **보상** = my_reward(MY_REWARD_CONFIG) — 커널이 compute_reward 와 비트일치.
  - **행동공간** = **discrete**: 4채널(roll/pitch/rudder/throttle) × num_bins(기본 21) 균등격자
    linspace(-1,1,21), 채널별 독립 Categorical(model.MLPDiscreteActorCritic 과 동일). 정책 raw
    action(∈[-1,1]^4)→서버 command 변환은 throttle 만 [-1,1]→[0,1](0.5z+0.5), 나머지는 그대로
    (claude_code.action_provider.policy_action_to_command 과 동일). CUDA FCS 는 throttle∈[0,1]
    (thr_pos=2·c_thr, 0.8→afterburner), roll/pitch/rudder∈[-1,1].
  - **초기 상태 분포** = STANDARD_ENV_CONFIG(시나리오 A/B, 고도 2000~30000ft, 속도 200~300m/s).
  - **action_repeat** = step_ratio 6(substeps=6).

opponent pool 자기대전(원본 gated self-play 재현):
  - env 기체0 = **main actor**(학습 정책), 기체1 = **opponent**(pool 에서 env 별 샘플된 frozen
    snapshot). **main 기체 전이만** 학습 버퍼에 담겨 opponent 데이터는 업데이트에서 자동 배제.
  - **EMA 승률 게이팅**: 각 opponent 엔트리는 '우리(main) 승률 EMA'(alpha=selfplay_ema_alpha)를
    가진다. **evictable(net) 후보들의 최소 EMA ≥ gate_threshold** 면 현재 main 을 새 evictable
    snapshot(EMA 0.5)으로 추가(초과 시 oldest evictable FIFO). 고정주기 추가는 없다.
  - **softmax 가중 샘플링**: p_i = f/m + (1-f)·softmax(-ema_i/τ) (f=uniform_floor, τ=sample_temp)
    — 우리 승률(EMA)이 낮은(어려운) 후보가 더 자주 뽑힌다. env 별 multinomial.
  - **milestone**(milestone_period): 그때 main 을 permanent(never-evict) 추가 + capacity +1, 이어
    **exploiter** 를 그때 main 유일 상대(frozen)로 scratch 학습(승률 target/max_iters) 후 permanent 추가.

autoreset 계약: truncation 은 γ·V(terminal_obs_main) fold 부트스트랩, terminated 는 경계(CleanRL).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

# aux 라벨(미래위치 잔차)을 아군 body 좌표계로 변환할 때 obs 와 동일한 회전을 쓴다.
from cuda_fdm.obs_reward import ned_to_body as _ned_to_body, _mv as _mv3

ACTION_BINS = 21   # 원본 train.py --action-bins 기본값과 동일(채널당 21 균등격자).
ACT_HIST_DIM = 20  # 관측 마지막 블록 = 과거 5-step × 4채널 action. critic 의 상대-행동 extra 도 동일.

# ── auxiliary 미래위치 예측 (표현학습) ────────────────────────────────────────
# 내/상대 전투기의 '미래 위치'를 예측하는 보조 task. 라벨 = 각자 현재 속도로 등속직선
# 비행했을 때의 baseline 위치와의 잔차벡터를, **아군(main) body 좌표계(현재 t)** 로 표현.
# 상대는 AUX_OPP_H step 뒤, 나는 AUX_SELF_H step 뒤를 예측. actor·critic 두 trunk 에 각각
# head 를 달아 표현학습(제출 시 actor aux head 는 잘라냄).
AUX_OPP_H = 5      # 상대 미래위치 예측 지평(step)
AUX_SELF_H = 10    # 내 미래위치 예측 지평(step)
AUX_POS_SCALE_M = 100.0   # 잔차 라벨 정규화[m] (예측/라벨 O(1) 스케일)
AUX_DIM = 6        # [상대 잔차(3), 내 잔차(3)]


def make_action_grid(num_bins=ACTION_BINS, device="cpu"):
    """[-1,1] 균등 num_bins 격자(양끝 포함). 홀수면 가운데=정확히 0(중립)."""
    return torch.linspace(-1.0, 1.0, int(num_bins), device=device)


# ── 관측 running 정규화 (GPU, 배치 병렬분산) ──────────────────────────────────
class RunningNorm:
    def __init__(self, dim, device, clip=10.0, eps=1e-8):
        self.dim = dim
        self.device = device
        self.mean = torch.zeros(dim, device=device, dtype=torch.float32)
        self.var = torch.ones(dim, device=device, dtype=torch.float32)
        self.count = torch.zeros((), device=device, dtype=torch.float32) + 1e-4
        self.clip = float(clip)
        self.eps = float(eps)

    @torch.no_grad()
    def update(self, x):
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        b_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + (delta * delta) * (self.count * b_count / tot)
        self.var = M2 / tot
        self.count = tot

    @torch.no_grad()
    def normalize(self, x):
        n = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return n.clamp_(-self.clip, self.clip)

    @torch.no_grad()
    def normalize_slice(self, x, s, e):
        """관측의 [s:e] 차원 통계로 x(...,e-s) 를 정규화. critic 전용 상대-행동 히스토리처럼
        관측의 특정 블록과 동일 분포인 값을 같은 통계로 정규화할 때 쓴다."""
        n = (x - self.mean[s:e]) / torch.sqrt(self.var[s:e] + self.eps)
        return n.clamp(-self.clip, self.clip)

    def state_dict(self):
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count.clone()}

    def load_state_dict(self, sd):
        self.mean.copy_(sd["mean"]); self.var.copy_(sd["var"]); self.count.copy_(sd["count"])

    def clone(self):
        c = RunningNorm(self.dim, self.device, self.clip, self.eps)
        c.load_state_dict(self.state_dict())
        return c


# ── actor-critic (discrete: 채널별 Categorical) ──────────────────────────────
def _layer_init(layer, gain=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(inp, hidden, act):
    A = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[act]
    layers = []
    last = inp
    for h in hidden:
        layers += [_layer_init(nn.Linear(last, h)), A()]
        last = h
    return nn.Sequential(*layers), last


class ActorCritic(nn.Module):
    """분리형 actor/critic MLP. actor=(act_dim×num_bins) 로짓 → 채널별 독립 Categorical.
    claude_code.model.MLPDiscreteActorCritic 과 동일한 정책 구조.

    critic_extra_dim>0 이면 critic 만 관측(obs_dim) 뒤에 추가 입력(critic_extra_dim)을 더
    받는다(actor 는 그대로 obs_dim). self-play 에서 critic 은 actor 가 볼 수 없는 정보(상대의
    과거 action 등)를 관측해도 무방하므로, 상대 과거 5-step action(20dim)을 critic 에만 준다."""

    def __init__(self, obs_dim, act_dim=4, num_bins=ACTION_BINS, hidden=(256, 256),
                 activation="tanh", critic_extra_dim=0, aux_dim=0):
        super().__init__()
        self.act_dim = act_dim
        self.num_bins = int(num_bins)
        self.critic_extra_dim = int(critic_extra_dim)
        self.aux_dim = int(aux_dim)
        self.actor_body, ah = _mlp(obs_dim, hidden, activation)
        self.actor_logits = _layer_init(nn.Linear(ah, act_dim * self.num_bins), gain=0.01)
        self.critic_body, ch = _mlp(obs_dim + self.critic_extra_dim, hidden, activation)
        self.critic_head = _layer_init(nn.Linear(ch, 1), gain=1.0)
        # aux 미래위치 예측 head(선택). actor/critic trunk 을 공유해 표현학습(제출 시 actor
        # aux head 는 사용 안 함 → 번들 변환에서 잘라냄). critic 은 추론에서 아예 미사용.
        if self.aux_dim > 0:
            self.actor_aux_head = _layer_init(nn.Linear(ah, self.aux_dim), gain=0.01)
            self.critic_aux_head = _layer_init(nn.Linear(ch, self.aux_dim), gain=0.01)

    def actor_parameters(self):
        ps = list(self.actor_body.parameters()) + list(self.actor_logits.parameters())
        if self.aux_dim > 0:
            ps += list(self.actor_aux_head.parameters())
        return ps

    def critic_parameters(self):
        ps = list(self.critic_body.parameters()) + list(self.critic_head.parameters())
        if self.aux_dim > 0:
            ps += list(self.critic_aux_head.parameters())
        return ps

    def _critic_in(self, obs, cext):
        """critic 입력 = obs (+ 상대-행동 extra). extra 없으면 obs 그대로."""
        if self.critic_extra_dim > 0 and cext is not None:
            return torch.cat([obs, cext], dim=-1)
        return obs

    def get_value(self, obs, cext=None, with_aux=False):
        feat = self.critic_body(self._critic_in(obs, cext))
        v = self.critic_head(feat).squeeze(-1)
        if with_aux and self.aux_dim > 0:
            return v, self.critic_aux_head(feat)      # (B,), (B,aux_dim)
        return v

    def _dist(self, obs):
        logits = self.actor_logits(self.actor_body(obs)).view(-1, self.act_dim, self.num_bins)
        return torch.distributions.Categorical(logits=logits)

    def get_action_and_value(self, obs, cext=None, action=None):
        dist = self._dist(obs)
        if action is None:
            action = dist.sample()                          # (B, act_dim) long
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs, cext)
        return action, logp, entropy, value

    def evaluate_actions(self, obs, action, with_aux=False):
        feat = self.actor_body(obs)
        logits = self.actor_logits(feat).view(-1, self.act_dim, self.num_bins)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        if with_aux and self.aux_dim > 0:
            return logp, ent, self.actor_aux_head(feat)   # 공유 trunk 1-pass
        return logp, ent, None

    @torch.no_grad()
    def act(self, obs, sample=True):
        """frozen opponent 용 행동 index(B,act_dim). sample=False 면 argmax(deterministic)."""
        dist = self._dist(obs)
        return dist.sample() if sample else dist.logits.argmax(-1)


def action_to_env(idx, num_bins, grid=None):
    """행동 index(N,4) → env control(N,4). 격자 linspace(-1,1,num_bins) 로 연속화한 뒤
    throttle 만 [-1,1]→[0,1](0.5z+0.5), roll/pitch/rudder 는 그대로([-1,1])."""
    if grid is None:
        grid = make_action_grid(num_bins, device=idx.device)
    cont = grid[idx.long()]                                  # (N,4) in [-1,1]
    ctrl = cont.clone()
    ctrl[:, 3] = 0.5 * cont[:, 3] + 0.5                      # throttle → [0,1]
    return ctrl


# ── opponent pool (EMA 게이팅 + softmax 가중 샘플링) ──────────────────────────
class OpponentPool:
    """frozen opponent snapshot pool. evictable(net, ≤evict_cap) + permanent(never-evict).
    각 엔트리 = {net, norm, permanent, ema}. ema=우리(main) 승률 EMA."""

    def __init__(self, model_kwargs, device, evict_cap=4, sample=True):
        self.model_kwargs = dict(model_kwargs)
        self.device = device
        self.evict_cap = int(evict_cap)
        self.sample = bool(sample)
        self.entries = []
        self._next_id = 0            # 엔트리 안정 id(위치와 무관). 진행 중 에피소드의 상대
        #                              identity 를 pool 구조변경(add/evict) 뒤에도 유지하기 위함.

    def size(self):
        return len(self.entries)

    def entry_ids(self):
        """현재 위치 순서의 엔트리 id 목록(위치→id). 구조변경 전후 매핑용."""
        return [e["id"] for e in self.entries]

    def num_permanent(self):
        return sum(1 for e in self.entries if e["permanent"])

    def capacity(self):
        return self.evict_cap + self.num_permanent()

    def slot_emas(self):
        """슬롯별 EMA 리포팅용. entries 는 insert 순서라 non-permanent 만 뽑으면 FIFO 순서
        (index 0=가장 먼저 들어온 evictable). 반환 (evict_emas, perm_emas):
          - evict_emas[i] = i 번째 evict 슬롯의 현재 opponent EMA. 새 opponent 가 들어와
            oldest 가 빠지면 슬롯 i 는 '다음으로 오래된 opponent' 로 바뀔 뿐 슬롯 수(≤evict_cap)
            는 안 늘어난다 → wandb 그래프가 opponent 마다 늘어나지 않고 슬롯 단위로 고정.
          - perm_emas[i] = i 번째 permanent(never-evict) opponent EMA(추가 순, 안정적 identity).
        EMA 계산 자체는 opponent 별(update_emas)로 그대로 유지, 여기선 리포팅 매핑만 한다."""
        evict_emas = [e["ema"] for e in self.entries if not e["permanent"]]
        perm_emas = [e["ema"] for e in self.entries if e["permanent"]]
        return evict_emas, perm_emas

    def _mk_net(self, model):
        net = ActorCritic(**self.model_kwargs).to(self.device)
        net.load_state_dict(copy.deepcopy(model.state_dict()))
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    @torch.no_grad()
    def add(self, model, norm, permanent, ema=0.5):
        entry = {"net": self._mk_net(model), "norm": norm.clone() if norm is not None else None,
                 "permanent": bool(permanent), "ema": float(ema), "id": self._next_id}
        self._next_id += 1
        self.entries.append(entry)
        # evictable(비영구) 초과 시 oldest evictable FIFO 제거.
        ev = [i for i, e in enumerate(self.entries) if not e["permanent"]]
        while len(ev) > self.evict_cap:
            self.entries.pop(ev[0])
            ev = [i for i, e in enumerate(self.entries) if not e["permanent"]]

    def update_emas(self, win_by_opp, loss_by_opp, ep_by_opp, alpha):
        """이번 iteration 의 per-opponent 결과로 각 엔트리 EMA 갱신.
        완료 에피소드가 하나라도 있으면(ep>0) 무승부를 0.5 로 반영해 갱신한다:
        frac = (승 + 0.5·무) / 완료 = (w + 0.5·(ep - w - l)) / ep.  ep==0(그 iter 에 이
        opponent 가 샘플링 안 됐거나 결판/무승부 모두 없음)일 때만 이전 EMA 를 유지한다.
        (예전엔 '결판(win|loss)>0' 일 때만 갱신 → 무승부만 난 iter 는 EMA 가 안 움직여
        그래프가 가로 일직선으로 남았음. 이제 무승부도 0.5 로 EMA 를 끌어당긴다.)"""
        w = win_by_opp.detach().cpu().numpy()
        l = loss_by_opp.detach().cpu().numpy()
        ep = ep_by_opp.detach().cpu().numpy()
        for i, e in enumerate(self.entries):
            if i >= len(w):
                break
            if ep[i] > 0:
                frac = (w[i] + 0.5 * (ep[i] - w[i] - l[i])) / ep[i]
                e["ema"] = float((1.0 - alpha) * e["ema"] + alpha * frac)

    def gate_and_add(self, model, norm, threshold):
        """evictable(net) 후보 최소 EMA ≥ threshold 면 현재 main 을 새 evictable 로 추가."""
        net_emas = [e["ema"] for e in self.entries if not e["permanent"]]
        if net_emas and min(net_emas) >= threshold:
            self.add(model, norm, permanent=False, ema=0.5)
            return True
        return False

    @torch.no_grad()
    def weights(self, temp, floor):
        """샘플 확률 p_i = f/m + (1-f)·softmax(-ema_i/τ) (전 엔트리). tensor(P,)."""
        emas = np.array([e["ema"] for e in self.entries], dtype=np.float64)
        m = emas.size
        if m <= 1:
            return torch.ones(max(m, 1), device=self.device) / max(m, 1)
        lg = -emas / max(temp, 1e-6)
        lg -= lg.max()
        w = np.exp(lg); w /= w.sum()
        p = floor / m + (1.0 - floor) * w
        p /= p.sum()
        return torch.as_tensor(p, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, opp_obs, assign):
        """opp_obs(nenv,OBS), assign(nenv,) → 행동 index(nenv,4). P forward 후 gather."""
        P = self.size()
        outs = []
        for e in self.entries:
            on = e["norm"].normalize(opp_obs) if e["norm"] is not None else opp_obs
            outs.append(e["net"].act(on, sample=self.sample))     # (nenv,4) long
        stacked = torch.stack(outs, 0)                            # (P,nenv,4)
        idx = assign.clamp(0, P - 1).view(1, -1, 1).expand(1, opp_obs.shape[0], 4)
        return stacked.gather(0, idx).squeeze(0)

    def state_dicts(self):
        out = []
        for e in self.entries:
            out.append({"model": {k: v.detach().cpu() for k, v in e["net"].state_dict().items()},
                        "norm": (e["norm"].state_dict() if e["norm"] is not None else None),
                        "permanent": e["permanent"], "ema": e["ema"]})
        return out

    def load_state_dicts(self, dicts):
        self.entries = []
        for d in dicts:
            net = ActorCritic(**self.model_kwargs).to(self.device)
            net.load_state_dict({k: torch.as_tensor(v) for k, v in d["model"].items()})
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            norm = None
            if d["norm"] is not None:
                norm = RunningNorm(self.model_kwargs["obs_dim"], self.device)
                norm.load_state_dict({k: torch.as_tensor(v, device=self.device)
                                      for k, v in d["norm"].items()})
            self.entries.append({"net": net, "norm": norm,
                                 "permanent": bool(d["permanent"]), "ema": float(d.get("ema", 0.5)),
                                 "id": self._next_id})
            self._next_id += 1


# ── config / stats ───────────────────────────────────────────────────────────
@dataclass
class PPOGPUConfig:
    total_iterations: int = 1000
    rollout_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    critic_lr: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    num_bins: int = ACTION_BINS
    hidden: tuple = (768, 768)
    activation: str = "tanh"
    # critic 에만 상대의 과거 5-step action(20dim)을 추가 입력으로 준다(actor 는 불변).
    critic_opp_actions: bool = True
    # auxiliary 미래위치 예측(actor·critic 두 trunk 에 head). aux_coef 로 MSE 를 각 loss 에 가산.
    aux_pred: bool = True
    aux_coef: float = 0.1
    normalize_obs: bool = True
    norm_adv: bool = True
    seed: int = 0
    device: str = "cuda"
    # ── iteration 스케줄 (sched_period iter 마다 단계 k=(it-1)//period 증가) ──────────
    # lr = base_lr·lr_decay^k, ent_coef = base_ent·ent_decay^k, rollout = base + increment·k.
    # gamma / (own_dmg·shaping 없음) 는 불변. sched_period<=0 이면 스케줄 비활성(값 고정).
    sched_period: int = 2000
    sched_lr_decay: float = 1.0 / 3.0
    sched_ent_decay: float = 1.0 / 3.0
    sched_rollout_increment: int = 8
    # ── opponent pool / gated self-play (원본과 동일 규약) ──────────────────────
    pool_evict_cap: int = 4               # evictable(net) snapshot 최대 수
    selfplay_gate_threshold: float = 0.6  # evictable 최소 EMA ≥ 이 값이면 snapshot 추가
    selfplay_ema_alpha: float = 0.1       # 승률 EMA 갱신율
    pool_sample_temp: float = 0.3         # softmax 온도 τ
    pool_uniform_floor: float = 0.5       # 균등 분배 비율 f
    opp_sample: bool = True               # opponent 행동 확률적 샘플 여부
    milestone_period: int = 500           # permanent snapshot(+capacity) + exploiter 주기
    # ── exploiter ──────────────────────────────────────────────────────────────
    exploiter_iters: int = 500
    exploiter_win_target: float = 0.7
    # exploiter 를 scratch 대신 main net 파라미터로 초기화(비슷한 타이밍에 죽어 위상이 겹치는
    # 문제 완화). first(기본 500) iter 의 exploiter 는 그 시점 main net 으로, 나머지 모든
    # exploiter 는 rest(기본 1000) iter 시점 main net 으로 초기화한다.
    exploiter_init_iteration_first: int = 500
    exploiter_init_iteration_rest: int = 1000
    exploiter_lr: float = 1e-4
    exploiter_ent_coef: float = 5e-5
    exploiter_clip_coef: float = 0.4
    # exploiter 는 shaping 대신 상대 고도 log 사냥 보상을 쓴다(reward_mode=1). 매 step
    #   C*(ln(상대 이전고도) - ln(상대 현재고도)) = C*ln(상대고도 감소비). C = 아래 계수.
    exploiter_alt_hunt_coef: float = 5.0


@dataclass
class GPUIterationStats:
    iteration: int
    global_step: int
    mean_return: float
    mean_length: float
    completed_episodes: float
    win_rate: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    explained_variance: float
    steps_per_sec: float
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


# ── trainer ──────────────────────────────────────────────────────────────────
class PPOGPUTrainer:
    def __init__(self, env, config: PPOGPUConfig):
        self.env = env
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        dev = config.device

        self.nenv = env.nenv
        self.nac = env.nac
        self.obs_dim = int(env.OBS_SIZE)
        self.act_dim = 4
        self.min_alt = float(getattr(env, "min_altitude_m", 300.0))
        # critic 전용 상대-행동 extra: 관측 마지막 20dim(=자기 과거5×4 action) 과 동일 형태의
        # '상대' 블록. 상대의 관측 [ah_lo:obs_dim] 이 곧 상대의 과거 5-step action 이다.
        self._ah_lo = self.obs_dim - ACT_HIST_DIM
        self.cext_dim = ACT_HIST_DIM if config.critic_opp_actions else 0
        self.aux_dim = AUX_DIM if config.aux_pred else 0
        self._aux_dt = float(getattr(env, "obr", None).dt) if hasattr(env, "obr") else 0.1
        self._model_kwargs = dict(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                  num_bins=config.num_bins, hidden=tuple(config.hidden),
                                  activation=config.activation, critic_extra_dim=self.cext_dim,
                                  aux_dim=self.aux_dim)
        self.grid = make_action_grid(config.num_bins, device=dev)

        self.model = ActorCritic(**self._model_kwargs).to(dev)
        self._build_optim()
        self.norm = RunningNorm(self.obs_dim, dev) if config.normalize_obs else None
        self.global_step = 0

        # opponent pool + 샘플 가중치 + env 별 opponent 인덱스.
        self.pool = OpponentPool(self._model_kwargs, dev, evict_cap=config.pool_evict_cap,
                                 sample=config.opp_sample)
        self.pool.add(self.model, self.norm, permanent=False, ema=0.5)   # 초기 상대(=시작 정책)
        self.opp_weights = self.pool.weights(config.pool_sample_temp, config.pool_uniform_floor)
        self.opp_assign = torch.zeros(self.nenv, dtype=torch.long, device=dev)

        # 롤아웃 버퍼(main 기체만; (T,nenv,...)). 행동은 index(long).
        self._alloc_rollout_buffers(config.rollout_steps)
        self.ep_ret = torch.zeros(self.nenv, device=dev)
        self.ep_len = torch.zeros(self.nenv, device=dev)

        # exploiter 초기화 시드 스냅샷 2종(각 해당 iteration 에서 1회 캡처 후 재사용, resume 대비
        # checkpoint 에도 저장/복원). first = exploiter_init_iteration_first(기본 500) 시점 main
        # net → first iter 의 exploiter 전용. rest = exploiter_init_iteration_rest(기본 1000)
        # 시점 main net → 그 외 모든 exploiter.
        self._exp_init_first = None
        self._exp_init_rest = None

        self._reset_env_state()

    def _alloc_rollout_buffers(self, T):
        """(T,nenv,...) 롤아웃 버퍼 (재)할당. 스케줄로 rollout 이 바뀌면 다시 호출한다."""
        T = int(T)
        dev = self.cfg.device
        self.b_obs = torch.zeros(T, self.nenv, self.obs_dim, device=dev)
        self.b_act = torch.zeros(T, self.nenv, self.act_dim, dtype=torch.long, device=dev)
        self.b_logp = torch.zeros(T, self.nenv, device=dev)
        self.b_rew = torch.zeros(T, self.nenv, device=dev)
        self.b_done = torch.zeros(T, self.nenv, device=dev)
        self.b_val = torch.zeros(T, self.nenv, device=dev)
        # critic 전용 상대-행동 extra 버퍼(cext_dim==0 이면 미사용).
        self.b_cext = (torch.zeros(T, self.nenv, self.cext_dim, device=dev)
                       if self.cext_dim > 0 else None)
        # aux 미래위치 예측용 궤적 캡처(main/opp NED 위치·속도, main body 회전) + 라벨/마스크.
        if self.aux_dim > 0:
            self.b_mpos = torch.zeros(T, self.nenv, 3, device=dev)   # main NED 위치
            self.b_mvel = torch.zeros(T, self.nenv, 3, device=dev)   # main NED 속도
            self.b_opos = torch.zeros(T, self.nenv, 3, device=dev)   # opp NED 위치
            self.b_ovel = torch.zeros(T, self.nenv, 3, device=dev)   # opp NED 속도
            self.b_mR = torch.zeros(T, self.nenv, 3, 3, device=dev)  # main ned->body
            self.b_auxlab = torch.zeros(T, self.nenv, AUX_DIM, device=dev)   # [opp3, self3]
            self.b_auxmask = torch.zeros(T, self.nenv, 2, device=dev)        # [opp valid, self valid]
        else:
            self.b_mpos = self.b_auxlab = self.b_auxmask = None

    def _apply_schedule(self, it):
        """sched_period iter 마다 단계 k=(it-1)//period 로: lr·ent_coef ×= decay^k,
        rollout += increment·k. gamma 등은 불변. base 값은 첫 호출(=학습/재개 시작) 시점의
        cfg 값으로 고정하고 매 iter iteration 번호만으로 결정 → 재개(resume) 안전.
        (checkpoint 는 cfg 를 복원하지 않으므로 cfg 는 항상 CLI base 값 그대로다.)"""
        period = int(getattr(self.cfg, "sched_period", 0) or 0)
        if period <= 0:
            return
        if not hasattr(self, "_sched_base"):
            clr = self.cfg.critic_lr if self.cfg.critic_lr is not None else self.cfg.lr
            self._sched_base = {"rollout": int(self.cfg.rollout_steps),
                                "ent": float(self.cfg.ent_coef),
                                "actor_lr": float(self.cfg.lr),
                                "critic_lr": float(clr)}
            self._sched_phase = -1
        b = self._sched_base
        k = (int(it) - 1) // period
        lr_factor = float(self.cfg.sched_lr_decay) ** k
        self.cfg.ent_coef = b["ent"] * (float(self.cfg.sched_ent_decay) ** k)
        for g in self.actor_opt.param_groups:
            g["lr"] = b["actor_lr"] * lr_factor
        for g in self.critic_opt.param_groups:
            g["lr"] = b["critic_lr"] * lr_factor
        new_T = b["rollout"] + int(self.cfg.sched_rollout_increment) * k
        if new_T != int(self.cfg.rollout_steps):
            self.cfg.rollout_steps = new_T
            self._alloc_rollout_buffers(new_T)
        if k != self._sched_phase:
            self._sched_phase = k
            print(f"[gpu-ppo] 스케줄 단계 k={k} (iter {it}): rollout={self.cfg.rollout_steps}, "
                  f"lr={b['actor_lr']*lr_factor:.3e}, ent_coef={self.cfg.ent_coef:.3e}", flush=True)

    def _build_optim(self):
        c = self.cfg
        clr = c.critic_lr if c.critic_lr is not None else c.lr
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=c.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)

    def _sample_opp(self, n):
        """가중치(opp_weights) 기반 opponent 인덱스 n 개 샘플."""
        return torch.multinomial(self.opp_weights, n, replacement=True)

    def _opp_cext(self, opp_obs):
        """상대 관측(nenv,OBS)에서 상대 과거 5-step action(20dim)을 뽑아 critic extra 로.
        관측의 action-history 블록과 동일 분포이므로 같은 정규화 통계([ah_lo:obs_dim])를 쓴다.
        cext_dim==0 이면 None."""
        if self.cext_dim == 0:
            return None
        raw = opp_obs[:, self._ah_lo:self.obs_dim]
        if self.norm is not None:
            return self.norm.normalize_slice(raw, self._ah_lo, self.obs_dim)
        return raw

    def _capture_aux(self, t):
        """step t 시작 시(=s_t, env.step 전) main/opp 의 NED 위치·속도와 main 의 ned->body
        회전을 저장한다. 이후 미래 위치(t+H) 와 함께 aux 라벨을 만든다."""
        if self.aux_dim == 0:
            return
        s9 = self.env.state9()                       # (nenv,2,9) fp64
        m = s9[:, 0, :]; o = s9[:, 1, :]
        Rm = _ned_to_body(m[:, 3:6]); Ro = _ned_to_body(o[:, 3:6])
        self.b_mpos[t] = m[:, 0:3].float()
        self.b_opos[t] = o[:, 0:3].float()
        self.b_mvel[t] = _mv3(Rm.transpose(1, 2), m[:, 6:9]).float()   # body vel → NED
        self.b_ovel[t] = _mv3(Ro.transpose(1, 2), o[:, 6:9]).float()
        self.b_mR[t] = Rm.float()

    def _build_aux_labels(self, T):
        """롤아웃 후 lookahead 로 aux 라벨/마스크 계산. 라벨 = (실제 t+H 위치 - 등속직선 baseline)
        을 main body 좌표계(t)로 표현하고 AUX_POS_SCALE 로 정규화. 에피소드 경계를 넘거나
        (b_done) 롤아웃 끝을 넘는 t 는 마스크 0(손실 제외). 상대 H=AUX_OPP_H, 나 H=AUX_SELF_H."""
        if self.aux_dim == 0:
            return
        dt = self._aux_dt
        self.b_auxlab.zero_(); self.b_auxmask.zero_()
        done = self.b_done                            # (T,nenv): s_t 가 새 에피소드 시작이면 1
        for t in range(T):
            if t + AUX_OPP_H < T:
                valid = (done[t + 1:t + AUX_OPP_H + 1].sum(0) == 0).float()
                base = self.b_opos[t] + self.b_ovel[t] * (AUX_OPP_H * dt)
                res = self.b_opos[t + AUX_OPP_H] - base
                self.b_auxlab[t, :, 0:3] = _mv3(self.b_mR[t], res) / AUX_POS_SCALE_M
                self.b_auxmask[t, :, 0] = valid
            if t + AUX_SELF_H < T:
                valid = (done[t + 1:t + AUX_SELF_H + 1].sum(0) == 0).float()
                base = self.b_mpos[t] + self.b_mvel[t] * (AUX_SELF_H * dt)
                res = self.b_mpos[t + AUX_SELF_H] - base
                self.b_auxlab[t, :, 3:6] = _mv3(self.b_mR[t], res) / AUX_POS_SCALE_M
                self.b_auxmask[t, :, 1] = valid

    @staticmethod
    def _aux_loss(pred, lab, mask):
        """masked MSE. pred/lab (B,6)=[opp3,self3], mask (B,2)=[opp valid, self valid]."""
        err = (pred - lab) ** 2
        opp = (err[:, 0:3].sum(1) * mask[:, 0]).sum()
        slf = (err[:, 3:6].sum(1) * mask[:, 1]).sum()
        denom = (mask[:, 0].sum() + mask[:, 1].sum()) * 3.0 + 1e-8
        return (opp + slf) / denom

    def _reset_env_state(self):
        obs = self.env.reset(stagger=True)
        self._next_obs = obs[:, 0, :].contiguous()
        self._next_opp_obs = obs[:, 1, :].contiguous()
        self._next_done = torch.zeros(self.nenv, device=self.cfg.device)
        self.opp_assign = self._sample_opp(self.nenv)
        self.ep_ret.zero_(); self.ep_len.zero_()

    def _refresh_weights(self):
        self.opp_weights = self.pool.weights(self.cfg.pool_sample_temp, self.cfg.pool_uniform_floor)
        self.opp_assign = self._sample_opp(self.nenv)      # pool 구성 변경 시 전 재샘플(인덱스 정합)

    def _remap_assign(self, old_ids):
        """pool 구조변경(gate add/evict, milestone add) 뒤 opp_weights 만 갱신하고, opp_assign 은
        **동일 상대(안정 id)를 계속 가리키도록 재매핑**한다. 전역 재샘플(_refresh_weights)은 진행
        중 에피소드의 상대를 중간에 갈아치워(에피소드가 rollout 여러 개에 걸침) 관측·승패 귀속을
        오염시키므로, pool 이 바뀌어도 env 는 자기 상대를 에피소드 끝(done)까지 유지해야 한다.
        old_ids = 구조변경 **직전** 위치→id. 상대가 evict 된 env 만(불가피) 새로 샘플한다."""
        self.opp_weights = self.pool.weights(self.cfg.pool_sample_temp, self.cfg.pool_uniform_floor)
        new_ids = self.pool.entry_ids()
        if not old_ids or not new_ids:
            self.opp_assign = self._sample_opp(self.nenv)
            return
        id_to_new = {i: p for p, i in enumerate(new_ids)}
        old_to_new = torch.tensor([id_to_new.get(i, -1) for i in old_ids],
                                  dtype=torch.long, device=self.cfg.device)
        cur = self.opp_assign.clamp(0, old_to_new.numel() - 1)
        remapped = old_to_new[cur]                         # 살아남은 상대는 새 위치로, evict 는 -1
        evicted = remapped < 0
        if bool(evicted.any()):                            # 상대가 사라진 env 만 새로 샘플(드묾)
            remapped = torch.where(evicted, self._sample_opp(self.nenv), remapped)
        self.opp_assign = remapped

    # ── 롤아웃 수집 (main=학습, opponent=frozen) ──────────────────────────────
    @torch.no_grad()
    def collect_rollout(self, opp_kind="pool", frozen_opp=None):
        cfg = self.cfg
        T = cfg.rollout_steps
        gamma = cfg.gamma
        dev = cfg.device
        P = self.pool.size()
        ret_sum = torch.zeros((), device=dev)
        len_sum = torch.zeros((), device=dev)
        ep_count = torch.zeros((), device=dev)
        win_sum = torch.zeros((), device=dev)
        loss_sum = torch.zeros((), device=dev)
        alt_loss_sum = torch.zeros((), device=dev)   # main 이 고도제한 위반으로 패한 게임 수
        win_by_opp = torch.zeros(P, device=dev)
        loss_by_opp = torch.zeros(P, device=dev)
        ep_by_opp = torch.zeros(P, device=dev)       # opponent 별 완료 에피소드 수(무승부 포함, EMA 분모)

        for t in range(T):
            if opp_kind == "pool":
                sampled = self._sample_opp(self.nenv)
                self.opp_assign = torch.where(self._next_done.bool(), sampled, self.opp_assign)

            self._capture_aux(t)                # s_t 궤적(위치/속도/회전) 캡처(aux 라벨용)
            main_obs = self._next_obs
            if self.norm is not None:
                self.norm.update(main_obs)
                obs_n = self.norm.normalize(main_obs)
            else:
                obs_n = main_obs
            self.b_obs[t] = obs_n
            self.b_done[t] = self._next_done

            cext_n = self._opp_cext(self._next_opp_obs)     # critic 전용 상대-행동 extra
            if self.cext_dim > 0:
                self.b_cext[t] = cext_n
            act_idx, logp, _, value = self.model.get_action_and_value(obs_n, cext_n)
            self.b_act[t] = act_idx
            self.b_logp[t] = logp
            self.b_val[t] = value

            if opp_kind == "pool":
                opp_idx = self.pool.act(self._next_opp_obs, self.opp_assign)
            else:
                oo = self._next_opp_obs
                on = frozen_opp[1].normalize(oo) if frozen_opp[1] is not None else oo
                opp_idx = frozen_opp[0].act(on, sample=cfg.opp_sample)

            act = torch.empty(self.nac, self.act_dim, device=dev)
            act[0::2] = action_to_env(act_idx, cfg.num_bins, self.grid)
            act[1::2] = action_to_env(opp_idx, cfg.num_bins, self.grid)
            obs, reward, done, info = self.env.step(act)

            raw_reward = reward[:, 0].float()
            trunc = info["truncated"]
            done_env = done.float()

            term_obs = info["terminal_obs"][:, 0, :]
            tn = self.norm.normalize(term_obs) if self.norm is not None else term_obs
            term_cext = self._opp_cext(info["terminal_obs"][:, 1, :])
            v_boot = self.model.get_value(tn, term_cext)
            self.b_rew[t] = raw_reward + gamma * v_boot * trunc.float()

            self._next_obs = obs[:, 0, :]
            self._next_opp_obs = obs[:, 1, :]
            self._next_done = done_env
            self.global_step += self.nenv

            self.ep_ret += raw_reward
            self.ep_len += 1.0
            ret_sum += (self.ep_ret * done_env).sum()
            len_sum += (self.ep_len * done_env).sum()
            ep_count += done_env.sum()
            keep = 1.0 - done_env
            self.ep_ret = self.ep_ret * keep
            self.ep_len = self.ep_len * keep

            th = info["terminal_hp"]; ta = info["terminal_alt_m"]
            own_alt_dead = ta[:, 0] < self.min_alt
            own_dead = (th[:, 0] <= 0.0) | own_alt_dead
            opp_dead = (th[:, 1] <= 0.0) | (ta[:, 1] < self.min_alt)
            both_alive = (~own_dead) & (~opp_dead)
            win = (opp_dead & ~own_dead) | (both_alive & (th[:, 0] > th[:, 1] + 1e-9))
            loss = (own_dead & ~opp_dead) | (both_alive & (th[:, 0] < th[:, 1] - 1e-9))
            done_b = done.bool()
            dwin = (win & done_b).float()
            dloss = (loss & done_b).float()
            win_sum += dwin.sum(); loss_sum += dloss.sum()
            # 고도제한 패배: 패로 판정된 게임 중 main 이 min_alt 아래로 내려간 경우.
            alt_loss_sum += (loss & own_alt_dead & done_b).float().sum()
            if opp_kind == "pool":       # per-opponent 집계(EMA 갱신용, sync-free scatter).
                win_by_opp.scatter_add_(0, self.opp_assign, dwin)
                loss_by_opp.scatter_add_(0, self.opp_assign, dloss)
                ep_by_opp.scatter_add_(0, self.opp_assign, done_b.float())

        self._build_aux_labels(T)               # lookahead 로 aux 미래위치 라벨/마스크 생성

        last_n = self.norm.normalize(self._next_obs) if self.norm is not None else self._next_obs
        last_cext = self._opp_cext(self._next_opp_obs)
        last_value = self.model.get_value(last_n, last_cext)

        adv = torch.zeros_like(self.b_rew)
        lastgae = torch.zeros(self.nenv, device=dev)
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - self._next_done
                next_values = last_value
            else:
                next_nonterminal = 1.0 - self.b_done[t + 1]
                next_values = self.b_val[t + 1]
            delta = self.b_rew[t] + gamma * next_values * next_nonterminal - self.b_val[t]
            lastgae = delta + gamma * cfg.gae_lambda * next_nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + self.b_val

        stats = {"ret_sum": ret_sum, "len_sum": len_sum, "ep_count": ep_count,
                 "win_sum": win_sum, "loss_sum": loss_sum, "alt_loss_sum": alt_loss_sum,
                 "win_by_opp": win_by_opp, "loss_by_opp": loss_by_opp, "ep_by_opp": ep_by_opp}
        return adv, ret, stats

    # ── 정책 업데이트 ────────────────────────────────────────────────────────
    def update(self, adv, ret):
        cfg = self.cfg
        N = cfg.rollout_steps * self.nenv
        b_obs = self.b_obs.reshape(N, self.obs_dim)
        b_act = self.b_act.reshape(N, self.act_dim)
        b_logp = self.b_logp.reshape(N)
        b_adv = adv.reshape(N)
        b_ret = ret.reshape(N)
        b_val = self.b_val.reshape(N)
        b_cext = self.b_cext.reshape(N, self.cext_dim) if self.cext_dim > 0 else None
        use_aux = self.aux_dim > 0
        b_auxlab = self.b_auxlab.reshape(N, AUX_DIM) if use_aux else None
        b_auxmask = self.b_auxmask.reshape(N, 2) if use_aux else None

        mb_size = max(1, N // cfg.num_minibatches)
        idx = torch.arange(N, device=cfg.device)
        clip = cfg.clip_coef
        last_pl = last_vl = last_ent = last_kl = last_cf = last_aux = 0.0
        early = False
        epoch = 0
        for epoch in range(cfg.update_epochs):
            perm = idx[torch.randperm(N, device=cfg.device)]
            kls = []
            for s in range(0, N, mb_size):
                mb = perm[s:s + mb_size]
                new_logp, entropy, a_aux = self.model.evaluate_actions(
                    b_obs[mb], b_act[mb], with_aux=use_aux)
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg1, pg2).mean()
                ent = entropy.mean()
                actor_loss = policy_loss - cfg.ent_coef * ent
                # actor trunk aux 미래위치 예측(표현학습). 작은 계수로 정책 gradient 에 가산.
                a_aux_loss = (self._aux_loss(a_aux, b_auxlab[mb], b_auxmask[mb])
                              if use_aux else None)
                if use_aux:
                    actor_loss = actor_loss + cfg.aux_coef * a_aux_loss

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor_parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                mb_cext = b_cext[mb] if b_cext is not None else None
                if use_aux:
                    new_value, c_aux = self.model.get_value(b_obs[mb], mb_cext, with_aux=True)
                    c_aux_loss = self._aux_loss(c_aux, b_auxlab[mb], b_auxmask[mb])
                else:
                    new_value = self.model.get_value(b_obs[mb], mb_cext)
                value_loss = 0.5 * ((new_value - b_ret[mb]) ** 2).mean()
                critic_loss = cfg.vf_coef * value_loss
                if use_aux:                                   # critic trunk aux(추론 미사용)
                    critic_loss = critic_loss + cfg.aux_coef * c_aux_loss
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.model.critic_parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean())
                    clipfrac = ((ratio - 1.0).abs() > clip).float().mean()
                last_pl = policy_loss.detach(); last_vl = value_loss.detach()
                last_ent = ent.detach(); last_cf = clipfrac
                if use_aux:
                    last_aux = a_aux_loss.detach()
            last_kl = torch.stack(kls).mean()
            if cfg.target_kl is not None and float(last_kl) > cfg.target_kl:
                early = True
                break

        var_y = b_ret.var()
        ev = torch.where(var_y == 0, torch.zeros((), device=cfg.device),
                         1.0 - (b_ret - b_val).var() / (var_y + 1e-8))
        return {"pl": last_pl, "vl": last_vl, "ent": last_ent, "kl": last_kl,
                "cf": last_cf, "ev": ev, "epochs": epoch + 1, "early": early, "aux": last_aux}

    # ── main 상태 저장/복원 (exploiter 학습이 self.model 등을 임시 사용) ────────
    def _snapshot_learner(self):
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "actor_opt": copy.deepcopy(self.actor_opt.state_dict()),
                "critic_opt": copy.deepcopy(self.critic_opt.state_dict()),
                "norm": (self.norm.state_dict() if self.norm is not None else None),
                "global_step": int(self.global_step)}

    def _capture_exploiter_init(self):
        """현재 main net(+norm) 을 CPU 로 복사해 exploiter 초기화 시드로 보관.
        norm 은 live 버퍼 참조가 아니라 clone 을 저장(이후 학습에 오염되지 않도록)."""
        norm_sd = None
        if self.norm is not None:
            norm_sd = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                       for k, v in self.norm.state_dict().items()}
        return {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "norm": norm_sd,
                "iteration": int(getattr(self, "iteration", 0))}

    def _restore_learner(self, snap):
        self.model.load_state_dict({k: v.to(self.cfg.device) for k, v in snap["model"].items()})
        self._build_optim()
        self.actor_opt.load_state_dict(snap["actor_opt"])
        self.critic_opt.load_state_dict(snap["critic_opt"])
        if self.norm is not None and snap["norm"] is not None:
            self.norm.load_state_dict(snap["norm"])
        self.global_step = int(snap["global_step"])

    # ── exploiter 학습: 현재 main 을 유일 상대로 새 정책을 scratch 부터 학습 ────
    def train_exploiter(self, log=None, metric_cb=None, alt_hunt=True):
        """metric_cb(i, metrics_dict): 매 exploiter iteration 의 전체 지표를 넘겨(wandb 섹터
        로깅용). log(i, wr, eps): 콘솔 출력용 간단 콜백(기존 호환).
        alt_hunt=True 면 상대고도 log 사냥 보상(reward_mode=1)으로 '고도추락 전문' exploiter 를,
        False 면 main 과 동일한 보상(reward_mode=0)으로 일반 exploiter 를 학습한다."""
        cfg = self.cfg
        if cfg.exploiter_iters <= 0:
            return None
        saved = self._snapshot_learner()
        # exploiter 보상 모드 토글(끝나면 원복). alt_hunt=True: shaping 제거 + 상대(=frozen main)
        # 고도 log 사냥(reward_mode=1) → 고도추락 전문. alt_hunt=False: main 과 동일(reward_mode=0).
        _saved_rmode, _saved_coef = self.env.reward_mode, self.env.alt_hunt_coef
        self.env.reward_mode = 1 if alt_hunt else 0
        self.env.alt_hunt_coef = float(cfg.exploiter_alt_hunt_coef)
        if alt_hunt:
            print(f"[gpu-ppo]   exploiter 보상: 상대고도 log 사냥(reward_mode=1, "
                  f"C={cfg.exploiter_alt_hunt_coef}) + damage/고도이탈 종료 보상", flush=True)
        else:
            print("[gpu-ppo]   exploiter 보상: main 과 동일(reward_mode=0, "
                  "거리/조준 shaping + damage/고도이탈 종료 보상)", flush=True)
        frozen_net = ActorCritic(**self._model_kwargs).to(cfg.device)
        frozen_net.load_state_dict(copy.deepcopy(self.model.state_dict()))
        frozen_net.eval()
        for p in frozen_net.parameters():
            p.requires_grad_(False)
        frozen_norm = self.norm.clone() if self.norm is not None else None

        # exploiter 초기화: first(기본 500) iter 의 exploiter 는 500-iter main 스냅샷으로,
        # 나머지 모든 exploiter 는 rest(기본 1000) iter main 스냅샷으로 초기화한다.
        cur_it = int(getattr(self, "iteration", 0))
        if cur_it == int(getattr(cfg, "exploiter_init_iteration_first", 500)):
            init = self._exp_init_first
        else:
            init = self._exp_init_rest
        if init is None:   # 캡처 전(구 checkpoint resume 등) 폴백: 현재 main 으로 초기화.
            print("[gpu-ppo] (경고) exploiter_init 스냅샷 없음 → 현재 main net 으로 초기화(폴백)",
                  flush=True)
            init = self._capture_exploiter_init()
        self.model.load_state_dict({k: v.to(cfg.device) for k, v in init["model"].items()})
        if self.norm is not None:
            if init["norm"] is not None:
                self.norm.load_state_dict(init["norm"])   # copy_ 가 cpu→gpu 자동 처리
            else:
                self.norm = RunningNorm(self.obs_dim, cfg.device)
        print(f"[gpu-ppo]   exploiter 초기화: iter{init.get('iteration', '?')} main net 파라미터",
              flush=True)
        base_ent, base_clip = cfg.ent_coef, cfg.clip_coef
        cfg.ent_coef, cfg.clip_coef = cfg.exploiter_ent_coef, cfg.exploiter_clip_coef
        self.actor_opt = torch.optim.Adam(self.model.actor_parameters(), lr=cfg.exploiter_lr, eps=1e-5)
        clr = cfg.critic_lr if cfg.critic_lr is not None else cfg.exploiter_lr
        self.critic_opt = torch.optim.Adam(self.model.critic_parameters(), lr=clr, eps=1e-5)
        self._reset_env_state()

        # exploiter(=학습 중인 self.model)의 frozen-main 상대 승률 EMA. pool 엔트리와 동일하게
        # 중립값 0.5 에서 시작하고 selfplay_ema_alpha 로 갱신한다. 이 EMA 가 target 을 넘으면
        # "안정적으로 main 을 압도" 로 보고 exploiter 학습을 멈춰 pool 에 추가한다.
        alpha = float(cfg.selfplay_ema_alpha)
        wr_ema = 0.5
        wr = 0.0
        for i in range(1, int(cfg.exploiter_iters) + 1):
            t0 = time.time()
            adv, ret, rs = self.collect_rollout(opp_kind="frozen",
                                                frozen_opp=(frozen_net, frozen_norm))
            u = self.update(adv, ret)
            ep_c = float(rs["ep_count"])
            dec = float(rs["win_sum"] + rs["loss_sum"])
            wr = float(rs["win_sum"]) / dec if dec > 0 else 0.0
            # 무승부 0.5 반영 win_rate(리포팅용; main 과 동일 규약).
            n_draw = max(0.0, ep_c - float(rs["win_sum"]) - float(rs["loss_sum"]))
            wr_draw = (float(rs["win_sum"]) + 0.5 * n_draw) / ep_c if ep_c > 0 else float("nan")
            mean_ret = float(rs["ret_sum"] / rs["ep_count"]) if ep_c > 0 else float("nan")
            alt_lr = float(rs["alt_loss_sum"]) / ep_c if ep_c > 0 else float("nan")
            # 승률 EMA 갱신(완료 에피소드 있을 때만; 무승부 0.5 반영 규약 동일). ep_c==0 이면
            # 직전 EMA 유지(정보 없음).
            if ep_c > 0:
                wr_ema = (1.0 - alpha) * wr_ema + alpha * wr_draw
            dt = time.time() - t0
            # 매 iteration 콘솔 출력(main 루프와 비슷한 정보량).
            print(f"[gpu-ppo]   exp it {i:4d} | wr_ema {wr_ema:.3f} wr {wr:.3f}(draw {wr_draw:.3f}) "
                  f"| eps {int(ep_c):4d} ret {mean_ret:7.2f} altL {alt_lr:.3f} "
                  f"| pl {float(u['pl']):+.3f} vl {float(u['vl']):.3f} ent {float(u['ent']):.3f} "
                  f"kl {float(u['kl']):.4f} | {dt:.1f}s", flush=True)
            if log is not None:
                log(i, wr, ep_c)
            if metric_cb is not None:
                metric_cb(i, {
                    "win_rate": wr_draw, "win_rate_decided": wr, "win_rate_ema": wr_ema,
                    "completed_episodes": ep_c,
                    "mean_return": mean_ret, "alt_loss_rate": alt_lr,
                    "policy_loss": float(u["pl"]), "value_loss": float(u["vl"]),
                    "entropy": float(u["ent"]), "approx_kl": float(u["kl"]),
                    "clipfrac": float(u["cf"]), "explained_variance": float(u["ev"]),
                    "aux_pred_mse": float(u.get("aux", 0.0)),
                    "ent_coef": float(cfg.ent_coef), "clip_coef": float(cfg.clip_coef),
                    "lr": float(cfg.exploiter_lr), "elapsed_sec": dt,
                })
            # 게이팅: 판정 승률이 아니라 승률 EMA 가 target 을 넘으면 조기 종료.
            if wr_ema >= cfg.exploiter_win_target:
                print(f"[gpu-ppo]   exploiter 조기 종료: wr_ema {wr_ema:.3f} ≥ "
                      f"target {cfg.exploiter_win_target} (it {i})", flush=True)
                break

        self.pool.add(self.model, self.norm, permanent=True, ema=0.5)

        # 보상 모드 원복(main 학습은 다시 reward_mode=0).
        self.env.reward_mode, self.env.alt_hunt_coef = _saved_rmode, _saved_coef
        cfg.ent_coef, cfg.clip_coef = base_ent, base_clip
        if saved["norm"] is not None and self.norm is None:
            self.norm = RunningNorm(self.obs_dim, cfg.device)
        self._restore_learner(saved)
        self._refresh_weights()
        self._reset_env_state()
        return wr_ema

    # ── 메인 루프 ────────────────────────────────────────────────────────────
    def train(self, on_iteration: Optional[Callable[[GPUIterationStats], None]] = None,
              start_iteration=1, on_exploiter_iter: Optional[Callable[[int, int, dict], None]] = None):
        """on_exploiter_iter(milestone_it, exp_iter, metrics): milestone 에서 도는 exploiter
        학습의 매 iteration 지표를 넘긴다(wandb 섹터별 로깅용)."""
        history = []
        it = start_iteration
        while self.cfg.total_iterations <= 0 or it <= self.cfg.total_iterations:
            self.iteration = it
            self._apply_schedule(it)   # 2000-iter 스케줄: lr·ent_coef 감쇠, rollout 증가
            t0 = time.time()
            adv, ret, rstats = self.collect_rollout(opp_kind="pool")
            u = self.update(adv, ret)

            # 이 update 직후 main net 은 정확히 'it 번 학습된' 버전. exploiter 초기화 시드는
            # first(기본 500)/rest(기본 1000) iter 시점에 각각 1회만 캡처해 이후 재사용한다.
            if self._exp_init_first is None and \
                    it == int(getattr(self.cfg, "exploiter_init_iteration_first", 0)):
                self._exp_init_first = self._capture_exploiter_init()
                print(f"[gpu-ppo] exploiter(first) 초기화 스냅샷 캡처: iter{it} main net", flush=True)
            if self._exp_init_rest is None and \
                    it == int(getattr(self.cfg, "exploiter_init_iteration_rest", 0)):
                self._exp_init_rest = self._capture_exploiter_init()
                print(f"[gpu-ppo] exploiter(rest) 초기화 스냅샷 캡처: iter{it} main net", flush=True)

            ep_count = float(rstats["ep_count"])
            mean_ret = float(rstats["ret_sum"] / rstats["ep_count"]) if ep_count > 0 else float("nan")
            mean_len = float(rstats["len_sum"] / rstats["ep_count"]) if ep_count > 0 else float("nan")
            # 리포팅 win_rate: 무승부(draw=완료 - 승 - 패)를 0.5 로 계산 → 완료 에피소드가
            # 있으면 nan 이 나오지 않는다. 분모는 전체 완료 에피소드(승+패+무).
            n_win = float(rstats["win_sum"]); n_loss = float(rstats["loss_sum"])
            n_draw = max(0.0, ep_count - n_win - n_loss)
            win_rate = (n_win + 0.5 * n_draw) / ep_count if ep_count > 0 else float("nan")
            # main 이 고도제한 위반으로 패한 게임 비율(완료 에피소드 대비). exploiter 학습은
            # 별도 루프(train_exploiter)라 on_iteration 을 안 타므로 자동으로 제외된다.
            alt_loss_rate = float(rstats["alt_loss_sum"]) / ep_count if ep_count > 0 else float("nan")
            elapsed = time.time() - t0
            sps = self.cfg.rollout_steps * self.nenv / max(elapsed, 1e-9)
            pool_event = None

            # pool 구조변경 **직전** 위치→id 스냅샷(상대 identity 보존 재매핑용). rollout·update 는
            # pool 을 바꾸지 않으므로 이 시점 id 순서가 이번 rollout 의 opp_assign 위치와 정합.
            old_pool_ids = self.pool.entry_ids()
            # EMA 갱신 → 게이팅(evictable 최소 EMA ≥ threshold 면 현재 main 추가).
            self.pool.update_emas(rstats["win_by_opp"], rstats["loss_by_opp"],
                                  rstats["ep_by_opp"], self.cfg.selfplay_ema_alpha)
            if self.pool.gate_and_add(self.model, self.norm, self.cfg.selfplay_gate_threshold):
                pool_event = "gate"
            ema_min = min((e["ema"] for e in self.pool.entries if not e["permanent"]), default=float("nan"))
            ema_mean = float(np.mean([e["ema"] for e in self.pool.entries])) if self.pool.size() else float("nan")

            # milestone: permanent main snapshot(+capacity) + exploiter.
            exploiter_ran = False
            if self.cfg.milestone_period > 0 and it % self.cfg.milestone_period == 0:
                self.pool.add(self.model, self.norm, permanent=True, ema=0.5)
                pool_event = "milestone"
                if self.cfg.exploiter_iters > 0:
                    # milestone_period 의 홀수 배수(500,1500,2500,…) exploiter 만 고도사냥
                    # (reward_mode=1), 짝수 배수(1000,2000,3000,…) 는 main 과 동일 보상.
                    alt_hunt = (it // self.cfg.milestone_period) % 2 == 1
                    print(f"[gpu-ppo] === exploiter 학습 시작 @it{it} "
                          f"({'고도사냥' if alt_hunt else 'main보상'}, "
                          f"max {self.cfg.exploiter_iters}it, target wr_ema ≥ {self.cfg.exploiter_win_target}) ===",
                          flush=True)
                    _mcb = ((lambda i, m: on_exploiter_iter(it, i, m))
                            if on_exploiter_iter is not None else None)
                    ewr = self.train_exploiter(metric_cb=_mcb, alt_hunt=alt_hunt)  # 매 iter 출력은 내부에서 처리
                    exploiter_ran = True   # train_exploiter 가 끝에서 env 리셋+weights 재샘플 수행
                    print(f"[gpu-ppo] === exploiter 완료 wr_ema {ewr:.3f}, pool {self.pool.size()} "
                          f"(perm {self.pool.num_permanent()}, cap {self.pool.capacity()}) ===", flush=True)

            if pool_event is not None and not exploiter_ran:
                # 진행 중 에피소드의 상대를 중간에 바꾸지 않도록 전역 재샘플이 아니라 재매핑한다.
                # (exploiter 가 돈 경우엔 train_exploiter 가 env 를 새로 리셋해 이미 처리됨.)
                self._remap_assign(old_pool_ids)

            # 슬롯별 EMA(리포팅용): evict 슬롯은 FIFO 위치 기준, perm 은 추가 순.
            evict_slot_emas, perm_slot_emas = self.pool.slot_emas()

            stats = GPUIterationStats(
                iteration=it, global_step=self.global_step,
                mean_return=mean_ret, mean_length=mean_len, completed_episodes=ep_count,
                win_rate=win_rate,
                policy_loss=float(u["pl"]), value_loss=float(u["vl"]),
                entropy=float(u["ent"]), approx_kl=float(u["kl"]), clipfrac=float(u["cf"]),
                explained_variance=float(u["ev"]), steps_per_sec=sps, elapsed_sec=elapsed,
                extra={"epochs": int(u["epochs"]), "early_stop": bool(u["early"]),
                       "pool_size": self.pool.size(), "pool_perm": self.pool.num_permanent(),
                       "pool_cap": self.pool.capacity(), "pool_event": pool_event,
                       "ema_min": float(ema_min), "ema_mean": ema_mean,
                       "evict_slot_emas": evict_slot_emas, "perm_slot_emas": perm_slot_emas,
                       "alt_loss_rate": alt_loss_rate,
                       "lr": float(self.actor_opt.param_groups[0]["lr"]),
                       "ent_coef": float(self.cfg.ent_coef),
                       "aux_loss": float(u.get("aux", 0.0)),
                       "rollout": int(self.cfg.rollout_steps)})
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats)
            it += 1
        return history

    # ── checkpoint ───────────────────────────────────────────────────────────
    def save(self, path):
        ckpt = {"model": self.model.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "norm": self.norm.state_dict() if self.norm is not None else None,
                "pool": self.pool.state_dicts(),
                "pool_evict_cap": self.pool.evict_cap,
                "global_step": self.global_step,
                "iteration": int(getattr(self, "iteration", 0)),
                # exploiter 초기화 시드 2종(resume 보존): first=500-iter, rest=1000-iter.
                "exploiter_init_first": getattr(self, "_exp_init_first", None),
                "exploiter_init_rest": getattr(self, "_exp_init_rest", None),
                "cfg": self.cfg.__dict__}
        torch.save(ckpt, path)

    def load(self, path, map_location=None):
        # 자체 checkpoint(optimizer/cfg/pool 포함)라 weights_only=False (신뢰 소스).
        ckpt = torch.load(path, map_location=map_location or self.cfg.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.norm is not None and ckpt.get("norm") is not None:
            self.norm.load_state_dict(ckpt["norm"])
        if ckpt.get("pool") is not None:
            self.pool.load_state_dicts(ckpt["pool"])
        # exploiter 초기화 시드 2종 복원(없으면 None → 해당 iter 에 재캡처/폴백).
        self._exp_init_first = ckpt.get("exploiter_init_first", None)
        self._exp_init_rest = ckpt.get("exploiter_init_rest", None)
        if self._exp_init_rest is None:   # 구 checkpoint 호환: 단일 exploiter_init 키를 rest 시드로.
            self._exp_init_rest = ckpt.get("exploiter_init", None)
        self.global_step = int(ckpt.get("global_step", 0))
        self._refresh_weights()
        self._reset_env_state()
        return ckpt


__all__ = ["PPOGPUConfig", "PPOGPUTrainer", "ActorCritic", "RunningNorm",
           "OpponentPool", "GPUIterationStats", "action_to_env", "make_action_grid",
           "ACTION_BINS"]
