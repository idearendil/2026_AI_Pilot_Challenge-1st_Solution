# -*- coding: utf-8 -*-
"""상대 태깅 replay buffer (self-play 비정상성 대응).

핵심 설계(사용자 승인 사항):
  - **raw obs 저장**: RunningMeanStd 통계가 계속 드리프트하므로 정규화된 obs 를 저장하면
    오래된 transition 이 낡은 통계로 정규화된 값이 된다. raw 로 저장하고 샘플링 시점에
    현재 통계로 정규화한다.
  - **terminated / truncated 분리**: Q bootstrap 은 terminated 일 때만 0 으로 잘라야 한다.
    200s 타임아웃(truncated)에서 자르면 후반 가치를 조직적으로 과소평가한다.
  - **상대 gen 태깅**: 각 transition 에 상대 식별자(pool gen; BT=-1)를 붙인다. pool 에서
    snapshot 후보가 evict 되면 그 gen 의 transition 을 purge 한다. BT 는 만료 없음.

링 버퍼(고정 크기 numpy 배열). 오래되면 덮어쓴다. purge 는 tombstone 마스크로 처리해
샘플링에서 제외하고, 자연스럽게 링이 돌며 재사용된다.
"""
from __future__ import annotations

import numpy as np


class OpponentTaggedReplay:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self._rng = np.random.default_rng(int(seed))

        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)       # raw
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)  # raw
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.int64)     # 카테고리 index
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)   # bootstrap 차단용
        self.opp_gen = np.full(self.capacity, -2, dtype=np.int64)     # 상대 gen (BT=-1)
        self.age = np.zeros(self.capacity, dtype=np.int64)            # 삽입 시점 global step
        self.valid = np.zeros(self.capacity, dtype=bool)             # tombstone 마스크

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self.valid.sum())

    @property
    def filled(self) -> int:
        """덮어쓰기 여부와 무관하게 물리적으로 채워진 슬롯 수(포인터 진행)."""
        return self._size

    def add_batch(self, obs, next_obs, actions, rewards, terminated, opp_gen, global_step):
        """여러 transition 을 한 번에 삽입(worker 청크 단위). 배열 인자는 같은 길이."""
        n = len(rewards)
        if n == 0:
            return
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.float32)
        opp_gen = np.asarray(opp_gen, dtype=np.int64)

        idx = (self._ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.terminated[idx] = terminated
        self.opp_gen[idx] = opp_gen
        self.age[idx] = int(global_step)
        self.valid[idx] = True

        self._ptr = int((self._ptr + n) % self.capacity)
        self._size = int(min(self._size + n, self.capacity))

    def purge_opponent(self, gen: int) -> int:
        """상대 gen 의 transition 을 buffer 에서 무효화(tombstone). 제거 수 반환.

        BT(gen=-1)는 pool 에서 evict 되지 않으므로 호출되지 않는다(호출돼도 안전).
        """
        mask = self.valid & (self.opp_gen == int(gen))
        removed = int(mask.sum())
        if removed:
            self.valid[mask] = False
        return removed

    def sample(self, batch_size: int):
        """유효 슬롯에서 batch_size 개 균등 샘플. (obs, next_obs, act, rew, term, gen) 반환.

        obs/next_obs 는 **raw**(정규화 전). 정규화는 호출부(sac.update)가 현재 통계로 한다.
        """
        valid_idx = np.flatnonzero(self.valid)
        if valid_idx.size == 0:
            raise ValueError("replay buffer 가 비어 있음(유효 슬롯 0)")
        take = self._rng.integers(0, valid_idx.size, size=int(batch_size))
        sel = valid_idx[take]
        return {
            "obs": self.obs[sel],
            "next_obs": self.next_obs[sel],
            "actions": self.actions[sel],
            "rewards": self.rewards[sel],
            "terminated": self.terminated[sel],
            "opp_gen": self.opp_gen[sel],
        }

    def age_stats(self, global_step: int) -> dict:
        """유효 transition 의 age(=경과 env-step) 분포 + 상대 gen 구성."""
        valid_idx = np.flatnonzero(self.valid)
        if valid_idx.size == 0:
            return {"count": 0, "age_p50": float("nan"), "age_p90": float("nan"),
                    "bt_frac": float("nan"), "n_opponents": 0}
        ages = int(global_step) - self.age[valid_idx]
        gens = self.opp_gen[valid_idx]
        bt_frac = float((gens == -1).mean())
        return {
            "count": int(valid_idx.size),
            "age_p50": float(np.percentile(ages, 50)),
            "age_p90": float(np.percentile(ages, 90)),
            "bt_frac": bt_frac,
            "n_opponents": int(np.unique(gens).size),
        }


__all__ = ["OpponentTaggedReplay"]
