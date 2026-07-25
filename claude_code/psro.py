# -*- coding: utf-8 -*-
"""PSRO (Policy-Space Response Oracles) 용 empirical payoff matrix + Nash meta-solver.

opponent pool 을 하나의 population 으로 보고, 멤버 간 pairwise 승률을 empirical payoff
행렬로 유지한다. 새 opponent(현재 정책 snapshot)가 pool 에 추가되면 그 opponent 를
기존 모든 멤버와 T판 붙여 승률을 채운다. 그 행렬의 **Nash 균형**(대칭 zero-sum 게임의
maximin 혼합전략)을 opponent 샘플링 분포 σ 로 쓴다.

pool 슬롯 순서와 정확히 정렬한다: [BT_0..BT_{n_bt-1}, snapshot_0, snapshot_1, ...].
BT_i vs BT_j(i≠j)는 한 프로세스 공존 불가라 측정 못 하므로 0.5(=무승부, centered 0)로 둔다.

payoff 는 centered antisymmetric 로 저장한다: A[i][j] = winrate(i vs j) − 0.5,
A[j][i] = −A[i][j], 대각(자기 자신=거울)=0. Nash 는 scipy linprog(HiGHS) 로 푼다.
"""
from __future__ import annotations

import numpy as np


def nash_meta_strategy(A: np.ndarray, prefer_uniform: bool = True) -> np.ndarray:
    """centered antisymmetric payoff 행렬 A(N×N)의 대칭 Nash 혼합전략 σ 를 반환.

    1단계 maximin LP:  max v  s.t.  Σ_i σ_i A[i][j] ≥ v  ∀j,  Σ σ = 1,  σ ≥ 0.
    (대칭 zero-sum 이라 게임값 v=0, 양 플레이어 Nash 동일 → opponent 도 σ 로 샘플.)

    2단계(prefer_uniform): Nash 균형이 유일하지 않을 때(예: BT끼리 승률 0.5로 가정 →
    교환 가능해 균형 집합이 면(面)을 이룸) LP 는 꼭짓점(한 BT)에 몰린 해를 준다. 그래서
    **maximin 값 v* 를 유지하는 전략 중 가장 균등한(min Σσ²) 것**을 골라 대칭 BT 들이
    고르게 퍼지게 한다. 실패 시 1단계 해로 폴백.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    try:
        from scipy.optimize import linprog
    except Exception:
        return np.ones(n) / n

    # ── 1단계: maximin LP. 변수 x = [σ_0..σ_{n-1}, v], minimize -v. ──
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.zeros((n, n + 1))     # 각 열 j:  -Σ_i σ_i A[i][j] + v ≤ 0
    for j in range(n):
        A_ub[j, :n] = -A[:, j]
        A_ub[j, n] = 1.0
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return np.ones(n) / n
    if not getattr(res, "success", False):
        return np.ones(n) / n
    sigma_lp = np.clip(np.asarray(res.x[:n], dtype=np.float64), 0.0, None)
    s = float(sigma_lp.sum())
    sigma_lp = sigma_lp / s if s > 1e-12 else np.ones(n) / n
    if not prefer_uniform:
        return sigma_lp

    # ── 2단계: maximin 값 v* 를 유지하며 가장 균등한(min Σσ²) 균형 선택. ──
    v_star = float(res.x[n])
    try:
        from scipy.optimize import minimize
        eps = 1e-6
        cons = [{"type": "eq", "fun": lambda z: float(np.sum(z) - 1.0)}]
        for j in range(n):
            cons.append({"type": "ineq",
                         "fun": (lambda z, jj=j: float(A[:, jj] @ z) - (v_star - eps))})
        res2 = minimize(lambda z: float(np.dot(z, z)), sigma_lp,
                        jac=lambda z: 2.0 * z, method="SLSQP",
                        bounds=[(0.0, 1.0)] * n, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-10})
        if getattr(res2, "success", False):
            sig = np.clip(np.asarray(res2.x, dtype=np.float64), 0.0, None)
            tot = float(sig.sum())
            if tot > 1e-12:
                return sig / tot
    except Exception:
        pass
    return sigma_lp


class PayoffMatrix:
    """pool 과 정렬된 centered antisymmetric 승률 행렬. add/remove 로 pool 변화를 따라간다."""

    def __init__(self):
        self._A = np.zeros((0, 0), dtype=np.float64)

    @property
    def n(self) -> int:
        return int(self._A.shape[0])

    def winrate_matrix(self) -> np.ndarray:
        """행 i 가 열 j 를 이길 승률(대각·미측정은 0.5)."""
        return self._A + 0.5

    def add_member(self, winrates_vs_existing) -> None:
        """새 멤버를 맨 뒤에 추가. winrates_vs_existing[j] = winrate(new vs 기존 멤버 j).

        길이는 현재 멤버 수 n 과 같아야 한다. new vs new(자기)=0.5(centered 0).
        """
        n = self.n
        wr = np.asarray(winrates_vs_existing, dtype=np.float64).reshape(-1)
        if wr.size != n:
            raise ValueError(f"add_member 길이 불일치: {wr.size} != {n}")
        a_row = wr - 0.5
        new = np.zeros((n + 1, n + 1), dtype=np.float64)
        if n:
            new[:n, :n] = self._A
            new[n, :n] = a_row       # new vs 기존
            new[:n, n] = -a_row      # 기존 vs new (antisymmetric)
        self._A = new

    def remove_member(self, idx: int) -> None:
        keep = [i for i in range(self.n) if i != idx]
        self._A = self._A[np.ix_(keep, keep)] if keep else np.zeros((0, 0))

    def set_pair(self, i: int, j: int, winrate_i_vs_j: float) -> None:
        """멤버 i 가 j 를 이긴 승률로 (i,j)/(j,i) 갱신(antisymmetric)."""
        a = float(winrate_i_vs_j) - 0.5
        self._A[i, j] = a
        self._A[j, i] = -a

    def nash(self) -> np.ndarray:
        return nash_meta_strategy(self._A)

    def state(self) -> list:
        return self._A.tolist()

    def load_state(self, rows) -> None:
        self._A = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 0))


__all__ = ["PayoffMatrix", "nash_meta_strategy"]
