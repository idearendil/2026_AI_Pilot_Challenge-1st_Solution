# -*- coding: utf-8 -*-
"""JSBSim FGTable 보간을 double 정밀도로 정확 복제 (bit-정합 목표).

원본: jsbsim/src/math/FGTable.cpp
- 1D  GetValue(key)          : L432-471
- 2D  GetValue(row,col)      : L475-505
클램프/외삽거동(테이블 밖이면 끝값, 외삽 안함)까지 동일하게 재현.
lastRowIndex/lastColumnIndex 캐시는 결과에 영향없으므로(단순 탐색가속) 생략하고
매번 전탐색 — 반환값은 동일.
"""
import numpy as np


class Table1D:
    """독립변수 1개. rows: shape (n,2) [[x0,y0],...] x 오름차순."""
    def __init__(self, rows):
        a = np.asarray(rows, dtype=np.float64)
        self.x = a[:, 0].copy()
        self.y = a[:, 1].copy()
        self.n = len(self.x)

    def value(self, key):
        x, y, n = self.x, self.y, self.n
        # 테이블 밖: 끝값 반환 (외삽 안함) — FGTable.cpp L439-447
        if key <= x[0]:
            return y[0]
        if key >= x[n - 1]:
            return y[n - 1]
        # 중간 구간 탐색 (r: 1..n-1, 원본 1-indexed r 에서 Data[r],Data[r-1])
        r = 1
        while r < n - 1 and x[r] < key:
            r += 1
        span = x[r] - x[r - 1]
        if span != 0.0:
            factor = (key - x[r - 1]) / span
            if factor > 1.0:
                factor = 1.0
        else:
            factor = 1.0
        return factor * (y[r] - y[r - 1]) + y[r - 1]


class Table2D:
    """독립변수 2개(row,col). rowvals(nr,), colvals(nc,), data(nr,nc)."""
    def __init__(self, rowvals, colvals, data):
        self.rx = np.asarray(rowvals, dtype=np.float64)
        self.cx = np.asarray(colvals, dtype=np.float64)
        self.d = np.asarray(data, dtype=np.float64)
        self.nr = len(self.rx)
        self.nc = len(self.cx)

    def value(self, rowKey, colKey):
        rx, cx, d = self.rx, self.cx, self.d
        nr, nc = self.nr, self.nc
        # 원본 L481-485: r,c 를 구간으로 이동 (2..nRows, 2..nCols; 1-indexed)
        # python 0-indexed: r in [1..nr-1], c in [1..nc-1]
        r = 1
        while r < nr - 1 and rx[r] < rowKey:
            r += 1
        # 아래로도 이동(원본 while r>2 && Data[r-1]>rowKey): rowKey 가 구간 아래면
        while r > 1 and rx[r - 1] > rowKey:
            r -= 1
        c = 1
        while c < nc - 1 and cx[c] < colKey:
            c += 1
        while c > 1 and cx[c - 1] > colKey:
            c -= 1
        rFactor = (rowKey - rx[r - 1]) / (rx[r] - rx[r - 1])
        cFactor = (colKey - cx[c - 1]) / (cx[c] - cx[c - 1])
        if rFactor > 1.0:
            rFactor = 1.0
        elif rFactor < 0.0:
            rFactor = 0.0
        if cFactor > 1.0:
            cFactor = 1.0
        elif cFactor < 0.0:
            cFactor = 0.0
        col1 = rFactor * (d[r][c - 1] - d[r - 1][c - 1]) + d[r - 1][c - 1]
        col2 = rFactor * (d[r][c] - d[r - 1][c]) + d[r - 1][c]
        return col1 + cFactor * (col2 - col1)
