# -*- coding: utf-8 -*-
"""3x3 행렬 / 3벡터 double 헬퍼 (JSBSim FGMatrix33/FGColumnVector3 연산순서 준수)."""
import math


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(A, v):
    return (A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
            A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
            A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2])


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vmag(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
