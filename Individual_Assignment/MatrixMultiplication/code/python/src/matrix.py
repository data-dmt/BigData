from __future__ import annotations
import random
from typing import List, Literal

Order = Literal["ijk","ikj","jik"]

def rand_matrix(n: int, seed: int) -> List[List[float]]:
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(n)] for _ in range(n)]

def zeros(n: int) -> List[List[float]]:
    return [[0.0]*n for _ in range(n)]

def matmul(A: List[List[float]], B: List[List[float]], order: Order="ikj") -> List[List[float]]:
    n = len(A)
    C = zeros(n)
    if order == "ijk":
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i][k]*B[k][j]
                C[i][j] = s
    elif order == "ikj":
        for i in range(n):
            for k in range(n):
                aik = A[i][k]
                for j in range(n):
                    C[i][j] += aik*B[k][j]
    else:
        for j in range(n):
            for i in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i][k]*B[k][j]
                C[i][j] = s
    return C
