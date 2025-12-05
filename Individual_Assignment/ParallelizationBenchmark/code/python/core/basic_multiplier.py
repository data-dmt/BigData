from __future__ import annotations
from typing import List
from .matrix_utils import Matrix


def multiply_basic(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    C: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C
