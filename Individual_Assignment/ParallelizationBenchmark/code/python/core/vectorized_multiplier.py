from __future__ import annotations
from typing import List
import numpy as np
from .matrix_utils import Matrix


def multiply_vectorized(A: Matrix, B: Matrix) -> Matrix:
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C_np = A_np @ B_np
    C: Matrix = C_np.tolist()
    return C
