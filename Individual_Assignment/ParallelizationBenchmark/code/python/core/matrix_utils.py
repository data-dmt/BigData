from __future__ import annotations
import os
import random
import time
from typing import List, Tuple


Matrix = List[List[float]]

def alloc_matrix(n: int) -> Matrix:
    return [[0.0 for _ in range(n)] for _ in range(n)]

def random_matrix(n: int, seed: int = 12345) -> Matrix:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(n)] for _ in range(n)]

def elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0

def cpu_info() -> Tuple[int, int]:
    logical = os.cpu_count() or 4
    physical = max(1, logical // 2)
    return logical, physical
