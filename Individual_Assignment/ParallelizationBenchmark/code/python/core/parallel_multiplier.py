from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from .matrix_utils import Matrix
from .basic_multiplier import multiply_basic


def _multiply_rows(args: Tuple[Matrix, Matrix, int, int]) -> Matrix:
    A, B, row_start, row_end = args
    n = len(A)
    sub_C: Matrix = [[0.0 for _ in range(n)] for _ in range(row_end - row_start)]
    for local_i, i in enumerate(range(row_start, row_end)):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                sub_C[local_i][j] += aik * B[k][j]
    return sub_C


def multiply_parallel(A: Matrix, B: Matrix, num_processes: int) -> Matrix:
    n = len(A)
    if num_processes <= 1:
        return multiply_basic(A, B)

    rows_per_proc = n // num_processes
    extra = n % num_processes
    tasks = []
    current_row = 0
    for p in range(num_processes):
        start = current_row
        count = rows_per_proc + (1 if p < extra else 0)
        end = start + count
        current_row = end
        tasks.append((A, B, start, end))
    C: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(_multiply_rows, tasks))
    row_offset = 0
    for block in results:
        for row in block:
            C[row_offset] = row
            row_offset += 1
    return C
