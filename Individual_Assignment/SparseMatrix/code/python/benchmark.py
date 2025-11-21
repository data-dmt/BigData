import time
from dataclasses import dataclass
import numpy as np
from dense_matrix import random_dense_matrix, matmul_basic, matmul_blocked, equal_dense
from sparse_matrix import dense_to_csr, csr_matmul_dense

@dataclass
class BenchmarkResult:
    n: int
    input_density: float
    nnz_density: float
    time_basic: float
    time_blocked: float
    time_sparse: float

def run_benchmark_case(n: int, density: float, block_size: int) -> BenchmarkResult:
    A = random_dense_matrix(n, density)
    B = random_dense_matrix(n, density)

    t0 = time.perf_counter()
    C_basic = matmul_basic(A, B)
    t1 = time.perf_counter()
    time_basic = t1 - t0

    t0 = time.perf_counter()
    C_blocked = matmul_blocked(A, B, block_size)
    t1 = time.perf_counter()
    time_blocked = t1 - t0

    if n <= 128 and not equal_dense(C_basic, C_blocked, tol=1e-8):
        print(f"Warning: C_basic and C_blocked differ for n={n}")

    A_csr = dense_to_csr(A, tol=1e-12)
    t0 = time.perf_counter()
    C_sparse = csr_matmul_dense(A_csr, B)
    t1 = time.perf_counter()
    time_sparse = t1 - t0

    if n <= 128 and not equal_dense(C_basic, C_sparse, tol=1e-8):
        print(f"Warning: C_basic and C_sparse differ for n={n}")

    nnz_density = A_csr.nnz / (n * n)

    return BenchmarkResult(
        n=n,
        input_density=density,
        nnz_density=nnz_density,
        time_basic=time_basic,
        time_blocked=time_blocked,
        time_sparse=time_sparse,
    )
