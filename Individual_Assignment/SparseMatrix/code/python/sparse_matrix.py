import numpy as np
from scipy.sparse import csr_matrix


def dense_to_csr(A: np.ndarray, tol: float = 1e-12) -> csr_matrix:
    A_filtered = A.copy()
    A_filtered[np.abs(A_filtered) <= tol] = 0.0
    return csr_matrix(A_filtered)


def csr_matmul_dense(A_csr: csr_matrix, B: np.ndarray) -> np.ndarray:
    return A_csr.dot(B)
