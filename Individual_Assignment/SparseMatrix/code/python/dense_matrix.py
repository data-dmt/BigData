import numpy as np

def random_dense_matrix(n: int, density: float) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    mask = np.random.rand(n, n) < density
    A[mask] = np.random.rand(mask.sum())
    return A


def matmul_basic(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            aik = A[i, k]
            for j in range(n):
                C[i, j] += aik * B[k, j]
    return C


def matmul_blocked(A: np.ndarray, B: np.ndarray, block: int) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros((n, n), dtype=float)
    BT = B.T.copy()

    for ii in range(0, n, block):
        i_max = min(ii + block, n)
        for jj in range(0, n, block):
            j_max = min(jj + block, n)
            for kk in range(0, n, block):
                k_max = min(kk + block, n)
                for i in range(ii, i_max):
                    for j in range(jj, j_max):
                        s = C[i, j]
                        Arow = A[i, :]
                        BTrow = BT[j, :]
                        for k in range(kk, k_max):
                            s += Arow[k] * BTrow[k]
                        C[i, j] = s
    return C


def equal_dense(A: np.ndarray, B: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(A, B, atol=tol)
