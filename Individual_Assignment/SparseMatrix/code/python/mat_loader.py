from typing import Optional
import numpy as np
from scipy.io import loadmat


def load_matrix_from_mat(path: str, key: Optional[str] = None) -> np.ndarray:
    data = loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]

    if not keys:
        raise ValueError("No matrix-like variables found in the .mat file.")

    if key is None:
        print("Available variables in .mat file:", keys)
        key = keys[0]
        print(f"Using variable '{key}' by default.")

    A = data[key]
    if A.ndim != 2:
        raise ValueError(f"Variable '{key}' is not a 2D matrix (shape={A.shape}).")

    print(f"Loaded matrix '{key}' with shape {A.shape}")
    return A
