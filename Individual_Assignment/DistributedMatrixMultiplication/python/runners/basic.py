import time
from typing import Any, Dict

import numpy as np

from utils import Case, dtype_np


def run_basic(case: Case, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dt = dtype_np(cfg["dtype"])
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    t0 = time.time()
    A = rng_a.standard_normal((case.m, case.n), dtype=dt)
    B = rng_b.standard_normal((case.n, case.p), dtype=dt)
    C = A @ B
    checksum = float(np.sum(C))
    elapsed = time.time() - t0
    return {"elapsed_s": float(elapsed), "checksum": checksum, "extra": {}}
