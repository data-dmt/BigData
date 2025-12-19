import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple, List
import numpy as np
from utils import Case, dtype_np


def run_parallel(case: Case, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dt = dtype_np(cfg["dtype"])
    threads = int(cfg["parallel_threads"])
    chunk_rows = int(cfg["parallel_chunk_rows"])
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    t0 = time.time()
    A = rng_a.standard_normal((case.m, case.n), dtype=dt)
    B = rng_b.standard_normal((case.n, case.p), dtype=dt)

    def chunk_sum(i0: int, i1: int) -> float:
        C_part = A[i0:i1, :] @ B
        return float(np.sum(C_part))

    futures = []
    checksum = 0.0

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i0 in range(0, case.m, chunk_rows):
            i1 = min(case.m, i0 + chunk_rows)
            futures.append(ex.submit(chunk_sum, i0, i1))
        for f in as_completed(futures):
            checksum += float(f.result())

    elapsed = time.time() - t0
    extra = {"threads": threads, "chunkRows": chunk_rows}
    return {"elapsed_s": float(elapsed), "checksum": float(checksum), "extra": extra}
