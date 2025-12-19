import time
from typing import Any, Dict
import numpy as np
from utils import Case, dtype_np, spark_make_session, spark_collect_metrics_from_ui


def run_spark_blocks(case: Case, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dt = dtype_np(cfg["dtype"])
    parts = int(cfg["spark_partitions"])
    bs = int(cfg["block_size"])
    m, n, p = case.m, case.n, case.p
    if (m % bs) != 0 or (n % bs) != 0 or (p % bs) != 0:
        raise ValueError(f"m,n,p must be multiples of block_size={bs}")

    nbI, nbK, nbJ = m // bs, n // bs, p // bs
    spark = spark_make_session(cfg, run_tag=f"blocks_{m}_{n}_{p}")
    sc = spark.sparkContext
    part_ids = sc.parallelize(range(parts), parts)
    total_A = nbI * nbK
    total_B = nbK * nbJ

    def make_A_blocks(part_id: int):
        rng = np.random.default_rng(2000 + part_id)
        start = (total_A * part_id) // parts
        end = (total_A * (part_id + 1)) // parts
        out = []
        for t in range(start, end):
            bi = t // nbK
            bk = t % nbK
            blk = rng.standard_normal((bs, bs), dtype=dt)
            out.append((bk, ("A", bi, blk)))
        return out

    def make_B_blocks(part_id: int):
        rng = np.random.default_rng(3000 + part_id)
        start = (total_B * part_id) // parts
        end = (total_B * (part_id + 1)) // parts
        out = []
        for t in range(start, end):
            bk = t // nbJ
            bj = t % nbJ
            blk = rng.standard_normal((bs, bs), dtype=dt)
            out.append((bk, ("B", bj, blk)))
        return out

    A = part_ids.flatMap(make_A_blocks)
    B = part_ids.flatMap(make_B_blocks)
    t0 = time.time()
    grouped = A.cogroup(B)

    def produce_partials(kv):
        _k, (as_iter, bs_iter) = kv
        a_list = list(as_iter)
        b_list = list(bs_iter)
        out = []
        for (tagA, i, Ablk) in a_list:
            if tagA != "A":
                continue
            for (tagB, j, Bblk) in b_list:
                if tagB != "B":
                    continue
                Cblk = Ablk @ Bblk
                out.append(((i, j), Cblk))
        return out
    partials = grouped.flatMap(produce_partials)

    def add_blocks(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    Cblocks = partials.reduceByKey(add_blocks)
    checksum = float(Cblocks.map(lambda kv: float(np.sum(kv[1]))).sum())
    elapsed = time.time() - t0
    ui_metrics = spark_collect_metrics_from_ui(spark)
    spark.stop()
    extra = {
        "partitions": parts,
        "blockSize": bs,
        "blocksI": nbI,
        "blocksK": nbK,
        "blocksJ": nbJ,
        "notes": "Block-based Spark approach: shuffle expected (cogroup + reduceByKey).",
    }
    extra.update(ui_metrics)

    return {"elapsed_s": float(elapsed), "checksum": float(checksum), "extra": extra}
