import time
from typing import Any, Dict
import numpy as np
from utils import Case, dtype_np, spark_make_session, spark_collect_metrics_from_ui


def run_spark_rows(case: Case, cfg: Dict[str, Any]) -> Dict[str, Any]:
    dt = dtype_np(cfg["dtype"])
    parts = int(cfg["spark_partitions"])
    spark = spark_make_session(cfg, run_tag=f"rows_{case.m}_{case.n}_{case.p}")
    sc = spark.sparkContext
    rng_b = np.random.default_rng(1)
    B = rng_b.standard_normal((case.n, case.p), dtype=dt)
    bB = sc.broadcast(B)
    rows_per_part = (case.m + parts - 1) // parts

    def part_rows(part_id: int):
        rng = np.random.default_rng(1000 + part_id)
        i0 = part_id * rows_per_part
        i1 = min(case.m, i0 + rows_per_part)
        out = []
        for i in range(i0, i1):
            a = rng.standard_normal((case.n,), dtype=dt)
            out.append((i, a))
        return out

    t0 = time.time()
    idx = list(range(parts))
    rdd = sc.parallelize(idx, parts).flatMap(lambda pid: part_rows(pid))

    def row_checksum(item):
        _, a = item
        Bb = bB.value
        return float(np.sum(a @ Bb))

    checksum = float(rdd.map(row_checksum).sum())
    elapsed = time.time() - t0
    ui_metrics = spark_collect_metrics_from_ui(spark)
    spark.stop()
    extra = {
        "partitions": parts,
        "rowsPerPart": rows_per_part,
        "broadcastB_bytes": int(B.size * B.itemsize),
        "notes": "Row-based Spark approach: broadcast B, shuffle is typically near zero.",
    }
    extra.update(ui_metrics)

    return {"elapsed_s": float(elapsed), "checksum": float(checksum), "extra": extra}
