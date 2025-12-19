import time
from typing import Any, Dict, List

from utils import (
    Case,
    Result,
    SafetyGuard,
    JsonReporter,
    dtype_np,
    ensure_dir,
    wipe_dir,
)

from runners.basic import run_basic
from runners.parallel import run_parallel
from runners.spark_rows import run_spark_rows
from runners.spark_blocks import run_spark_blocks


CONFIG: Dict[str, Any] = {
    "modes": ["basic", "parallel", "spark-rows", "spark-blocks"],
    "cases": [
        Case(512, 512, 512),
        Case(1024, 1024, 1024),
    ],
    "dtype": "float32",
    "max_estimated_dense_bytes": 1_500_000_000,
    "min_free_disk_bytes": 6_000_000_000,

    "parallel_threads": 4,
    "parallel_chunk_rows": 128,

    "spark_master": "local[*]",
    "spark_app_name": "DMM-Python",
    "spark_partitions": 8,
    "spark_local_dir": "./spark_tmp",

    "block_size": 128,

    "spark_eventlog_enabled": True,
    "spark_eventlog_dir": "./spark_eventlog",

    "output_dir": "./results",
    "cleanup_spark_local_dir": True,
}


def run_one(mode: str, case: Case, cfg: Dict[str, Any], guard: SafetyGuard) -> Result:
    dt = dtype_np(cfg["dtype"])
    if mode in ("basic", "parallel"):
        reason = guard.skip_dense_reason(case.m, case.n, case.p, dt)
        if reason:
            print(f"[SKIP] mode={mode} case={case.m}x{case.n}x{case.p} -> {reason}")
            return Result(
                mode=mode, m=case.m, n=case.n, p=case.p,
                dtype=cfg["dtype"],
                ok=False, elapsed_s=0.0, checksum=None,
                notes=reason, extra={}
            )

    if mode.startswith("spark"):
        ok, free = guard.spark_disk_ok(cfg["spark_local_dir"])
        if not ok:
            reason = f"Skipped: low disk space in spark_local_dir ({free/1e9:.2f} GB available)."
            print(f"[SKIP] mode={mode} case={case.m}x{case.n}x{case.p} -> {reason}")
            return Result(
                mode=mode, m=case.m, n=case.n, p=case.p,
                dtype=cfg["dtype"],
                ok=False, elapsed_s=0.0, checksum=None,
                notes=reason, extra={"freeDiskBytes": free}
            )

        if cfg.get("spark_eventlog_enabled", True):
            ok2, free2 = guard.spark_disk_ok(cfg["spark_eventlog_dir"])
            if not ok2:
                reason = f"Skipped: low disk space in spark_eventlog_dir ({free2/1e9:.2f} GB available)."
                print(f"[SKIP] mode={mode} case={case.m}x{case.n}x{case.p} -> {reason}")
                return Result(
                    mode=mode, m=case.m, n=case.n, p=case.p,
                    dtype=cfg["dtype"],
                    ok=False, elapsed_s=0.0, checksum=None,
                    notes=reason, extra={"freeDiskBytes": free2}
                )

    print(f"\n[RUN] mode={mode} case={case.m}x{case.n} · {case.n}x{case.p}")
    t0 = time.time()

    try:
        if mode == "basic":
            out = run_basic(case, cfg)
        elif mode == "parallel":
            out = run_parallel(case, cfg)
        elif mode == "spark-rows":
            out = run_spark_rows(case, cfg)
        elif mode == "spark-blocks":
            out = run_spark_blocks(case, cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        elapsed = float(out["elapsed_s"])
        checksum = out.get("checksum", None)
        extra = out.get("extra", {}) or {}
        print(f"[OK]  mode={mode} elapsed={elapsed:.3f}s checksum={checksum}")
        return Result(
            mode=mode, m=case.m, n=case.n, p=case.p,
            dtype=cfg["dtype"],
            ok=True, elapsed_s=elapsed, checksum=float(checksum) if checksum is not None else None,
            notes="OK", extra=extra
        )

    except Exception as e:
        elapsed = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] mode={mode} after {elapsed:.3f}s -> {msg}")
        return Result(
            mode=mode, m=case.m, n=case.n, p=case.p,
            dtype=cfg["dtype"],
            ok=False, elapsed_s=float(elapsed), checksum=None,
            notes=f"ERROR: {msg}", extra={}
        )


def run_all() -> str:
    ensure_dir(CONFIG["output_dir"])
    ensure_dir(CONFIG["spark_local_dir"])
    if CONFIG.get("spark_eventlog_enabled", True):
        ensure_dir(CONFIG["spark_eventlog_dir"])

    guard = SafetyGuard(
        max_estimated_dense_bytes=int(CONFIG["max_estimated_dense_bytes"]),
        min_free_disk_bytes=int(CONFIG["min_free_disk_bytes"]),
    )
    reporter = JsonReporter(CONFIG["output_dir"])

    results: List[Result] = []
    for case in CONFIG["cases"]:
        for mode in CONFIG["modes"]:
            results.append(run_one(mode, case, CONFIG, guard))

    out_path = reporter.save(CONFIG, results)
    print(f"\n[DONE] Results saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    print("[START] Running DMM experiments (Python).")
    out = run_all()

    if CONFIG.get("cleanup_spark_local_dir", True):
        print("[CLEANUP] Cleaning Spark local temp directory.")
        wipe_dir(CONFIG["spark_local_dir"])

    print(f"[END] Finished. Output: {out}")
