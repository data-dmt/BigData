from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import List
from core.matrix_utils import random_matrix, elapsed_ms, cpu_info
from core.basic_multiplier import multiply_basic
from core.parallel_multiplier import multiply_parallel
from core.vectorized_multiplier import multiply_vectorized


def run_benchmark() -> None:
    sizes: List[int] = [128, 256, 512]
    logical, physical = cpu_info()
    thread_options_temp = [1, physical, logical]
    thread_options = sorted(set(thread_options_temp))
    print(f"Detected logical CPUs (Python): {logical}")
    print(f"Process configurations to test: {thread_options}\n")
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "processes",
            "time_basic_ms", "time_parallel_ms", "time_vectorized_ms",
            "speedup_parallel", "speedup_vectorized",
            "eff_parallel",
        ])

        for n in sizes:
            print(f"===== Matrix size n = {n} =====")
            A = random_matrix(n, seed=12345)
            B = random_matrix(n, seed=54321)

            # Basic
            t1 = time.perf_counter()
            C_basic = multiply_basic(A, B)
            t2 = time.perf_counter()
            time_basic = elapsed_ms(t1, t2)
            print(f"Basic: {time_basic:.2f} ms")

            # Vectorized
            t1 = time.perf_counter()
            C_vec = multiply_vectorized(A, B)
            t2 = time.perf_counter()
            time_vec = elapsed_ms(t1, t2)
            speedup_vec = time_basic / time_vec if time_vec > 0 else 0.0
            print(f"Vectorized (NumPy): {time_vec:.2f} ms "
                  f"(speedup vs basic: {speedup_vec:.2f})")

            for procs in thread_options:
                print(f" -> processes = {procs}")
                t1 = time.perf_counter()
                C_par = multiply_parallel(A, B, procs)
                t2 = time.perf_counter()
                time_par = elapsed_ms(t1, t2)
                speedup_par = time_basic / time_par if time_par > 0 else 0.0
                eff_par = speedup_par / procs if procs > 0 else 0.0
                print(f"Parallel: {time_par:.2f} ms "
                      f"(speedup: {speedup_par:.2f}, eff: {eff_par:.3f})")
                writer.writerow([
                    n, procs,
                    f"{time_basic:.4f}",
                    f"{time_par:.4f}",
                    f"{time_vec:.4f}",
                    f"{speedup_par:.4f}",
                    f"{speedup_vec:.4f}",
                    f"{eff_par:.4f}",
                ])
            print()
    print(f"Results saved in '{csv_path}'")


if __name__ == "__main__":
    run_benchmark()
