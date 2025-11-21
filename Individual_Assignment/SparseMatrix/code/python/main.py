import csv
import os
from benchmark import run_benchmark_case

def get_output_path() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "results_python.csv")

def main():
    output_path = get_output_path()
    sizes = [64, 128, 256]  # you can add 512, 1024... but Python loops will be slow
    densities = [1.0, 0.1, 0.01, 0.001]
    block_size = 32
    print("n,input_density,nnz_density,time_basic,time_blocked,time_sparse")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["n", "input_density", "nnz_density", "time_basic", "time_blocked", "time_sparse"]
        )
        for density in densities:
            for n in sizes:
                result = run_benchmark_case(n, density, block_size)
                row = [
                    result.n,
                    f"{result.input_density:.6f}",
                    f"{result.nnz_density:.6f}",
                    f"{result.time_basic:.6f}",
                    f"{result.time_blocked:.6f}",
                    f"{result.time_sparse:.6f}",
                ]
                writer.writerow(row)
                print(",".join(map(str, row)))
    print(f"\nPython results saved to {output_path}")


if __name__ == "__main__":
    main()
