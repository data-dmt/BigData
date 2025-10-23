import argparse, time, statistics, json, tracemalloc
from src.matrix import rand_matrix, matmul

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1024)
    ap.add_argument("-r", "--runs", type=int, default=5)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-o", "--order", choices=["ijk","ikj","jik"], default="ikj")
    ap.add_argument("--mem", action="store_true", help="medir memoria (tracemalloc)")
    args = ap.parse_args()

    A = rand_matrix(args.n, args.seed)
    B = rand_matrix(args.n, args.seed+1)

    times = []
    peak = None
    if args.mem: tracemalloc.start()

    for _ in range(args.runs):
        t0 = time.perf_counter()
        C = matmul(A, B, args.order)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if C[0][0] == 1.23456789: print("!")  # evita DCE

    if args.mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    out = {
        "lang": "Python",
        "n": args.n,
        "runs": args.runs,
        "order": args.order,
        "mean_sec": statistics.fmean(times),
        "stdev_sec": statistics.pstdev(times) if len(times) > 1 else 0.0
    }
    if peak is not None: out["peak_bytes"] = int(peak)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
