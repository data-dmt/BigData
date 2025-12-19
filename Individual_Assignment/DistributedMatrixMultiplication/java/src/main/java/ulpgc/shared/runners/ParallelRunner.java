package ulpgc.shared.runners;

import ulpgc.shared.Utils;
import java.util.*;
import java.util.concurrent.*;

public class ParallelRunner implements Runner {
    @Override public String mode() { return "parallel"; }

    @Override
    public RunOut run(Utils.Case c, Utils.Config cfg) throws Exception {
        float[] A = Utils.randMatrixRowMajor(c.m(), c.n(), 0);
        float[] B = Utils.randMatrixRowMajor(c.n(), c.p(), 1);
        int m = c.m(), n = c.n(), p = c.p();
        int chunk = cfg.parallelChunkRows;
        ExecutorService pool = Executors.newFixedThreadPool(cfg.parallelThreads);
        try {
            long t0 = System.nanoTime();
            List<Future<Double>> futures = new ArrayList<>();
            for (int start = 0; start < m; start += chunk) {
                final int s = start;
                final int e = Math.min(m, start + chunk);
                futures.add(pool.submit(() -> {
                    double sum = 0.0;
                    for (int i = s; i < e; i++) {
                        int aRow = i * n;
                        for (int j = 0; j < p; j++) {
                            double acc = 0.0;
                            for (int k = 0; k < n; k++) acc += (double) A[aRow + k] * (double) B[k * p + j];
                            sum += acc;
                        }
                    }
                    return sum;
                }));
            }

            double checksum = 0.0;
            for (Future<Double> f : futures) checksum += f.get();

            double elapsed = (System.nanoTime() - t0) / 1e9;
            return new RunOut(elapsed, checksum, Map.of("threads", cfg.parallelThreads, "chunkRows", chunk));
        } finally {
            pool.shutdownNow();
        }
    }
}
