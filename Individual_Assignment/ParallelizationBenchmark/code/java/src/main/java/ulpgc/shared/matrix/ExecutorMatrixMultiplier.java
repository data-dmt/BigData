package ulpgc.shared.matrix;
import java.util.Arrays;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class ExecutorMatrixMultiplier implements MatrixMultiplier {
    private final int numThreads;

    public ExecutorMatrixMultiplier(int numThreads) {
        this.numThreads = numThreads;
    }

    public int getNumThreads() {
        return numThreads;
    }

    @Override
    public String getName() {
        return "Executor(" + numThreads + " threads)";
    }

    @Override
    public void multiply(double[] A, double[] B, double[] C, int n) {
        if (numThreads <= 1) {
            new BasicMatrixMultiplier().multiply(A, B, C, n);
            return;
        }
        Arrays.fill(C, 0.0);
        ExecutorService pool = Executors.newFixedThreadPool(numThreads);
        AtomicInteger tasksDone = new AtomicInteger(0);
        Semaphore allDone = new Semaphore(0);

        int rowsPerTask = n / numThreads;
        int extra = n % numThreads;
        int currentRow = 0;
        for (int t = 0; t < numThreads; t++) {
            int start = currentRow;
            int count = rowsPerTask + (t < extra ? 1 : 0);
            int end = start + count;
            currentRow = end;
            final int rowStart = start;
            final int rowEnd = end;
            pool.submit(() -> {
                for (int i = rowStart; i < rowEnd; i++) {
                    int baseAi = i * n;
                    int baseCi = i * n;
                    for (int k = 0; k < n; k++) {
                        double aik = A[baseAi + k];
                        int baseBk = k * n;
                        for (int j = 0; j < n; j++) {
                            C[baseCi + j] += aik * B[baseBk + j];
                        }
                    }
                }
                if (tasksDone.incrementAndGet() == numThreads) {
                    allDone.release();
                }
            });
        }
        try {
            allDone.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            pool.shutdown();
        }
    }
}
