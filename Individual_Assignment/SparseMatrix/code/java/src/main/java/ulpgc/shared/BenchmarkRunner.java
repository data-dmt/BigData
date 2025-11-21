package ulpgc.shared;

import java.util.Random;

public class BenchmarkRunner {
    private final Random rng;

    public BenchmarkRunner(long seed) {
        this.rng = new Random(seed);
    }

    public BenchmarkResult runBenchmarkCase(int n, double density, int blockSize) {
        double[][] A = DenseMatrix.randomDenseMatrix(rng, n, density);
        double[][] B = DenseMatrix.randomDenseMatrix(rng, n, density);
        long t0 = System.nanoTime();
        double[][] Cbasic = DenseMatrix.matmulBasic(A, B);
        long t1 = System.nanoTime();
        double timeBasic = (t1 - t0) / 1e9;

        t0 = System.nanoTime();
        double[][] Cblocked = DenseMatrix.matmulBlocked(A, B, blockSize);
        t1 = System.nanoTime();
        double timeBlocked = (t1 - t0) / 1e9;
        if (n <= 256 && !DenseMatrix.equalDense(Cbasic, Cblocked, 1e-8)) {
            System.err.printf("Warning: C_basic and C_blocked differ for n=%d%n", n);
        }

        SparseMatrixCSR csr = SparseMatrixCSR.fromDense(A, 1e-12);
        t0 = System.nanoTime();
        double[][] Csparse = csr.multiplyDense(B);
        t1 = System.nanoTime();
        double timeSparse = (t1 - t0) / 1e9;
        if (n <= 256 && !DenseMatrix.equalDense(Cbasic, Csparse, 1e-8)) {
            System.err.printf("Warning: C_basic and C_sparse differ for n=%d%n", n);
        }

        double nnzDensity = csr.nnz / (double) (n * n);
        return new BenchmarkResult(
                n,
                density,
                nnzDensity,
                timeBasic,
                timeBlocked,
                timeSparse
        );
    }
}
