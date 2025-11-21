package ulpgc.shared;

public class SparseMatrixCSR {
    public final int n;
    public final int nnz;
    public final double[] val;
    public final int[] colIdx;
    public final int[] rowPtr;

    public SparseMatrixCSR(int n, int nnz, double[] val, int[] colIdx, int[] rowPtr) {
        this.n = n;
        this.nnz = nnz;
        this.val = val;
        this.colIdx = colIdx;
        this.rowPtr = rowPtr;
    }

    public static SparseMatrixCSR fromDense(double[][] A, double tol) {
        int n = A.length;
        int count = 0;
        for (int i = 0; i < n; i++) {
            double[] row = A[i];
            for (int j = 0; j < n; j++) {
                if (Math.abs(row[j]) > tol) {
                    count++;
                }
            }
        }

        double[] val = new double[count];
        int[] colIdx = new int[count];
        int[] rowPtr = new int[n + 1];

        int nnzCounter = 0;
        for (int i = 0; i < n; i++) {
            rowPtr[i] = nnzCounter;
            double[] row = A[i];
            for (int j = 0; j < n; j++) {
                double v = row[j];
                if (Math.abs(v) > tol) {
                    val[nnzCounter] = v;
                    colIdx[nnzCounter] = j;
                    nnzCounter++;
                }
            }
        }
        rowPtr[n] = nnzCounter;
        return new SparseMatrixCSR(n, nnzCounter, val, colIdx, rowPtr);
    }

    public double[][] multiplyDense(double[][] B) {
        int n = this.n;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            int rowStart = rowPtr[i];
            int rowEnd = rowPtr[i + 1];
            double[] Crow = C[i];
            for (int k = rowStart; k < rowEnd; k++) {
                int colA = colIdx[k];
                double a = val[k];
                double[] Brow = B[colA];
                for (int j = 0; j < n; j++) {
                    Crow[j] += a * Brow[j];
                }
            }
        }
        return C;
    }
}
