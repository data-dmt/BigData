package ulpgc.shared;

import java.util.Random;

public class DenseMatrix {
    public static double[][] randomDenseMatrix(Random rng, int n, double density) {
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double r = rng.nextDouble();
                if (r < density) {
                    A[i][j] = rng.nextDouble();
                } else {
                    A[i][j] = 0.0;
                }
            }
        }
        return A;
    }

    public static double[][] matmulBasic(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = A[i][k];
                if (aik == 0.0) continue;
                for (int j = 0; j < n; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
        return C;
    }

    public static double[][] matmulBlocked(double[][] A, double[][] B, int block) {
        int n = A.length;
        double[][] C = new double[n][n];
        double[][] BT = transpose(B);

        for (int ii = 0; ii < n; ii += block) {
            int iMax = Math.min(ii + block, n);
            for (int jj = 0; jj < n; jj += block) {
                int jMax = Math.min(jj + block, n);
                for (int kk = 0; kk < n; kk += block) {
                    int kMax = Math.min(kk + block, n);
                    for (int i = ii; i < iMax; i++) {
                        for (int j = jj; j < jMax; j++) {
                            double sum = C[i][j];
                            double[] Arow = A[i];
                            double[] BTrow = BT[j];
                            for (int k = kk; k < kMax; k++) {
                                sum += Arow[k] * BTrow[k];
                            }
                            C[i][j] = sum;
                        }
                    }
                }
            }
        }

        return C;
    }

    public static double[][] transpose(double[][] M) {
        int n = M.length;
        double[][] MT = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] row = M[i];
            for (int j = 0; j < n; j++) {
                MT[j][i] = row[j];
            }
        }
        return MT;
    }

    public static boolean equalDense(double[][] A, double[][] B, double tol) {
        int n = A.length;
        for (int i = 0; i < n; i++) {
            double[] rowA = A[i];
            double[] rowB = B[i];
            for (int j = 0; j < n; j++) {
                if (Math.abs(rowA[j] - rowB[j]) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
}
