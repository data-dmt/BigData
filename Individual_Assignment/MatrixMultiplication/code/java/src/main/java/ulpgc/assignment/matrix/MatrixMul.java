package ulpgc.assignment.matrix;

import java.util.Random;

public class MatrixMul {
    public enum Order { IJK, IKJ, JIK }

    public static double[][] randMatrix(int n, long seed){
        Random r = new Random(seed);
        double[][] m = new double[n][n];
        for(int i=0;i<n;i++) for(int j=0;j<n;j++) m[i][j]=r.nextDouble();
        return m;
    }
    public static double[][] zeros(int n){ return new double[n][n]; }

    public static double[][] mul(double[][] A, double[][] B, Order order){
        int n=A.length;
        double[][] C = zeros(n);
        switch(order){
            case IJK -> {
                for(int i=0;i<n;i++)
                    for(int j=0;j<n;j++){
                        double s=0;
                        for(int k=0;k<n;k++) s += A[i][k]*B[k][j];
                        C[i][j]=s;
                    }
            }
            case IKJ -> {
                for(int i=0;i<n;i++)
                    for(int k=0;k<n;k++){
                        double aik=A[i][k];
                        for(int j=0;j<n;j++)
                            C[i][j]+=aik*B[k][j];
                    }
            }
            case JIK -> {
                for(int j=0;j<n;j++)
                    for(int i=0;i<n;i++){
                        double s=0;
                        for(int k=0;k<n;k++) s += A[i][k]*B[k][j];
                        C[i][j]=s;
                    }
            }
        }
        return C;
    }
}
