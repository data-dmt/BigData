package ulpgc.shared.matrix;

public interface MatrixMultiplier {
    String getName();

    void multiply(double[] A, double[] B, double[] C, int n);
}
