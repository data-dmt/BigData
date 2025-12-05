package ulpgc.shared.matrix;
import java.util.Arrays;

public class BasicMatrixMultiplier implements MatrixMultiplier {
    @Override
    public String getName() {
        return "Basic";
    }

    @Override
    public void multiply(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        for (int i = 0; i < n; i++) {
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
    }
}
