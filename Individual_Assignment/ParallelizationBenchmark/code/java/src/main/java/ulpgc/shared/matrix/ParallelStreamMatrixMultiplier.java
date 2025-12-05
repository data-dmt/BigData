package ulpgc.shared.matrix;
import java.util.Arrays;
import java.util.stream.IntStream;

public class ParallelStreamMatrixMultiplier implements MatrixMultiplier {
    @Override
    public String getName() {
        return "ParallelStream";
    }

    @Override
    public void multiply(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        IntStream.range(0, n).parallel().forEach(i -> {
            int baseAi = i * n;
            int baseCi = i * n;
            for (int k = 0; k < n; k++) {
                double aik = A[baseAi + k];
                int baseBk = k * n;
                for (int j = 0; j < n; j++) {
                    C[baseCi + j] += aik * B[baseBk + j];
                }
            }
        });
    }
}
