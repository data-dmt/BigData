package ulpgc.shared.matrix;
import java.util.Random;

public final class MatrixUtils {
    private MatrixUtils() {
    }

    public static double[] alloc(int n) {
        return new double[n * n];
    }

    public static void randomFill(double[] m, int n, Random rng) {
        int total = n * n;
        for (int i = 0; i < total; i++) {
            m[i] = rng.nextDouble();
        }
    }

    public static double elapsedMs(long startNs, long endNs) {
        return (endNs - startNs) / 1_000_000.0;
    }
}
