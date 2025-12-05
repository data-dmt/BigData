package ulpgc.shared.matrix;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import static ulpgc.shared.matrix.MatrixUtils.*;

public class BenchmarkRunner {
    private static synchronized void writeCsvLine(PrintWriter out, String line) {
        out.println(line);
    }

    public static void main(String[] args) {
        int[] sizes = {256, 512, 1024};
        int hwThreads = Runtime.getRuntime().availableProcessors();
        if (hwThreads <= 0) hwThreads = 4;
        int physicalCores = Math.max(1, hwThreads / 2);
        int[] threadOptionsTemp = {1, physicalCores, hwThreads};
        int[] threadOptions = Arrays.stream(threadOptionsTemp).distinct().toArray();

        System.out.println("Detected hardware threads (Java): " + hwThreads);
        System.out.println("Thread configurations to test: " + Arrays.toString(threadOptions));
        System.out.println();

        try (PrintWriter out = new PrintWriter(new FileWriter("java_results.csv"))) {
            writeCsvLine(out,
                    "n,threads," +
                            "time_basic_ms,time_exec_ms,time_stream_ms," +
                            "speedup_exec,speedup_stream," +
                            "eff_exec,eff_stream");

            Random rng = new Random(12345L);
            BasicMatrixMultiplier basic = new BasicMatrixMultiplier();
            ParallelStreamMatrixMultiplier streamMult = new ParallelStreamMatrixMultiplier();

            for (int n : sizes) {
                System.out.println("===== Matrix size n = " + n + " =====");
                double[] A = alloc(n);
                double[] B = alloc(n);
                double[] C = alloc(n);
                randomFill(A, n, rng);
                randomFill(B, n, rng);

                // Basic
                long t1 = System.nanoTime();
                basic.multiply(A, B, C, n);
                long t2 = System.nanoTime();
                double timeBasic = elapsedMs(t1, t2);
                System.out.printf("Basic: %.2f ms%n", timeBasic);

                // Parallel Stream
                t1 = System.nanoTime();
                streamMult.multiply(A, B, C, n);
                t2 = System.nanoTime();
                double timeStream = elapsedMs(t1, t2);
                double speedupStream = timeBasic / timeStream;
                double effStream = speedupStream / hwThreads;
                System.out.printf("ParallelStream: %.2f ms (speedup: %.2f, eff ~ %.3f)%n",
                        timeStream, speedupStream, effStream);

                // Executor (for each number of threads)
                for (int threads : threadOptions) {
                    ExecutorMatrixMultiplier execMult = new ExecutorMatrixMultiplier(threads);
                    t1 = System.nanoTime();
                    execMult.multiply(A, B, C, n);
                    t2 = System.nanoTime();
                    double timeExec = elapsedMs(t1, t2);
                    double speedupExec = timeBasic / timeExec;
                    double effExec = speedupExec / threads;
                    System.out.printf("%s: %.2f ms (speedup: %.2f, eff: %.3f)%n",
                            execMult.getName(), timeExec, speedupExec, effExec);
                    String line = String.format(Locale.ROOT,
                            "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                            n, threads,
                            timeBasic, timeExec, timeStream,
                            speedupExec, speedupStream,
                            effExec, effStream);
                    writeCsvLine(out, line);
                }
                System.out.println();
            }
            System.out.println("Results saved in 'java_results.csv'");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
