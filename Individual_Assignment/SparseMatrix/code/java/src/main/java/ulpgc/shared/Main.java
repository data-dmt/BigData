package ulpgc.shared;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    private static Path getOutputPath() throws IOException {
        String userDir = System.getProperty("user.dir"); // project root when run from IntelliJ
        Path outputDir = Paths.get(userDir, "output");
        Files.createDirectories(outputDir);
        return outputDir.resolve("results_java.csv");
    }

    public static void main(String[] args) {
        try {
            Path outputPath = getOutputPath();
            int[] sizes = {64, 128, 256, 512};
            double[] densities = {1.0, 0.1, 0.01, 0.001};
            int blockSize = 32;
            BenchmarkRunner runner = new BenchmarkRunner(System.currentTimeMillis());

            try (BufferedWriter bw = Files.newBufferedWriter(outputPath);
                 PrintWriter out = new PrintWriter(bw)) {
                String header = "n,input_density,nnz_density,time_basic,time_blocked,time_sparse";
                out.println(header);
                System.out.println(header);

                for (double density : densities) {
                    for (int n : sizes) {
                        BenchmarkResult result = runner.runBenchmarkCase(n, density, blockSize);
                        String line = result.toCsvLine();
                        out.println(line);
                        System.out.println(line);
                    }
                }
            }

            System.out.println("\nJava results saved to: " + outputPath.toAbsolutePath());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
