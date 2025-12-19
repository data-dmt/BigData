package ulpgc.shared;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.File;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class Utils {
    public record Case(int m, int n, int p) {}

    public static class Result {
        public String mode;
        public int m, n, p;
        public String dtype;
        public boolean ok;
        public double elapsedSec;
        public Double checksum;
        public String notes;
        public Map<String, Object> extra;

        public Result(String mode, int m, int n, int p, String dtype,
                      boolean ok, double elapsedSec, Double checksum,
                      String notes, Map<String, Object> extra) {
            this.mode = mode;
            this.m = m; this.n = n; this.p = p;
            this.dtype = dtype;
            this.ok = ok;
            this.elapsedSec = elapsedSec;
            this.checksum = checksum;
            this.notes = notes;
            this.extra = extra;
        }
    }

    public static class Config {
        public List<String> modes = List.of("basic", "parallel", "spark-rows", "spark-blocks");

        public List<Case> cases = List.of(
                new Case(512, 512, 512),
                new Case(1024, 1024, 1024)
        );

        public String dtype = "float32";

        public int parallelThreads = 4;
        public int parallelChunkRows = 128;

        public String sparkMaster = "local[*]";
        public String sparkAppName = "DMM-Java";
        public String sparkLocalDir = Path.of("./spark_tmp").toAbsolutePath().toString();
        public int sparkPartitions = 8;

        public int blockSize = 128;

        public long minFreeDiskBytes = 6_000_000_000L;
        public boolean cleanupSparkLocalDir = true;

        public String outputDir = Path.of("./results").toAbsolutePath().toString();
    }

    public static long freeDiskBytes(Path path) throws Exception {
        return Files.getFileStore(path).getUsableSpace();
    }

    public static void ensureDir(Path p) throws Exception {
        Files.createDirectories(p);
    }

    public static void wipeDir(Path p) {
        try {
            if (Files.exists(p)) {
                Files.walk(p).sorted(Comparator.reverseOrder()).forEach(x -> {
                    try { Files.deleteIfExists(x); } catch (Exception ignored) {}
                });
            }
            Files.createDirectories(p);
        } catch (Exception ignored) {}
    }

    public static Path saveJson(String outputDir, Config cfg, List<Result> results) throws Exception {
        ensureDir(Path.of(outputDir));
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

        Map<String, Object> payload = new HashMap<>();
        payload.put("created_at", LocalDateTime.now().toString());
        payload.put("config", cfg);
        payload.put("results", results);

        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        Path out = Path.of(outputDir).resolve("dmm_results_" + ts + ".json");
        mapper.writeValue(new File(out.toString()), payload);
        return out;
    }

    public static float[] randMatrixRowMajor(int rows, int cols, long seed) {
        Random r = new Random(seed);
        float[] out = new float[rows * cols];
        for (int i = 0; i < out.length; i++) out[i] = (float) r.nextGaussian();
        return out;
    }

    public static double mulChecksum(float[] A, float[] B, int m, int n, int p) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            int aRow = i * n;
            for (int j = 0; j < p; j++) {
                double acc = 0.0;
                for (int k = 0; k < n; k++) acc += (double) A[aRow + k] * (double) B[k * p + j];
                sum += acc;
            }
        }
        return sum;
    }

    public static float[] blockMul(float[] Ablk, float[] Bblk, int bs) {
        float[] C = new float[bs * bs];
        for (int i = 0; i < bs; i++) {
            int aRow = i * bs;
            for (int j = 0; j < bs; j++) {
                double acc = 0.0;
                for (int k = 0; k < bs; k++) acc += (double) Ablk[aRow + k] * (double) Bblk[k * bs + j];
                C[i * bs + j] = (float) acc;
            }
        }
        return C;
    }

    public static float[] blockAdd(float[] x, float[] y) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] + y[i];
        return out;
    }

    public static double sumBlock(float[] blk) {
        double s = 0.0;
        for (float v : blk) s += v;
        return s;
    }
}

