package ulpgc.shared;

import ulpgc.shared.runners.*;
import java.nio.file.Path;
import java.util.*;

public class App {
    public static void main(String[] args) throws Exception {
        Utils.Config cfg = new Utils.Config();
        Utils.ensureDir(Path.of(cfg.outputDir));
        Utils.ensureDir(Path.of(cfg.sparkLocalDir));
        Map<String, Runner> runners = Map.of(
                "basic", new BasicRunner(),
                "parallel", new ParallelRunner(),
                "spark-rows", new SparkRowsRunner(),
                "spark-blocks", new SparkBlocksRunner()
        );

        List<Utils.Result> results = new ArrayList<>();

        for (Utils.Case c : cfg.cases) {
            for (String mode : cfg.modes) {
                Runner r = runners.get(mode);
                if (r == null) {
                    results.add(new Utils.Result(mode, c.m(), c.n(), c.p(), cfg.dtype, false, 0.0, null,
                            "Unknown mode", Map.of()));
                    continue;
                }

                if (mode.startsWith("spark")) {
                    long free = Utils.freeDiskBytes(Path.of(cfg.sparkLocalDir));
                    if (free < cfg.minFreeDiskBytes) {
                        results.add(new Utils.Result(mode, c.m(), c.n(), c.p(), cfg.dtype, false, 0.0, null,
                                "Skipped: not enough disk space in sparkLocalDir (" + (free / 1e9) + " GB)",
                                Map.of("freeDiskBytes", free)));
                        continue;
                    }
                }

                try {
                    Runner.RunOut out = r.run(c, cfg);
                    results.add(new Utils.Result(mode, c.m(), c.n(), c.p(), cfg.dtype, true,
                            out.elapsedSec(), out.checksum(), "OK", out.extra()));

                    System.out.printf("[OK] %s %dx%d·%dx%d  t=%.3fs  chk=%.3f%n",
                            mode, c.m(), c.n(), c.n(), c.p(), out.elapsedSec(), out.checksum());

                } catch (Exception e) {
                    results.add(new Utils.Result(mode, c.m(), c.n(), c.p(), cfg.dtype, false,
                            0.0, null, "ERROR: " + e.getClass().getSimpleName() + ": " + e.getMessage(), Map.of()));
                    System.out.printf("[FAIL] %s %dx%d·%dx%d  %s%n",
                            mode, c.m(), c.n(), c.n(), c.p(), e.toString());
                }
            }
        }

        var out = Utils.saveJson(cfg.outputDir, cfg, results);
        System.out.println("JSON saved in: " + out);

        if (cfg.cleanupSparkLocalDir) {
            Utils.wipeDir(Path.of(cfg.sparkLocalDir));
        }
    }
}
