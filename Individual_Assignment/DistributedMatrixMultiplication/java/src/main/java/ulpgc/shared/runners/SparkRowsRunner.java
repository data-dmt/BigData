package ulpgc.shared.runners;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.executor.TaskMetrics;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import scala.Tuple2;
import ulpgc.shared.Utils;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public class SparkRowsRunner implements Runner {
    @Override public String mode() { return "spark-rows"; }

    static class MetricsListener extends SparkListener {
        private final AtomicLong tasks = new AtomicLong(0);
        private final AtomicLong shuffleReadBytes = new AtomicLong(0);
        private final AtomicLong shuffleWriteBytes = new AtomicLong(0);
        private final AtomicLong memSpilledBytes = new AtomicLong(0);
        private final AtomicLong diskSpilledBytes = new AtomicLong(0);
        private final AtomicLong executorRunTimeMs = new AtomicLong(0);
        private final AtomicLong jvmGCTimeMs = new AtomicLong(0);
        private final AtomicLong resultSizeBytes = new AtomicLong(0);

        @Override
        public void onTaskEnd(SparkListenerTaskEnd taskEnd) {
            tasks.incrementAndGet();
            TaskMetrics tm = taskEnd.taskMetrics();
            if (tm == null) return;
            executorRunTimeMs.addAndGet(tm.executorRunTime());
            jvmGCTimeMs.addAndGet(tm.jvmGCTime());
            memSpilledBytes.addAndGet(tm.memoryBytesSpilled());
            diskSpilledBytes.addAndGet(tm.diskBytesSpilled());
            resultSizeBytes.addAndGet(tm.resultSize());
            if (tm.shuffleReadMetrics() != null) {
                shuffleReadBytes.addAndGet(tm.shuffleReadMetrics().totalBytesRead());
            }
            if (tm.shuffleWriteMetrics() != null) {
                shuffleWriteBytes.addAndGet(tm.shuffleWriteMetrics().bytesWritten());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new HashMap<>();
            m.put("tasks", tasks.get());
            m.put("shuffleReadBytes", shuffleReadBytes.get());
            m.put("shuffleWriteBytes", shuffleWriteBytes.get());
            m.put("memoryBytesSpilled", memSpilledBytes.get());
            m.put("diskBytesSpilled", diskSpilledBytes.get());
            m.put("executorRunTimeMs", executorRunTimeMs.get());
            m.put("jvmGCTimeMs", jvmGCTimeMs.get());
            m.put("resultSizeBytes", resultSizeBytes.get());
            return m;
        }
    }

    @Override
    public RunOut run(Utils.Case c, Utils.Config cfg) {
        SparkConf conf = new SparkConf()
                .setMaster(cfg.sparkMaster)
                .setAppName(cfg.sparkAppName)
                .set("spark.local.dir", cfg.sparkLocalDir)
                .set("spark.default.parallelism", String.valueOf(cfg.sparkPartitions))
                .set("spark.sql.shuffle.partitions", String.valueOf(cfg.sparkPartitions));

        try (JavaSparkContext sc = new JavaSparkContext(conf)) {

            MetricsListener listener = new MetricsListener();
            sc.sc().addSparkListener(listener);
            int m = c.m(), n = c.n(), p = c.p();
            int parts = cfg.sparkPartitions;
            float[] B = Utils.randMatrixRowMajor(n, p, 1);
            Broadcast<float[]> bB = sc.broadcast(B);
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < parts; i++) idx.add(i);
            JavaRDD<Integer> partIdx = sc.parallelize(idx, parts);
            int rowsPerPart = (int) Math.ceil((double) m / parts);

            JavaPairRDD<Integer, float[]> rows = partIdx.flatMapToPair(pi -> {
                int start = pi * rowsPerPart;
                int end = Math.min(m, start + rowsPerPart);
                Random r = new Random(1000L + pi);

                List<Tuple2<Integer, float[]>> out = new ArrayList<>();
                for (int i = start; i < end; i++) {
                    float[] a = new float[n];
                    for (int k = 0; k < n; k++) a[k] = (float) r.nextGaussian();
                    out.add(new Tuple2<>(i, a));
                }
                return out.iterator();
            });

            long t0 = System.nanoTime();
            Double checksum = rows.map(t -> {
                float[] a = t._2;
                float[] Bb = bB.value();
                double sum = 0.0;
                for (int j = 0; j < p; j++) {
                    double acc = 0.0;
                    for (int k = 0; k < n; k++) acc += (double) a[k] * (double) Bb[k * p + j];
                    sum += acc;
                }
                return sum;
            }).reduce(Double::sum);

            double elapsed = (System.nanoTime() - t0) / 1e9;
            Map<String, Object> extra = new HashMap<>();
            extra.put("partitions", parts);
            extra.put("rowsPerPart", rowsPerPart);
            extra.put("broadcastB_bytes", (long) B.length * 4L);
            extra.putAll(listener.toMap());

            return new RunOut(elapsed, checksum, extra);
        }
    }
}
