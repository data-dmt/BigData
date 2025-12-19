package ulpgc.shared.runners;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.executor.TaskMetrics;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import scala.Tuple2;
import ulpgc.shared.Utils;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public class SparkBlocksRunner implements Runner {
    @Override public String mode() { return "spark-blocks"; }

    static class Block implements Serializable {
        public char kind;
        public int index;
        public float[] data;
        public Block(char kind, int index, float[] data) {
            this.kind = kind;
            this.index = index;
            this.data = data;
        }
    }

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
        int m = c.m(), n = c.n(), p = c.p();
        int bs = cfg.blockSize;
        if (m % bs != 0 || n % bs != 0 || p % bs != 0) {
            throw new IllegalArgumentException("m,n,p must be multiples of blockSize=" + bs);
        }
        int nbI = m / bs, nbK = n / bs, nbJ = p / bs;
        int parts = cfg.sparkPartitions;
        SparkConf conf = new SparkConf()
                .setMaster(cfg.sparkMaster)
                .setAppName(cfg.sparkAppName)
                .set("spark.local.dir", cfg.sparkLocalDir)
                .set("spark.default.parallelism", String.valueOf(parts))
                .set("spark.sql.shuffle.partitions", String.valueOf(parts));

        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            MetricsListener listener = new MetricsListener();
            sc.sc().addSparkListener(listener);
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < parts; i++) idx.add(i);
            JavaRDD<Integer> partIdx = sc.parallelize(idx, parts);
            JavaPairRDD<Integer, Block> A = partIdx.flatMapToPair(pi -> {
                Random r = new Random(2000L + pi);
                int total = nbI * nbK;
                int start = (total * pi) / parts;
                int end = (total * (pi + 1)) / parts;

                List<Tuple2<Integer, Block>> out = new ArrayList<>();
                for (int t = start; t < end; t++) {
                    int i = t / nbK;
                    int k = t % nbK;
                    float[] blk = new float[bs * bs];
                    for (int x = 0; x < blk.length; x++) blk[x] = (float) r.nextGaussian();
                    out.add(new Tuple2<>(k, new Block('A', i, blk)));
                }
                return out.iterator();
            });

            JavaPairRDD<Integer, Block> B = partIdx.flatMapToPair(pi -> {
                Random r = new Random(3000L + pi);
                int total = nbK * nbJ;
                int start = (total * pi) / parts;
                int end = (total * (pi + 1)) / parts;

                List<Tuple2<Integer, Block>> out = new ArrayList<>();
                for (int t = start; t < end; t++) {
                    int k = t / nbJ;
                    int j = t % nbJ;
                    float[] blk = new float[bs * bs];
                    for (int x = 0; x < blk.length; x++) blk[x] = (float) r.nextGaussian();
                    out.add(new Tuple2<>(k, new Block('B', j, blk)));
                }
                return out.iterator();
            });

            long t0 = System.nanoTime();
            JavaPairRDD<Integer, Tuple2<Iterable<Block>, Iterable<Block>>> grouped = A.cogroup(B);
            JavaPairRDD<Tuple2<Integer, Integer>, float[]> partial = grouped.flatMapToPair(kv -> {
                Iterable<Block> As = kv._2._1;
                Iterable<Block> Bs = kv._2._2;

                List<Block> aList = new ArrayList<>();
                for (Block b : As) aList.add(b);

                List<Block> bList = new ArrayList<>();
                for (Block b : Bs) bList.add(b);

                List<Tuple2<Tuple2<Integer, Integer>, float[]>> out = new ArrayList<>();
                for (Block aBlk : aList) {
                    for (Block bBlk : bList) {
                        float[] cBlk = Utils.blockMul(aBlk.data, bBlk.data, bs);
                        out.add(new Tuple2<>(new Tuple2<>(aBlk.index, bBlk.index), cBlk));
                    }
                }
                return out.iterator();
            });

            JavaPairRDD<Tuple2<Integer, Integer>, float[]> Cblocks =
                    partial.reduceByKey(Utils::blockAdd);

            Double checksum = Cblocks.map(kv -> Utils.sumBlock(kv._2)).reduce(Double::sum);
            double elapsed = (System.nanoTime() - t0) / 1e9;
            Map<String, Object> extra = new HashMap<>();
            extra.put("blockSize", bs);
            extra.put("blocksI", nbI);
            extra.put("blocksK", nbK);
            extra.put("blocksJ", nbJ);
            extra.put("partitions", parts);
            extra.putAll(listener.toMap());

            return new RunOut(elapsed, checksum, extra);
        }
    }
}
