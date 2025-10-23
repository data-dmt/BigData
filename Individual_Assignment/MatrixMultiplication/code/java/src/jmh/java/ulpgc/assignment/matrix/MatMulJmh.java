package ulpgc.assignment.matrix;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
public class MatMulJmh {

    @State(Scope.Benchmark)
    public static class Data {
        @Param({"64","128","256","512","1024"})
        public int n;

        @Param({"ikj"})
        public String order;

        public double[][] A, B;

        @Setup(Level.Trial)
        public void setup() {
            A = MatrixMul.randMatrix(n, 42);
            B = MatrixMul.randMatrix(n, 43);
        }

        MatrixMul.Order toOrder() {
            return switch (order) {
                case "ijk" -> MatrixMul.Order.IJK;
                case "jik" -> MatrixMul.Order.JIK;
                default -> MatrixMul.Order.IKJ;
            };
        }
    }

    @Benchmark
    public double[][] mul(Data d) {
        return MatrixMul.mul(d.A, d.B, d.toOrder());
    }
}
