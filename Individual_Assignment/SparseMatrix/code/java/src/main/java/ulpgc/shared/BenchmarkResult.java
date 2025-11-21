package ulpgc.shared;

public class BenchmarkResult {
    private final int n;
    private final double inputDensity;
    private final double nnzDensity;
    private final double timeBasic;
    private final double timeBlocked;
    private final double timeSparse;

    public BenchmarkResult(int n, double inputDensity, double nnzDensity,
                           double timeBasic, double timeBlocked, double timeSparse) {
        this.n = n;
        this.inputDensity = inputDensity;
        this.nnzDensity = nnzDensity;
        this.timeBasic = timeBasic;
        this.timeBlocked = timeBlocked;
        this.timeSparse = timeSparse;
    }

    public String toCsvLine() {
        return String.format("%d,%.6f,%.6f,%.6f,%.6f,%.6f",
                n, inputDensity, nnzDensity, timeBasic, timeBlocked, timeSparse);
    }
}
