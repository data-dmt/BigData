package ulpgc.shared.runners;

import ulpgc.shared.Utils;
import java.util.Map;

public class BasicRunner implements Runner {
    @Override public String mode() { return "basic"; }

    @Override
    public RunOut run(Utils.Case c, Utils.Config cfg) {
        float[] A = Utils.randMatrixRowMajor(c.m(), c.n(), 0);
        float[] B = Utils.randMatrixRowMajor(c.n(), c.p(), 1);
        long t0 = System.nanoTime();
        double chk = Utils.mulChecksum(A, B, c.m(), c.n(), c.p());
        double elapsed = (System.nanoTime() - t0) / 1e9;
        return new RunOut(elapsed, chk, Map.of());
    }
}
