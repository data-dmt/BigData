package ulpgc.shared.runners;

import ulpgc.shared.Utils;
import java.util.Map;

public interface Runner {
    String mode();
    RunOut run(Utils.Case c, Utils.Config cfg) throws Exception;
    record RunOut(double elapsedSec, double checksum, Map<String, Object> extra) {}
}
