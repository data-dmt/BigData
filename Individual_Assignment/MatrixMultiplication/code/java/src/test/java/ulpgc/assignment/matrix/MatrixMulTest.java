package ulpgc.assignment.matrix;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixMulTest {
    @Test
    void smallCase(){
        double[][] A={{1,2},{3,4}};
        double[][] B={{5,6},{7,8}};
        double[][] C=MatrixMul.mul(A,B, MatrixMul.Order.IKJ);
        assertEquals(19.0, C[0][0], 1e-9);
        assertEquals(22.0, C[0][1], 1e-9);
        assertEquals(43.0, C[1][0], 1e-9);
        assertEquals(50.0, C[1][1], 1e-9);
    }
}
