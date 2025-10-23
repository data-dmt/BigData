#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void){
    size_t n=2;
    Matrix A = mat_alloc(n), B = mat_alloc(n), C = mat_alloc(n);
    mat_set(A,0,0,1); mat_set(A,0,1,2);
    mat_set(A,1,0,3); mat_set(A,1,1,4);
    mat_set(B,0,0,5); mat_set(B,0,1,6);
    mat_set(B,1,0,7); mat_set(B,1,1,8);

    mat_mul(A,B,C,ORDER_IKJ);

    assert(fabs(mat_get(C,0,0) - 19.0) < 1e-9);
    assert(fabs(mat_get(C,0,1) - 22.0) < 1e-9);
    assert(fabs(mat_get(C,1,0) - 43.0) < 1e-9);
    assert(fabs(mat_get(C,1,1) - 50.0) < 1e-9);

    mat_free(&A); mat_free(&B); mat_free(&C);
    printf("OK\n");
    return 0;
}
