#include "benchmark.h"
#include "matrix_dense.h"
#include "matrix_sparse.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int n;
    double input_density;
    double nnz_density;
    double time_basic;
    double time_blocked;
    double time_sparse;
} BenchmarkResult;

static double now_seconds() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

static void run_benchmark_internal(int n, double density, int block_size,
                                   BenchmarkResult *result) {
    double *A = alloc_dense(n);
    double *B = alloc_dense(n);
    double *C1 = alloc_dense(n);
    double *C2 = alloc_dense(n);
    double *C3 = alloc_dense(n);

    random_dense_matrix(A, n, density);
    random_dense_matrix(B, n, density);

    double t0, t1;

    t0 = now_seconds();
    matmul_basic(A, B, C1, n);
    t1 = now_seconds();
    double time_basic = t1 - t0;

    t0 = now_seconds();
    matmul_blocked(A, B, C2, n, block_size);
    t1 = now_seconds();
    double time_blocked = t1 - t0;

    if (n <= 256 && !equal_dense(C1, C2, n, 1e-8)) {
        fprintf(stderr,
                "Warning: C_basic and C_blocked differ for n=%d.\n",
                n);
    }

    CSRMatrix A_csr = dense_to_csr(A, n, 1e-12);

    t0 = now_seconds();
    csr_matmul_dense(&A_csr, B, C3);
    t1 = now_seconds();
    double time_sparse = t1 - t0;

    if (n <= 256 && !equal_dense(C1, C3, n, 1e-8)) {
        fprintf(stderr,
                "Warning: C_basic and C_sparse differ for n=%d.\n",
                n);
    }

    double nnz_density = (double) A_csr.nnz / (double) (n * n);

    result->n = n;
    result->input_density = density;
    result->nnz_density = nnz_density;
    result->time_basic = time_basic;
    result->time_blocked = time_blocked;
    result->time_sparse = time_sparse;

    free_csr(&A_csr);
    free_dense(A);
    free_dense(B);
    free_dense(C1);
    free_dense(C2);
    free_dense(C3);
}

void benchmark_case(int n, double density, int block_size) {
    BenchmarkResult r;
    run_benchmark_internal(n, density, block_size, &r);
    printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           r.n,
           r.input_density,
           r.nnz_density,
           r.time_basic,
           r.time_blocked,
           r.time_sparse);
}

void benchmark_case_to_file(int n, double density, int block_size, FILE *f) {
    BenchmarkResult r;
    run_benchmark_internal(n, density, block_size, &r);
    if (f != NULL) {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.n,
                r.input_density,
                r.nnz_density,
                r.time_basic,
                r.time_blocked,
                r.time_sparse);
    }
    printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           r.n,
           r.input_density,
           r.nnz_density,
           r.time_basic,
           r.time_blocked,
           r.time_sparse);
}
