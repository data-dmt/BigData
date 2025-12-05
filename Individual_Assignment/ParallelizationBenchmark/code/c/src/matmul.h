#ifndef MATRIX_MUL_BENCHMARK_C_MATMUL_H
#define MATRIX_MUL_BENCHMARK_C_MATMUL_H

double *alloc_matrix(int n);

void free_matrix(double *M);
void random_matrix(double *M, int n);
void matmul_basic(const double *A, const double *B, double *C, int n);
void matmul_parallel(const double *A, const double *B, double *C,
                     int n, int num_threads);

void matmul_simd(const double *A, const double *B, double *C, int n);
void matmul_parallel_simd(const double *A, const double *B, double *C,
                          int n, int num_threads);

#endif
