#include "matrix_dense.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

double *alloc_dense(int n) {
    double *m = (double *) malloc((size_t) n * n * sizeof(double));
    if (!m) {
        fprintf(stderr, "Error: could not save memory for matrix %d x %d\n", n, n);
        exit(EXIT_FAILURE);
    }
    return m;
}

void free_dense(double *m) {
    free(m);
}

void random_dense_matrix(double *A, int n, double density) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double r = (double) rand() / (double) RAND_MAX;
            if (r < density) {
                A[i * n + j] = (double) rand() / (double) RAND_MAX;
            } else {
                A[i * n + j] = 0.0;
            }
        }
    }
}

void zero_dense(double *A, int n) {
    memset(A, 0, (size_t) n * n * sizeof(double));
}

int equal_dense(const double *A, const double *B, int n, double tol) {
    for (int i = 0; i < n * n; ++i) {
        if (fabs(A[i] - B[i]) > tol) {
            return 0;
        }
    }
    return 1;
}

void transpose(const double *B, double *BT, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            BT[j * n + i] = B[i * n + j];
        }
    }
}

void matmul_basic(const double *A, const double *B, double *C, int n) {
    zero_dense(C, n);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

void matmul_blocked(const double *A, const double *B, double *C, int n, int block) {
    double *BT = alloc_dense(n);
    transpose(B, BT, n);
    zero_dense(C, n);

    for (int ii = 0; ii < n; ii += block) {
        int i_max = (ii + block < n) ? (ii + block) : n;
        for (int jj = 0; jj < n; jj += block) {
            int j_max = (jj + block < n) ? (jj + block) : n;
            for (int kk = 0; kk < n; kk += block) {
                int k_max = (kk + block < n) ? (kk + block) : n;
                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double sum = C[i * n + j];
                        const double *Arow = &A[i * n];
                        const double *BTrow = &BT[j * n];
                        for (int k = kk; k < k_max; ++k) {
                            sum += Arow[k] * BTrow[k];
                        }
                        C[i * n + j] = sum;
                    }
                }
            }
        }
    }

    free_dense(BT);
}
