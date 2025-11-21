#include "matrix_sparse.h"
#include "matrix_dense.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CSRMatrix dense_to_csr(const double *A, int n, double tol) {
    CSRMatrix S;
    S.n = n;
    S.nnz = 0;

    for (int i = 0; i < n * n; ++i) {
        if (fabs(A[i]) > tol) {
            S.nnz++;
        }
    }

    S.val = (double *) malloc((size_t) S.nnz * sizeof(double));
    S.col_idx = (int *) malloc((size_t) S.nnz * sizeof(int));
    S.row_ptr = (int *) malloc((size_t) (n + 1) * sizeof(int));

    if (!S.val || !S.col_idx || !S.row_ptr) {
        fprintf(stderr, "Error: could not save memory for CSR\n");
        exit(EXIT_FAILURE);
    }

    int nnz_counter = 0;
    for (int i = 0; i < n; ++i) {
        S.row_ptr[i] = nnz_counter;
        for (int j = 0; j < n; ++j) {
            double v = A[i * n + j];
            if (fabs(v) > tol) {
                S.val[nnz_counter] = v;
                S.col_idx[nnz_counter] = j;
                nnz_counter++;
            }
        }
    }
    S.row_ptr[n] = nnz_counter;

    return S;
}

void free_csr(CSRMatrix *A) {
    if (!A) return;
    free(A->val);
    free(A->col_idx);
    free(A->row_ptr);
    A->val = NULL;
    A->col_idx = NULL;
    A->row_ptr = NULL;
}

void csr_matvec(const CSRMatrix *A, const double *x, double *y) {
    int n = A->n;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        for (int k = row_start; k < row_end; ++k) {
            sum += A->val[k] * x[A->col_idx[k]];
        }
        y[i] = sum;
    }
}

void csr_matmul_dense(const CSRMatrix *A, const double *B, double *C) {
    int n = A->n;
    zero_dense(C, n);

    for (int i = 0; i < n; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];
        for (int k = row_start; k < row_end; ++k) {
            int colA = A->col_idx[k];
            double a = A->val[k];
            const double *B_row = &B[colA * n];
            double *C_row = &C[i * n];
            for (int j = 0; j < n; ++j) {
                C_row[j] += a * B_row[j];
            }
        }
    }
}
