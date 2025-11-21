#ifndef MATRIX_SPARSE_H
#define MATRIX_SPARSE_H

typedef struct {
    int n;
    int nnz;
    double *val;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

CSRMatrix dense_to_csr(const double *A, int n, double tol);
void free_csr(CSRMatrix *A);
void csr_matvec(const CSRMatrix *A, const double *x, double *y);
void csr_matmul_dense(const CSRMatrix *A, const double *B, double *C);

#endif
