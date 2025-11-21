#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H

double *alloc_dense(int n);
void free_dense(double *m);
void random_dense_matrix(double *A, int n, double density);
void zero_dense(double *A, int n);
int equal_dense(const double *A, const double *B, int n, double tol);
void transpose(const double *B, double *BT, int n);
void matmul_basic(const double *A, const double *B, double *C, int n);
void matmul_blocked(const double *A, const double *B, double *C, int n, int block);

#endif
