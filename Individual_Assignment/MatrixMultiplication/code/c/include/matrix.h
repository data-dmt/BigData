#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    size_t n;
    double *data;
} Matrix;

typedef enum { ORDER_IJK, ORDER_IKJ, ORDER_JIK } Order;

Matrix mat_alloc(size_t n);
void   mat_free(Matrix *m);
void   mat_zero(Matrix m);
void   mat_fill_random(Matrix m, unsigned int seed);

double mat_get(const Matrix m, size_t i, size_t j);
void   mat_set(Matrix m, size_t i, size_t j, double v);

void mat_mul(Matrix A, Matrix B, Matrix C, Order order);

#endif
