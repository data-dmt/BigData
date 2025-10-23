#include "matrix.h"
#include <stdlib.h>
#include <string.h>

static inline size_t idx(size_t n, size_t i, size_t j) { return i*n + j; }

Matrix mat_alloc(size_t n) {
    Matrix m = { n, NULL };
#if defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
    m.data = aligned_alloc(64, n*n*sizeof(double));
#else
    m.data = malloc(n*n*sizeof(double));
#endif
    return m;
}

void mat_free(Matrix *m) {
    if (!m || !m->data) return;
    free(m->data);
    m->data = NULL;
    m->n = 0;
}

void mat_zero(Matrix m) {
    memset(m.data, 0, m.n*m.n*sizeof(double));
}

void mat_fill_random(Matrix m, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < m.n*m.n; ++i)
        m.data[i] = (double)rand()/RAND_MAX;
}

double mat_get(const Matrix m, size_t i, size_t j) { return m.data[idx(m.n,i,j)]; }
void   mat_set(Matrix m, size_t i, size_t j, double v) { m.data[idx(m.n,i,j)] = v; }

void mat_mul(Matrix A, Matrix B, Matrix C, Order order) {
    size_t n = A.n;
    mat_zero(C);

    switch (order) {
        case ORDER_IJK:
            for (size_t i=0;i<n;++i)
                for (size_t j=0;j<n;++j) {
                    double sum = 0.0;
                    for (size_t k=0;k<n;++k)
                        sum += A.data[idx(n,i,k)] * B.data[idx(n,k,j)];
                    C.data[idx(n,i,j)] = sum;
                }
            break;

        case ORDER_IKJ:
            for (size_t i=0;i<n;++i)
                for (size_t k=0;k<n;++k) {
                    double aik = A.data[idx(n,i,k)];
                    for (size_t j=0;j<n;++j)
                        C.data[idx(n,i,j)] += aik * B.data[idx(n,k,j)];
                }
            break;

        case ORDER_JIK:
            for (size_t j=0;j<n;++j)
                for (size_t i=0;i<n;++i) {
                    double sum = 0.0;
                    for (size_t k=0;k<n;++k)
                        sum += A.data[idx(n,i,k)] * B.data[idx(n,k,j)];
                    C.data[idx(n,i,j)] = sum;
                }
            break;
    }
}
