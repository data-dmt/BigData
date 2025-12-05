#include "matmul.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

double *alloc_matrix(int n) {
    double *M = (double *) malloc((size_t) n * (size_t) n * sizeof(double));
    return M;
}

void free_matrix(double *M) {
    free(M);
}

void random_matrix(double *M, int n) {
    static int seeded = 0;
    if (!seeded) {
        srand(12345);
        seeded = 1;
    }
    int total = n * n;
    for (int i = 0; i < total; ++i) {
        M[i] = (double) rand() / (double) RAND_MAX;
    }
}

// Basic secuential multiplication : C = A * B
void matmul_basic(const double *A, const double *B, double *C, int n) {
    memset(C, 0, (size_t) n * (size_t) n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

// Parallel version
typedef struct {
    const double *A;
    const double *B;
    double *C;
    int n;
    int row_begin;
    int row_end;
} WorkerArgs;

static void *worker_basic(void *arg) {
    WorkerArgs *w = (WorkerArgs *) arg;
    const double *A = w->A;
    const double *B = w->B;
    double *C = w->C;
    int n = w->n;
    int row_begin = w->row_begin;
    int row_end = w->row_end;
    for (int i = row_begin; i < row_end; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
    return NULL;
}

void matmul_parallel(const double *A, const double *B, double *C,
                     int n, int num_threads) {
    if (num_threads <= 1) {
        matmul_basic(A, B, C, n);
        return;
    }
    memset(C, 0, (size_t) n * (size_t) n * sizeof(double));
    pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    WorkerArgs *args = (WorkerArgs *) malloc(num_threads * sizeof(WorkerArgs));

    int rows_per_thread = n / num_threads;
    int extra = n % num_threads;
    int current_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int start = current_row;
        int count = rows_per_thread + (t < extra ? 1 : 0);
        int end = start + count;
        current_row = end;
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].n = n;
        args[t].row_begin = start;
        args[t].row_end = end;
        pthread_create(&threads[t], NULL, worker_basic, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(args);
}

// SIMD version with AVX
void matmul_simd(const double *A, const double *B, double *C, int n) {
    memset(C, 0, (size_t) n * (size_t) n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            __m256d a_vec = _mm256_set1_pd(aik);
            int j = 0;
            for (; j + 3 < n; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(&B[k * n + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * n + j]);
                __m256d prod = _mm256_mul_pd(a_vec, b_vec);
                c_vec = _mm256_add_pd(c_vec, prod);
                _mm256_storeu_pd(&C[i * n + j], c_vec);
            }
            for (; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

// Parallel version + SIMD
typedef struct {
    const double *A;
    const double *B;
    double *C;
    int n;
    int row_begin;
    int row_end;
} WorkerArgsSimd;

static void *worker_simd(void *arg) {
    WorkerArgsSimd *w = (WorkerArgsSimd *) arg;
    const double *A = w->A;
    const double *B = w->B;
    double *C = w->C;
    int n = w->n;
    int row_begin = w->row_begin;
    int row_end = w->row_end;
    for (int i = row_begin; i < row_end; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[i * n + k];
            __m256d a_vec = _mm256_set1_pd(aik);
            int j = 0;
            for (; j + 3 < n; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(&B[k * n + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * n + j]);
                __m256d prod = _mm256_mul_pd(a_vec, b_vec);
                c_vec = _mm256_add_pd(c_vec, prod);
                _mm256_storeu_pd(&C[i * n + j], c_vec);
            }
            for (; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
    return NULL;
}

void matmul_parallel_simd(const double *A, const double *B, double *C,
                          int n, int num_threads) {
    if (num_threads <= 1) {
        matmul_simd(A, B, C, n);
        return;
    }

    memset(C, 0, (size_t) n * (size_t) n * sizeof(double));
    pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    WorkerArgsSimd *args = (WorkerArgsSimd *) malloc(num_threads * sizeof(WorkerArgsSimd));
    int rows_per_thread = n / num_threads;
    int extra = n % num_threads;
    int current_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int start = current_row;
        int count = rows_per_thread + (t < extra ? 1 : 0);
        int end = start + count;
        current_row = end;
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].n = n;
        args[t].row_begin = start;
        args[t].row_end = end;
        pthread_create(&threads[t], NULL, worker_simd, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(args);
}
