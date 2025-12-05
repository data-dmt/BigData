#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "matmul.h"

double elapsed_ms(struct timeval start, struct timeval end) {
    long sec_diff  = end.tv_sec  - start.tv_sec;
    long usec_diff = end.tv_usec - start.tv_usec;
    return (double)sec_diff * 1000.0 + (double)usec_diff / 1000.0;
}

int main(void) {
    int sizes[] = {256, 512, 1024};
    int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    long hw_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (hw_threads <= 0) hw_threads = 4;
    int physical_cores = (int)(hw_threads / 2);
    if (physical_cores < 1) physical_cores = 1;
    int thread_options[3];
    int num_thread_options = 0;
    thread_options[num_thread_options++] = 1;

    if (physical_cores != 1 && physical_cores != (int)hw_threads) {
        thread_options[num_thread_options++] = physical_cores;
    }
    if ((int)hw_threads != 1) {
        thread_options[num_thread_options++] = (int)hw_threads;
    }

    printf("Detected hardware threads: %ld\n", hw_threads);
    printf("Using thread configurations: ");
    for (int i = 0; i < num_thread_options; ++i) {
        printf("%d ", thread_options[i]);
    }
    printf("\n\n");

    FILE *f = fopen("results.csv", "w");
    if (!f) {
        perror("Error opening results.csv");
        return 1;
    }

    fprintf(f, "n,threads,"
               "time_basic_ms,time_parallel_ms,time_simd_ms,time_par_simd_ms,"
               "speedup_parallel,speedup_simd,speedup_par_simd,"
               "eff_parallel,eff_par_simd\n");

    for (int s = 0; s < num_sizes; ++s) {
        int n = sizes[s];
        printf("===== Matrix size n = %d =====\n", n);
        double *A = alloc_matrix(n);
        double *B = alloc_matrix(n);
        double *C = alloc_matrix(n);

        if (!A || !B || !C) {
            fprintf(stderr, "Error allocating matrices for n = %d\n", n);
            free_matrix(A);
            free_matrix(B);
            free_matrix(C);
            fclose(f);
            return 1;
        }
        random_matrix(A, n);
        random_matrix(B, n);
        struct timeval t1, t2;

        // Basic secuential
        gettimeofday(&t1, NULL);
        matmul_basic(A, B, C, n);
        gettimeofday(&t2, NULL);
        double time_basic = elapsed_ms(t1, t2);
        printf("Basic: %.2f ms\n", time_basic);

        // SIMD (vectorized secuential)
        gettimeofday(&t1, NULL);
        matmul_simd(A, B, C, n);
        gettimeofday(&t2, NULL);
        double time_simd = elapsed_ms(t1, t2);
        printf("SIMD: %.2f ms (speedup vs basic: %.2f)\n",
               time_simd, time_basic / time_simd);

        // For each thread, parallel and parallel+SIMD
        for (int t = 0; t < num_thread_options; ++t) {
            int threads = thread_options[t];
            printf(" -> threads = %d\n", threads);

            // Paralell
            gettimeofday(&t1, NULL);
            matmul_parallel(A, B, C, n, threads);
            gettimeofday(&t2, NULL);
            double time_parallel = elapsed_ms(t1, t2);

            // Paralell + SIMD
            gettimeofday(&t1, NULL);
            matmul_parallel_simd(A, B, C, n, threads);
            gettimeofday(&t2, NULL);
            double time_par_simd = elapsed_ms(t1, t2);
            double speedup_parallel = time_basic / time_parallel;
            double speedup_simd = time_basic / time_simd;
            double speedup_par_simd = time_basic / time_par_simd;
            double eff_parallel = speedup_parallel / (double)threads;
            double eff_par_simd = speedup_par_simd / (double)threads;
            printf("Parallel: %.2f ms (speedup: %.2f, eff: %.3f)\n",
                   time_parallel, speedup_parallel, eff_parallel);
            printf("Parallel+SIMD: %.2f ms (speedup: %.2f, eff: %.3f)\n",
                   time_par_simd, speedup_par_simd, eff_par_simd);
            fprintf(f, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    n, threads,
                    time_basic, time_parallel, time_simd, time_par_simd,
                    speedup_parallel, speedup_simd, speedup_par_simd,
                    eff_parallel, eff_par_simd);
        }
        printf("\n");
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
    }
    fclose(f);
    printf("Resultas saved in 'results.csv'\n");

    return 0;
}
