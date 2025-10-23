#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void){
#if defined(_POSIX_TIMERS)
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9*ts.tv_nsec;
#else
    return (double)clock()/CLOCKS_PER_SEC;
#endif
}

static Order parse_order(const char* s){
    if(!s) return ORDER_IKJ;
    if(strcmp(s,"ijk")==0) return ORDER_IJK;
    if(strcmp(s,"ikj")==0) return ORDER_IKJ;
    if(strcmp(s,"jik")==0) return ORDER_JIK;
    return ORDER_IKJ;
}

int main(int argc, char** argv){
    size_t n = 512;
    int runs = 5;
    unsigned int seed = 42;
    const char* order_s = "ikj";

    for (int i=1; i<argc; ++i){
        if (!strcmp(argv[i],"-n") && i+1<argc) n = (size_t) atoll(argv[++i]);
        else if (!strcmp(argv[i],"-r") && i+1<argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-s") && i+1<argc) seed = (unsigned) atoi(argv[++i]);
        else if (!strcmp(argv[i],"-o") && i+1<argc) order_s = argv[++i];
    }
    Order order = parse_order(order_s);

    Matrix A = mat_alloc(n), B = mat_alloc(n), C = mat_alloc(n);
    mat_fill_random(A, seed);
    mat_fill_random(B, seed+1);

    double *times = malloc((size_t)runs * sizeof(double));

    for (int r=0; r<runs; ++r){
        mat_zero(C);
        double t0 = now_sec();
        mat_mul(A, B, C, order);
        double t1 = now_sec();
        times[r] = t1 - t0;
        fprintf(stderr, "run %d/%d: %.6f s\n", r+1, runs, times[r]);
        if (C.data[0] == -1.0) puts("!");
    }

    double sum=0.0; for (int i=0;i<runs;++i) sum += times[i];
    double mean = sum / runs;
    double var=0.0; for (int i=0;i<runs;++i){ double d=times[i]-mean; var += d*d; }
    double stdev = runs>1 ? sqrt(var/(runs-1)) : 0.0;

    printf("{\"lang\":\"C\",\"n\":%zu,\"runs\":%d,\"order\":\"%s\","
           "\"mean_sec\":%.6f,\"stdev_sec\":%.6f}\n",
           n, runs, order_s, mean, stdev);

    free(times);
    mat_free(&A); mat_free(&B); mat_free(&C);
    return 0;
}
