#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "benchmark.h"

int main(void) {
    srand((unsigned) time(NULL));
    const char *output_path = OUTPUT_PATH;
    FILE *f = fopen(output_path, "w");
    if (!f) {
        perror("Could not create results file");
        return 1;
    }

    fprintf(f, "n,input_density,nnz_density,time_basic,time_blocked,time_sparse\n");
    printf("n,input_density,nnz_density,time_basic,time_blocked,time_sparse\n");

    int sizes[] = {64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double densities[] = {1.0, 0.1, 0.01, 0.001};
    int num_densities = sizeof(densities) / sizeof(densities[0]);
    int block_size = 32;

    for (int d = 0; d < num_densities; ++d) {
        for (int i = 0; i < num_sizes; ++i) {
            benchmark_case_to_file(sizes[i], densities[d], block_size, f);
        }
    }
    fclose(f);

    printf("\nResults saved to %s\n", output_path);
    return 0;
}
