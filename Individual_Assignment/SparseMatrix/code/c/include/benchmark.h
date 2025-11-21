#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdio.h>

void benchmark_case(int n, double density, int block_size);
void benchmark_case_to_file(int n, double density, int block_size, FILE *f);

#endif
