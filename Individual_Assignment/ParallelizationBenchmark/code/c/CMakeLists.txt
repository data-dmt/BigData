cmake_minimum_required(VERSION 3.24)
project(MatrixMulBenchmarkC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -pthread")

add_executable(MatrixMulBenchmarkC
        src/main.c
        src/matmul.c
        src/matmul.h
)
