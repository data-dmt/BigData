cmake_minimum_required(VERSION 3.20)
project(matrix_2 C)

set(CMAKE_C_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR}/include)
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output")

add_executable(matrix_2
        src/main.c
        src/matrix_dense.c
        src/matrix_sparse.c
        src/benchmark.c
)

target_compile_definitions(matrix_2 PRIVATE
        OUTPUT_PATH=\"${OUTPUT_DIR}/results.csv\"
)
