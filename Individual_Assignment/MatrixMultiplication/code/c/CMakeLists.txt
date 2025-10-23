cmake_minimum_required(VERSION 3.20)
project(matrix_c_clion C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O3 -march=native -mtune=native)
endif()

add_library(matrix STATIC src/matrix.c
        include/matrix.h)
target_include_directories(matrix PUBLIC include)

add_executable(bench app/bench.c
        include/matrix.h)
target_link_libraries(bench PRIVATE matrix m)

# Tests
include(CTest)
if (BUILD_TESTING)
    add_executable(test_matrix tests/test_matrix.c
            include/matrix.h)
    target_link_libraries(test_matrix PRIVATE matrix m)
    add_test(NAME matrix_small_case COMMAND test_matrix)
endif()
