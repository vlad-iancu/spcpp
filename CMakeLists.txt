cmake_minimum_required(VERSION 3.10)
find_package(OpenMP REQUIRED)

add_library(spcpp gmres_householder_dense.cpp gmres_mgs_sparse.cpp)

target_include_directories(spcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spcpp PUBLIC openblas aoclsparse)

add_executable(gmres_test gmres_test.cpp)
add_executable(gmres_sparse_test gmres_sparse_test.cpp)
target_link_libraries(gmres_test PUBLIC spcpp)
target_link_libraries(gmres_sparse_test PUBLIC spcpp)
target_link_libraries(spcpp PUBLIC OpenMP::OpenMP_CXX rt)
target_link_libraries(gmres_sparse_test PUBLIC OpenMP::OpenMP_CXX)

unset(USE_MY_LIB CACHE) # <---- this is the important!!

