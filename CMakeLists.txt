cmake_minimum_required(VERSION 3.13)
project(numcpp)
find_package(OpenMP REQUIRED)

add_library(spcpp gmres_householder_dense.cpp gmres_mgs_sparse.cpp cg_dense.cpp cg_sparse.cpp)

target_include_directories(spcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spcpp PUBLIC openblas aoclsparse OpenMP::OpenMP_CXX rt)

add_executable(gmres_test gmres_test.cpp)
add_executable(cg_dense_test cg_dense_test.cpp)
add_executable(cg_sparse_test cg_sparse_test.cpp)
add_executable(gmres_sparse_test gmres_sparse_test.cpp)
add_executable(eigentest eigen_test.cpp)
add_executable(listestgmres lis_test_gmres.cpp)
add_executable(listestcg lis_test_cg.cpp)
add_executable(listestcgsparse lis_test_cg_sparse.cpp)

target_link_libraries(gmres_test PUBLIC spcpp)
target_link_libraries(cg_dense_test PUBLIC spcpp)
target_link_libraries(cg_sparse_test PUBLIC spcpp)
target_link_libraries(gmres_sparse_test PUBLIC spcpp)
#target_link_libraries(spcpp PUBLIC OpenMP::OpenMP_CXX rt)
target_link_libraries(gmres_sparse_test PUBLIC OpenMP::OpenMP_CXX)

# If you want multithread use OpenMP::OpenMP_CXX library
target_link_libraries(listestgmres PUBLIC lis)
target_link_directories(listestgmres PUBLIC "/usr/local/lib/")
target_link_libraries(listestcg PUBLIC lis openblas OpenMP::OpenMP_CXX rt)
target_link_directories(listestcg PUBLIC "/usr/local/lib/")
target_link_directories(listestcgsparse PUBLIC "/usr/local/lib/")
target_link_libraries(listestcgsparse PUBLIC lis openblas OpenMP::OpenMP_CXX rt)


target_include_directories(eigentest PUBLIC "${CMAKE_SOURCE_DIR}/eigen/eigen-3.4.0/unsupported/Eigen/")
target_include_directories(listestgmres PUBLIC "/usr/local/include/")
target_include_directories(listestcg PUBLIC "/usr/local/include/")
target_include_directories(listestcgsparse PUBLIC "/usr/local/include/")

unset(USE_MY_LIB CACHE) # <---- this is the important!!

