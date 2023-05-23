cmake_minimum_required(VERSION 3.10)

add_library(spcpp gmres_householder_dense.cpp)

target_include_directories(spcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spcpp PUBLIC openblas)

add_executable(gmres_test gmres_test.cpp)
target_link_libraries(gmres_test PUBLIC spcpp)
unset(USE_MY_LIB CACHE) # <---- this is the important!!

