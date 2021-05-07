cmake_minimum_required(VERSION 3.6)

project(solver)

add_executable(solver solver.cpp)

target_link_libraries(solver PRIVATE vtu11)

# target_compile_features(solver PUBLIC cxx_std_17)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solver PUBLIC OpenMP::OpenMP_CXX)
endif()