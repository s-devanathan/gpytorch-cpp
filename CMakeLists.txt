cmake_minimum_required(VERSION 3.15)
project(gaussian_process_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
find_package(Torch REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(GP_SOURCES
    src/kernels.cpp
    src/means.cpp
    src/gaussian_process.cpp
)

# Library
add_library(gp_lib ${GP_SOURCES})
target_link_libraries(gp_lib ${TORCH_LIBRARIES})

# Example executable
add_executable(gp_example examples/simple_gp.cpp)
target_link_libraries(gp_example gp_lib ${TORCH_LIBRARIES})

# Compile flags
if(MSVC)
  target_compile_options(gp_lib PRIVATE /W4)
  target_compile_options(gp_example PRIVATE /W4)
else()
  target_compile_options(gp_lib PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(gp_example PRIVATE -Wall -Wextra -Wpedantic)
endif()