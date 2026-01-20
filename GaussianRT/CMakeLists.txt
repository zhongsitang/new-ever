cmake_minimum_required(VERSION 3.20)
project(GaussianRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Slang configuration
set(SLANG_ROOT "" CACHE PATH "Path to Slang installation")
if(NOT SLANG_ROOT)
    message(FATAL_ERROR "SLANG_ROOT must be set to the Slang installation directory")
endif()

set(SLANG_INCLUDE_DIR "${SLANG_ROOT}/include")
set(SLANG_LIB_DIR "${SLANG_ROOT}/lib")
set(SLANGC "${SLANG_ROOT}/bin/slangc")

# PyTorch configuration
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SLANG_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link directories
link_directories(${SLANG_LIB_DIR})

# Slang libraries
set(SLANG_LIBS slang slang-rhi)

# Core library sources
set(CORE_SOURCES
    src/device.cpp
    src/acceleration_structure.cpp
    src/renderer.cpp
    src/volume_renderer.cpp
)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/aabb_builder.cu
    src/cuda/initialize_state.cu
    src/cuda/backward_kernel.cu
)

# Build core library
add_library(gaussian_rt_core SHARED
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(gaussian_rt_core
    ${SLANG_LIBS}
    CUDA::cudart
    ${TORCH_LIBRARIES}
)

# Python module
pybind11_add_module(_gaussian_rt
    python/bindings.cpp
    python/torch_extension.cpp
)

target_link_libraries(_gaussian_rt PRIVATE
    gaussian_rt_core
    ${TORCH_LIBRARIES}
)

# Install
install(TARGETS _gaussian_rt gaussian_rt_core
    LIBRARY DESTINATION lib
)

install(DIRECTORY python/
    DESTINATION python/gaussian_rt
    FILES_MATCHING PATTERN "*.py"
)
