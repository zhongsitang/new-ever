cmake_minimum_required(VERSION 3.18)
project(GaussianRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(GAUSSIANRT_BUILD_PYTHON "Build Python bindings" ON)
option(GAUSSIANRT_BUILD_TESTS "Build tests" ON)

# Find dependencies
find_package(CUDAToolkit REQUIRED)

# Slang SDK path
set(SLANG_SDK_PATH "" CACHE PATH "Path to Slang SDK")
if(NOT SLANG_SDK_PATH)
    # Try to find slang in common locations
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../slang-master")
        set(SLANG_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../slang-master")
    elseif(DEFINED ENV{SLANG_PATH})
        set(SLANG_SDK_PATH "$ENV{SLANG_PATH}")
    endif()
endif()

if(NOT EXISTS "${SLANG_SDK_PATH}")
    message(FATAL_ERROR "SLANG_SDK_PATH not found. Please set -DSLANG_SDK_PATH=<path>")
endif()

message(STATUS "Using Slang SDK at: ${SLANG_SDK_PATH}")

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompileSlangShaders)

# Find Slang libraries
find_library(SLANG_LIB slang HINTS "${SLANG_SDK_PATH}/build/Release/lib" "${SLANG_SDK_PATH}/lib")
find_library(SLANG_RHI_LIB slang-rhi HINTS "${SLANG_SDK_PATH}/build/Release/lib" "${SLANG_SDK_PATH}/lib")

if(NOT SLANG_LIB OR NOT SLANG_RHI_LIB)
    message(STATUS "Slang libraries not found in prebuilt locations, will build from source if available")
endif()

# Slang include directories
set(SLANG_INCLUDE_DIRS
    "${SLANG_SDK_PATH}/include"
    "${SLANG_SDK_PATH}/source"
    "${SLANG_SDK_PATH}/external/slang-rhi/include"
    "${SLANG_SDK_PATH}/external/slang-rhi/src"
)

# Compile backward shaders to CUDA
set(BACKWARD_SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/backward/backward_kernel.slang
)

set(SHADER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)

# Generate embedded shader headers for forward pass
compile_slang_to_header(
    NAME forward_shaders
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/forward/raytracing.slang
    INCLUDE_DIRS ${SHADER_INCLUDE_DIRS}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Compile backward kernel to CUDA
compile_slang_to_cuda(
    NAME backward_kernels
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/backward/backward_kernel.slang
    INCLUDE_DIRS ${SHADER_INCLUDE_DIRS}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Core library
add_library(gaussian_rt_core STATIC
    src/Device.cpp
    src/AccelStruct.cpp
    src/GaussianPrimitives.cpp
    src/ForwardRenderer.cpp
    src/BackwardPass.cpp
    src/GaussianRT.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/generated/backward_kernels.cu
)

target_include_directories(gaussian_rt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${SLANG_INCLUDE_DIRS}
)

target_link_libraries(gaussian_rt_core PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

if(SLANG_LIB AND SLANG_RHI_LIB)
    target_link_libraries(gaussian_rt_core PUBLIC
        ${SLANG_LIB}
        ${SLANG_RHI_LIB}
    )
endif()

target_compile_definitions(gaussian_rt_core PRIVATE
    GAUSSIANRT_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# CUDA specific flags
set_target_properties(gaussian_rt_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(gaussian_rt_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -Xcompiler -fPIC
    >
)

# Python bindings
if(GAUSSIANRT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR)
            find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
        endif()
    endif()

    if(pybind11_FOUND)
        # Find PyTorch
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(TORCH_CMAKE_PATH)
            list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
            find_package(Torch REQUIRED)
        endif()

        pybind11_add_module(gaussian_rt_ext MODULE
            src/python/bindings.cpp
        )

        target_link_libraries(gaussian_rt_ext PRIVATE
            gaussian_rt_core
        )

        if(Torch_FOUND)
            target_link_libraries(gaussian_rt_ext PRIVATE ${TORCH_LIBRARIES})
            target_compile_definitions(gaussian_rt_ext PRIVATE GAUSSIANRT_HAS_TORCH)
        endif()

        # Install Python module
        install(TARGETS gaussian_rt_ext
            LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python/gaussian_rt
        )
    else()
        message(WARNING "pybind11 not found, Python bindings will not be built")
    endif()
endif()

# Tests
if(GAUSSIANRT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(TARGETS gaussian_rt_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/gaussian_rt
    DESTINATION include
)
