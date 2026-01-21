# =============================================================================
# PrimTracer - Primitive-based Volume Rendering Library
# =============================================================================
# Modern CMake (3.28+) configuration.
#
# Build:
#   cmake --preset release
#   cmake --build --preset release
#
# Shader pipeline:
#   .slang -> .ptx (embedded as const char*)
# =============================================================================

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(primtracer
  VERSION 1.0.0
  DESCRIPTION "Primitive-based volume rendering with OptiX"
  LANGUAGES C CXX CUDA
)

# -----------------------------------------------------------------------------
# Global build settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/primtracer")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/primtracer")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Dependencies and local CMake modules
# -----------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(FindOptiX)
include(FindTorch)
include(SlangHelper)

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
set(TRACER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/primtracer/src")
set(SLANG_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/primtracer/slang")

# -----------------------------------------------------------------------------
# Main pybind11 extension module
# -----------------------------------------------------------------------------
pybind11_add_module(primtracer_core MODULE
  ${TRACER_DIR}/ray_tracer.cpp
  ${TRACER_DIR}/ray_tracer.h
  ${TRACER_DIR}/accel_structure.h
  ${TRACER_DIR}/py_binding.cpp
  ${TRACER_DIR}/structs.h
  ${TRACER_DIR}/cuda_buffer.h
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/create_aabbs.h
  ${TRACER_DIR}/initialize_density.cu
  ${TRACER_DIR}/initialize_density.h
)

set_source_files_properties(
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/initialize_density.cu
  PROPERTIES LANGUAGE CUDA
)

# -----------------------------------------------------------------------------
# Shader embedding (Slang -> PTX -> C++ string)
# -----------------------------------------------------------------------------
slang_ptx_embed(
  TARGET       primtracer_core
  NAME         shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/shaders.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS} ${OptiX_ROOT}/include
)

# -----------------------------------------------------------------------------
# Slang PyTorch modules
# -----------------------------------------------------------------------------
slang_add_py_module(backwards_kernel
  SLANG_FILES  ${SLANG_DIR}/backwards_kernel.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(backwards_kernel PRIVATE Torch::Torch)

slang_add_py_module(sh
  SLANG_FILES  ${SLANG_DIR}/sh.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(sh PRIVATE Torch::Torch)

# -----------------------------------------------------------------------------
# Linking and compile options
# -----------------------------------------------------------------------------
target_link_libraries(primtracer_core PRIVATE
  OptiX::OptiX
  Torch::Torch
)

target_compile_options(primtracer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177,550>
)

if(MSVC)
  set_target_properties(primtracer_core PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    CUDA_RUNTIME_LIBRARY Shared
  )
else()
  target_compile_options(primtracer_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  )
endif()

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "PrimTracer Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Project:         ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs:      ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OptiX:           ${OptiX_VERSION} (${OptiX_ROOT})")
message(STATUS "========================================")
