# =============================================================================
# Ever - Spline-based Volume Rendering Library
# =============================================================================
# Modern CMake (3.28+) top-level configuration.
#
# Build (presets recommended):
#   cmake --preset release
#   cmake --build --preset release
#
# Target:
#   - Pybind11 extension module: ellipsoid_splinetracer
#
# Shader pipeline:
#   .slang -> runtime compilation via slang-rhi
# =============================================================================

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(splinetracers
  VERSION 1.0.0
  DESCRIPTION "Spline-based ellipsoid volume rendering with OptiX"
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

# Single-config generators default to Release unless explicitly specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Common output directories (bin/lib layout).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Dependencies and local CMake modules
# -----------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)

set(SLANG_ENABLE_SLANG_RHI ON CACHE BOOL "" FORCE)
set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  slang_rhi
  GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
  GIT_TAG v0.19.0
)
FetchContent_GetProperties(slang_rhi)
if(NOT slang_rhi_POPULATED)
  FetchContent_Populate(slang_rhi)
endif()

set(SLANG_OVERRIDE_SLANG_RHI_PATH "${slang_rhi_SOURCE_DIR}" CACHE PATH "" FORCE)
add_subdirectory(slang-master EXCLUDE_FROM_ALL)

include(FindOptiX)            # provides: OptiX_ROOT, OptiX_VERSION, OptiX::OptiX
include(FindTorch)            # provides: Torch::Torch

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
set(TRACER_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/fast_ellipsoid_splinetracer")
set(SLANG_DIR      "${TRACER_DIR}/slang")
set(MAIN_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/slang")

# -----------------------------------------------------------------------------
# Main pybind11 extension module
# -----------------------------------------------------------------------------
pybind11_add_module(ellipsoid_splinetracer MODULE
  ${TRACER_DIR}/Forward.cpp
  ${TRACER_DIR}/Forward.h
  ${TRACER_DIR}/GAS.cpp
  ${TRACER_DIR}/GAS.h
  ${TRACER_DIR}/py_binding.cpp
  ${TRACER_DIR}/structs.h
  ${TRACER_DIR}/CUDABuffer.h
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/create_aabbs.h
  ${TRACER_DIR}/initialize_density.cu
  ${TRACER_DIR}/initialize_density.h
)

# Ensure .cu files are treated as CUDA sources (some IDEs/toolchains benefit).
set_source_files_properties(
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/initialize_density.cu
  PROPERTIES LANGUAGE CUDA
)

# -----------------------------------------------------------------------------
# Linking and compile options
# -----------------------------------------------------------------------------
target_link_libraries(ellipsoid_splinetracer PRIVATE
  OptiX::OptiX
  Torch::Torch
  CUDA::cudart
  CUDA::cuda_driver
  slang
  slang-rhi
)

# CUDA diagnostics (project-specific; keep minimal and explicit).
target_compile_options(ellipsoid_splinetracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177>
)

# Platform-specific configuration
if(MSVC)
  set_target_properties(ellipsoid_splinetracer PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    CUDA_RUNTIME_LIBRARY Shared
  )
else()
  target_compile_options(ellipsoid_splinetracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  )
endif()

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Ever Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Project:            ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OptiX:              ${OptiX_VERSION} (${OptiX_ROOT})")
message(STATUS "  Extension target:   ellipsoid_splinetracer")
message(STATUS "========================================")
