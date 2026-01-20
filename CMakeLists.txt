# =============================================================================
# Ever - Spline-based Volume Rendering Library
# =============================================================================
# Modern CMake (3.28+) with C++20 and direct Slang->PTX compilation.
#
# Build (presets recommended):
#   cmake --preset release
#   cmake --build --preset release
#
# Target:
#   - Pybind11 extension module: ellipsoid_splinetracer
#
# Shader pipeline (simplified):
#   .slang -> .ptx -> inline constexpr string
#
# =============================================================================

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(splinetracers
  VERSION 2.0.0
  DESCRIPTION "Spline-based ellipsoid volume rendering with OptiX 9.1"
  LANGUAGES C CXX CUDA
)

# -----------------------------------------------------------------------------
# Global build settings - C++20
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
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

include(FindOptiX)            # provides: OptiX_ROOT, OptiX_VERSION, OptiX::OptiX
include(FindTorch)            # provides: Torch::Torch
include(SlangPTXEmbed)        # provides: slang_ptx_embed() - NEW simplified pipeline

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
set(TRACER_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/fast_ellipsoid_splinetracer")
set(SLANG_DIR      "${TRACER_DIR}/slang")
set(MAIN_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/slang")

# Output directory for generated PTX headers
set(PTX_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")

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

# Ensure .cu files are treated as CUDA sources
set_source_files_properties(
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/initialize_density.cu
  PROPERTIES LANGUAGE CUDA
)

# -----------------------------------------------------------------------------
# Shader embedding (Slang -> PTX -> inline constexpr string)
# -----------------------------------------------------------------------------
# This simplified pipeline:
#   1. Compiles .slang directly to .ptx using slangc
#   2. Embeds PTX as a raw string literal in a header file
#   3. No bin2c, no size variable needed - just const char*
#
slang_ptx_embed(
  TARGET       ellipsoid_splinetracer
  NAME         shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/shaders.slang
  OUT_DIR      ${PTX_OUT_DIR}
  INCLUDE_DIRS ${MAIN_SLANG_DIR} ${SLANG_DIR} ${OptiX_ROOT}/include
)

slang_ptx_embed(
  TARGET       ellipsoid_splinetracer
  NAME         fast_shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/fast_shaders.slang
  OUT_DIR      ${PTX_OUT_DIR}
  INCLUDE_DIRS ${MAIN_SLANG_DIR} ${SLANG_DIR} ${OptiX_ROOT}/include
)

# -----------------------------------------------------------------------------
# Linking and compile options
# -----------------------------------------------------------------------------
target_link_libraries(ellipsoid_splinetracer PRIVATE
  OptiX::OptiX
  Torch::Torch
  CUDA::cudart
  CUDA::cuda_driver
)

# C++20 features
target_compile_features(ellipsoid_splinetracer PRIVATE cxx_std_20)

# CUDA diagnostics (project-specific; keep minimal and explicit)
target_compile_options(ellipsoid_splinetracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177>
)

# Platform-specific configuration
if(MSVC)
  set_target_properties(ellipsoid_splinetracer PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    CUDA_RUNTIME_LIBRARY Shared
  )
  # Enable designated initializers and other C++20 features
  target_compile_options(ellipsoid_splinetracer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /permissive->
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
message(STATUS "  C++ Standard:       C++${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA archs:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OptiX:              ${OptiX_VERSION} (${OptiX_ROOT})")
message(STATUS "  PTX OUT_DIR:        ${PTX_OUT_DIR}")
message(STATUS "  Extension target:   ellipsoid_splinetracer")
message(STATUS "========================================")
