# =============================================================================
# Ever - Spline-based Volume Rendering Library
# =============================================================================
# Modern CMake (3.28+) configuration.
#
# Build:
#   cmake --preset release
#   cmake --build --preset release
#
# Shader compilation:
#   Slang shaders are compiled at runtime using the Slang C++ API.
#   No build-time PTX generation needed - shaders are compiled to PTX
#   when the Forward class is constructed.
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

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Dependencies and local CMake modules
# -----------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(FindOptiX)
include(FindTorch)
include(FindSlang)

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
set(TRACER_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/fast_ellipsoid_splinetracer")
set(SLANG_DIR      "${TRACER_DIR}/slang")
set(MAIN_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/splinetracers/slang")
set(PTX_OUT_DIR    "${CMAKE_CURRENT_BINARY_DIR}/ptx")

# -----------------------------------------------------------------------------
# Main pybind11 extension module
# -----------------------------------------------------------------------------
pybind11_add_module(ellipsoid_splinetracer MODULE
  ${TRACER_DIR}/Forward.cpp
  ${TRACER_DIR}/Forward.h
  ${TRACER_DIR}/SlangCompiler.cpp
  ${TRACER_DIR}/SlangCompiler.h
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

set_source_files_properties(
  ${TRACER_DIR}/create_aabbs.cu
  ${TRACER_DIR}/initialize_density.cu
  PROPERTIES LANGUAGE CUDA
)

# -----------------------------------------------------------------------------
# Slang runtime compilation (no build-time PTX embedding needed)
# -----------------------------------------------------------------------------
# Slang shaders are now compiled at runtime using the Slang C++ API.
# The .slang files are loaded and compiled to PTX when Forward is constructed.
# Search paths for import resolution:
#   - ${SLANG_DIR}      : Local shader modules (tri-intersect, spline-machine, etc.)
#   - ${MAIN_SLANG_DIR} : Shared modules (safe-math, sh, optix, tensor-util)

# -----------------------------------------------------------------------------
# Linking and compile options
# -----------------------------------------------------------------------------
target_link_libraries(ellipsoid_splinetracer PRIVATE
  OptiX::OptiX
  Torch::Torch
  Slang::Slang
  CUDA::cudart
  CUDA::cuda_driver
)

target_compile_options(ellipsoid_splinetracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177>
)

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
message(STATUS "  Project:         ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs:      ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OptiX:           ${OptiX_VERSION} (${OptiX_ROOT})")
message(STATUS "  Slang:           ${Slang_LIBRARY}")
message(STATUS "  Slang mode:      Runtime compilation (C++ API)")
message(STATUS "========================================")
