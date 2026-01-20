# =============================================================================
# Ever - Spline-based Volume Rendering Library
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
include(SlangPTXEmbed)

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
# Shader embedding (Slang -> PTX -> C++ string)
# -----------------------------------------------------------------------------
slang_ptx_embed(
  TARGET       ellipsoid_splinetracer
  NAME         shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/shaders.slang
  OPTIX_ROOT   ${OptiX_ROOT}
  OUT_DIR      ${PTX_OUT_DIR}
  INCLUDE_DIRS ${MAIN_SLANG_DIR} ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
)

slang_ptx_embed(
  TARGET       ellipsoid_splinetracer
  NAME         fast_shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/fast_shaders.slang
  OPTIX_ROOT   ${OptiX_ROOT}
  OUT_DIR      ${PTX_OUT_DIR}
  INCLUDE_DIRS ${MAIN_SLANG_DIR} ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
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
message(STATUS "  PTX OUT_DIR:     ${PTX_OUT_DIR}")
message(STATUS "========================================")
