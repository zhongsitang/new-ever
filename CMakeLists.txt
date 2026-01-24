# =============================================================================
# Ever - Primitive Volume Rendering Library (primtracer)
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
  DESCRIPTION "Primitive-based volume rendering with OptiX (ellipsoid, etc.)"
  LANGUAGES C CXX CUDA
)

# -----------------------------------------------------------------------------
# Global build settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
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
find_package(glm REQUIRED)

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
pybind11_add_module(ellipsoid_tracer MODULE
  ${TRACER_DIR}/ray_pipeline.cpp
  ${TRACER_DIR}/ray_pipeline.h
  ${TRACER_DIR}/accel_structure.cpp
  ${TRACER_DIR}/accel_structure.h
  ${TRACER_DIR}/python_bindings.cpp
  ${TRACER_DIR}/types.h
  ${TRACER_DIR}/primitive_kernels.cu
  ${TRACER_DIR}/primitive_kernels.h
)

set_source_files_properties(
  ${TRACER_DIR}/primitive_kernels.cu
  PROPERTIES LANGUAGE CUDA
)

# -----------------------------------------------------------------------------
# Shader embedding (Slang -> PTX -> C++ string)
# -----------------------------------------------------------------------------
slang_ptx_embed(
  TARGET       ellipsoid_tracer
  NAME         shaders_ptx
  SLANG_FILE   ${SLANG_DIR}/optix_shaders.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS} ${OptiX_ROOT}/include
)

# -----------------------------------------------------------------------------
# Slang PyTorch modules
# -----------------------------------------------------------------------------
slang_add_py_module(backwards_kernel
  SLANG_FILES  ${SLANG_DIR}/backward_pass.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(backwards_kernel PRIVATE Torch::Torch)

slang_add_py_module(sh_kernel
  SLANG_FILES  ${SLANG_DIR}/sh_evaluation.slang
  INCLUDE_DIRS ${SLANG_DIR}
  NVRTC_DIRS   ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(sh_kernel PRIVATE Torch::Torch)

# -----------------------------------------------------------------------------
# Linking and compile options
# -----------------------------------------------------------------------------
target_link_libraries(ellipsoid_tracer PRIVATE
  OptiX::OptiX
  Torch::Torch
  glm::glm
)

target_compile_options(ellipsoid_tracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177,550>
)

if(MSVC)
  # Prevent Windows.h from defining min/max macros
  add_compile_definitions(NOMINMAX)

  set_target_properties(ellipsoid_tracer PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    CUDA_RUNTIME_LIBRARY Shared
  )
else()
  target_compile_options(ellipsoid_tracer PRIVATE
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
message(STATUS "========================================")
