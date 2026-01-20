# =============================================================================
# GaussianRT - Clean slang-rhi Ray Tracing Implementation
# =============================================================================
# A modern, well-structured implementation of ellipsoid volume rendering
# using slang-rhi hardware ray tracing.
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#
# Test:
#   ./bin/gaussianrt_test
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(gaussianrt
    VERSION 1.0.0
    DESCRIPTION "Modern slang-rhi ray tracing for ellipsoid volume rendering"
    LANGUAGES C CXX CUDA
)

# -----------------------------------------------------------------------------
# Build Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Find Dependencies
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# Use parent project's cmake modules
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# -----------------------------------------------------------------------------
# Slang and slang-rhi
# -----------------------------------------------------------------------------
if(NOT TARGET slang-rhi)
    # Check if we can use the parent slang-master
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../slang-master/CMakeLists.txt")
        message(STATUS "Using slang from parent directory")

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
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../slang-master" slang EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "slang-master not found. Please ensure slang-master directory exists in parent folder.")
    endif()
endif()

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------
set(GAUSSIANRT_HEADERS
    include/Types.h
    include/Device.h
    include/AccelerationStructure.h
    include/RayTracer.h
)

set(GAUSSIANRT_SOURCES
    src/Device.cpp
    src/AccelerationStructure.cpp
    src/RayTracer.cpp
)

set(GAUSSIANRT_SHADERS
    shaders/safe_math.slang
    shaders/sh.slang
    shaders/volume.slang
    shaders/ellipsoid.slang
    shaders/raytracer.slang
    shaders/backward.slang
)

# -----------------------------------------------------------------------------
# Main Library
# -----------------------------------------------------------------------------
add_library(gaussianrt STATIC
    ${GAUSSIANRT_HEADERS}
    ${GAUSSIANRT_SOURCES}
)

target_include_directories(gaussianrt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gaussianrt
    PUBLIC
        slang
        slang-rhi
        CUDA::cudart
        CUDA::cuda_driver
)

# Copy shaders to build directory
add_custom_command(TARGET gaussianrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${CMAKE_BINARY_DIR}/shaders"
    COMMENT "Copying shaders to build directory"
)

# -----------------------------------------------------------------------------
# Test Executable
# -----------------------------------------------------------------------------
add_executable(gaussianrt_test
    test/main.cpp
)

target_link_libraries(gaussianrt_test
    PRIVATE
        gaussianrt
)

# Ensure shaders are available for tests
add_dependencies(gaussianrt_test gaussianrt)

# -----------------------------------------------------------------------------
# Python Bindings (optional)
# -----------------------------------------------------------------------------
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via pip
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_RESULT
        )
        if(PYBIND11_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(WARNING "pybind11 not found. Python bindings will not be built.")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    endif()

    if(BUILD_PYTHON_BINDINGS)
        # Find PyTorch
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_RESULT
        )
        if(TORCH_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
        endif()

        find_package(Torch QUIET)

        if(Torch_FOUND)
            pybind11_add_module(gaussianrt_ext
                python/bindings.cpp
            )

            target_link_libraries(gaussianrt_ext
                PRIVATE
                    gaussianrt
                    ${TORCH_LIBRARIES}
            )

            target_include_directories(gaussianrt_ext
                PRIVATE
                    ${TORCH_INCLUDE_DIRS}
            )

            target_compile_definitions(gaussianrt_ext
                PRIVATE
                    TORCH_EXTENSION_NAME=gaussianrt_ext
            )

            # Copy Python files to build directory
            add_custom_command(TARGET gaussianrt_ext POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${CMAKE_CURRENT_SOURCE_DIR}/python/gaussianrt.py"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/gaussianrt.py"
                COMMENT "Copying Python wrapper"
            )

            message(STATUS "  Python bindings: Enabled (PyTorch ${Torch_VERSION})")
        else()
            message(WARNING "PyTorch not found. Python bindings require PyTorch.")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(gaussianrt PRIVATE /W4)
    target_compile_options(gaussianrt_test PRIVATE /W4)
else()
    target_compile_options(gaussianrt PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_options(gaussianrt_test PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# CUDA specific flags
target_compile_options(gaussianrt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS gaussianrt
    EXPORT gaussianrt-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/gaussianrt
)

install(DIRECTORY shaders/
    DESTINATION share/gaussianrt/shaders
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "GaussianRT Configuration")
message(STATUS "========================================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:           ${CUDAToolkit_VERSION}")
message(STATUS "  Output dir:     ${CMAKE_BINARY_DIR}")
message(STATUS "========================================")
