# =============================================================================
# fast_ellipsoid_splinetracer - OptiX-based ellipsoid spline tracer
# =============================================================================
# This module builds the Python extension for fast ellipsoid rendering.
# =============================================================================

enable_language(CUDA)

include(SlangOptiXIREmbed)

# -----------------------------------------------------------------------------
# Python module
# -----------------------------------------------------------------------------
set(EXTENSION_SOURCES
    # C++ sources
    Forward.cpp
    Forward.h
    GAS.cpp
    GAS.h
    py_binding.cpp
    structs.h
    CUDABuffer.h

    # CUDA sources
    create_aabbs.cu
    create_aabbs.h
    initialize_density.cu
    initialize_density.h
)

pybind11_add_module(fast_ellipsoid_splinetracer_cpp_extension MODULE
    ${EXTENSION_SOURCES}
)

# -----------------------------------------------------------------------------
# Shader compilation using OptiX-IR
# -----------------------------------------------------------------------------
# Define shader source directories
set(slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/slang)
set(main_slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/../slang)

# Determine CUDA profile from CMAKE_CUDA_ARCHITECTURES
# Extract first architecture (e.g., "86;89" -> "sm_86", or "sm_86" -> "sm_86")
if(CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 _first_arch)
    if(_first_arch MATCHES "^sm_")
        # Already in sm_XX format
        set(OPTIX_SLANG_PROFILE "${_first_arch}")
    else()
        # Just a number, add sm_ prefix
        set(OPTIX_SLANG_PROFILE "sm_${_first_arch}")
    endif()
else()
    set(OPTIX_SLANG_PROFILE "sm_70")  # Default fallback
endif()
message(STATUS "Using Slang CUDA profile: ${OPTIX_SLANG_PROFILE}")

# Get OptiX root directory
if(TARGET OptiX::Headers)
    get_target_property(_optix_inc OptiX::Headers INTERFACE_INCLUDE_DIRECTORIES)
    if(_optix_inc)
        list(GET _optix_inc 0 _optix_root)
        get_filename_component(_optix_root "${_optix_root}" DIRECTORY)
    else()
        set(_optix_root "${OptiX_INSTALL_DIR}")
    endif()
else()
    set(_optix_root "${OptiX_INSTALL_DIR}")
endif()

# Output directory for generated OptiX-IR files
set(OPTIX_IR_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/optix_ir")

# Compile fast_shaders.slang to OptiX-IR (contains all shaders: raygen, miss, anyhit, intersection)
slang_optixir_embed(
    TARGET      fast_ellipsoid_splinetracer_cpp_extension
    NAME        fast_optix_ir_file
    SLANG_FILE  ${slang_dir}/fast_shaders.slang
    OUT_DIR     ${OPTIX_IR_OUT_DIR}
    SYMBOL      fast_optix_ir_file
    OPTIX_ROOT  ${_optix_root}
    PROFILE     ${OPTIX_SLANG_PROFILE}
    INCLUDE_DIRS ${main_slang_dir} ${slang_dir}
)

# Compile shaders.slang to OptiX-IR (contains all shaders: raygen, miss, anyhit, intersection)
slang_optixir_embed(
    TARGET      fast_ellipsoid_splinetracer_cpp_extension
    NAME        optix_ir_file
    SLANG_FILE  ${slang_dir}/shaders.slang
    OUT_DIR     ${OPTIX_IR_OUT_DIR}
    SYMBOL      optix_ir_file
    OPTIX_ROOT  ${_optix_root}
    PROFILE     ${OPTIX_SLANG_PROFILE}
    INCLUDE_DIRS ${main_slang_dir} ${slang_dir}
)

# Note: slang_optixir_embed automatically adds OUT_DIR to target include directories

# Set target properties
set_target_properties(fast_ellipsoid_splinetracer_cpp_extension PROPERTIES
    # Output locations
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/splinetracer/extension"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"

    # Position independent code
    POSITION_INDEPENDENT_CODE ON

    # CUDA settings
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Multi-config generator output directories
foreach(config_type IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${config_type} config_upper)
    set_target_properties(fast_ellipsoid_splinetracer_cpp_extension PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/splinetracer/extension"
        ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# -----------------------------------------------------------------------------
# Link dependencies
# -----------------------------------------------------------------------------
target_link_libraries(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
    ever::common
    Torch::Torch
    CUDA::cudart
)

# Link additional CUDA libraries
target_link_libraries(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
    CUDA::cuda_driver
)

# -----------------------------------------------------------------------------
# Platform-specific settings
# -----------------------------------------------------------------------------
if(MSVC)
    # Use dynamic runtime to match PyTorch
    set_target_properties(fast_ellipsoid_splinetracer_cpp_extension PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
        CUDA_RUNTIME_LIBRARY Shared
    )

    # Linker flags for proper symbol resolution
    target_link_options(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
        /NODEFAULTLIB:libcmt
        /NODEFAULTLIB:libcmtd
    )

    # Compiler options for CUDA files
    target_compile_options(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/MD
            -Xcompiler=-fno-lto
        >
    )
else()
    # Unix-specific settings
    target_compile_options(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --compiler-options=-fPIC
            -Xcompiler=-fno-lto
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-deprecated-declarations
        >
    )
endif()

# -----------------------------------------------------------------------------
# CUDA compile options
# -----------------------------------------------------------------------------
target_compile_options(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -D__CUDA_NO_HALF_OPERATORS__
        -D__CUDA_NO_HALF_CONVERSIONS__
        -D__CUDA_NO_HALF2_OPERATORS__
        --expt-relaxed-constexpr
        --use_fast_math
    >
)

# Debug-specific CUDA options
target_compile_options(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:
        -G
        -O0
    >
)

# -----------------------------------------------------------------------------
# Source file specific properties
# -----------------------------------------------------------------------------
# Mark CUDA sources to ensure proper compilation
set_source_files_properties(
    create_aabbs.cu
    initialize_density.cu
    PROPERTIES
        LANGUAGE CUDA
)