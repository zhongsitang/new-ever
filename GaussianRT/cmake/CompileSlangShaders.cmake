# CMake functions for compiling Slang shaders

# Find slangc compiler
find_program(SLANGC_EXECUTABLE slangc
    HINTS
        "${SLANG_SDK_PATH}/build/Release/bin"
        "${SLANG_SDK_PATH}/bin"
    DOC "Slang compiler"
)

if(NOT SLANGC_EXECUTABLE)
    message(WARNING "slangc not found. Shader compilation will be skipped.")
endif()

# Function to compile Slang shaders to embedded header
function(compile_slang_to_header)
    cmake_parse_arguments(SLANG "" "NAME;OUTPUT_DIR" "SOURCES;INCLUDE_DIRS" ${ARGN})

    if(NOT SLANGC_EXECUTABLE)
        # Create a dummy header if slangc is not available
        set(HEADER_FILE "${SLANG_OUTPUT_DIR}/${SLANG_NAME}.h")
        file(WRITE ${HEADER_FILE}
            "// Auto-generated shader header (placeholder)\n"
            "#pragma once\n"
            "namespace gaussian_rt {\n"
            "static const char* ${SLANG_NAME}_source = \"\";\n"
            "static const size_t ${SLANG_NAME}_size = 0;\n"
            "}\n"
        )
        return()
    endif()

    set(INCLUDE_FLAGS "")
    foreach(INC_DIR ${SLANG_INCLUDE_DIRS})
        list(APPEND INCLUDE_FLAGS "-I${INC_DIR}")
    endforeach()

    set(OUTPUT_FILE "${SLANG_OUTPUT_DIR}/${SLANG_NAME}.spirv")
    set(HEADER_FILE "${SLANG_OUTPUT_DIR}/${SLANG_NAME}.h")

    # Ensure output directory exists
    file(MAKE_DIRECTORY ${SLANG_OUTPUT_DIR})

    # Compile to SPIR-V
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${SLANGC_EXECUTABLE}
            ${SLANG_SOURCES}
            -target spirv
            -o ${OUTPUT_FILE}
            -matrix-layout-column-major
            ${INCLUDE_FLAGS}
        DEPENDS ${SLANG_SOURCES}
        COMMENT "Compiling Slang to SPIR-V: ${SLANG_NAME}"
        VERBATIM
    )

    # Generate C++ header with embedded shader
    add_custom_command(
        OUTPUT ${HEADER_FILE}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating shader header: ${SLANG_NAME}.h"
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shader.py
            ${OUTPUT_FILE} ${HEADER_FILE} ${SLANG_NAME}
        DEPENDS ${OUTPUT_FILE}
        COMMENT "Generating embedded shader header: ${SLANG_NAME}.h"
        VERBATIM
    )

    # Create a target for the shader
    add_custom_target(${SLANG_NAME}_shaders DEPENDS ${HEADER_FILE})
endfunction()

# Function to compile Slang shaders to CUDA
function(compile_slang_to_cuda)
    cmake_parse_arguments(SLANG "" "NAME;OUTPUT_DIR" "SOURCES;INCLUDE_DIRS" ${ARGN})

    set(OUTPUT_FILE "${SLANG_OUTPUT_DIR}/${SLANG_NAME}.cu")

    # Ensure output directory exists
    file(MAKE_DIRECTORY ${SLANG_OUTPUT_DIR})

    if(NOT SLANGC_EXECUTABLE)
        # Create a placeholder .cu file
        file(WRITE ${OUTPUT_FILE}
            "// Auto-generated CUDA kernel (placeholder)\n"
            "#include <cuda_runtime.h>\n\n"
            "namespace gaussian_rt {\n"
            "void backward_kernel_placeholder() {}\n"
            "}\n"
        )
        return()
    endif()

    set(INCLUDE_FLAGS "")
    foreach(INC_DIR ${SLANG_INCLUDE_DIRS})
        list(APPEND INCLUDE_FLAGS "-I${INC_DIR}")
    endforeach()

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${SLANGC_EXECUTABLE}
            ${SLANG_SOURCES}
            -target cuda
            -o ${OUTPUT_FILE}
            ${INCLUDE_FLAGS}
        DEPENDS ${SLANG_SOURCES}
        COMMENT "Compiling Slang to CUDA: ${SLANG_NAME}"
        VERBATIM
    )

    # Create a target for the CUDA file
    add_custom_target(${SLANG_NAME}_cuda DEPENDS ${OUTPUT_FILE})
endfunction()
