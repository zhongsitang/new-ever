include_guard(GLOBAL)

# slang_ptx_embed(
#   TARGET       <target>           # Target to add the generated header to
#   NAME         <symbol>           # C++ variable name (valid identifier)
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OUT_DIR      <dir>              # Optional output directory
#   INCLUDE_DIRS <dir;...>          # Extra include dirs for slangc
#   DEPENDS      <file;...>         # Extra dependencies
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
# )
#
# Pipeline:
#   .slang -> .ptx (via slangc -target ptx)
#
# Generated header contains:
#   inline constexpr const char* <NAME> = R"(...)";
#
# This allows direct embedding without bin2c or size tracking.
#
function(slang_ptx_embed)
  set(_one TARGET NAME SLANG_FILE OUT_DIR)
  set(_multi INCLUDE_DIRS DEPENDS SLANG_FLAGS)
  cmake_parse_arguments(SPE "" "${_one}" "${_multi}" ${ARGN})

  # Validate required arguments
  foreach(_req TARGET NAME SLANG_FILE)
    if(NOT SPE_${_req})
      message(FATAL_ERROR "slang_ptx_embed: missing ${_req}")
    endif()
  endforeach()

  # NAME must be a valid C++ identifier
  if(NOT SPE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "slang_ptx_embed: NAME must be a valid C++ identifier, got: ${SPE_NAME}")
  endif()

  # Default output directory
  if(NOT SPE_OUT_DIR)
    set(SPE_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
  endif()
  file(MAKE_DIRECTORY "${SPE_OUT_DIR}")

  # Find slangc
  find_program(SLANGC slangc REQUIRED)

  # Find CUDA toolkit for cuda_fp16.h etc.
  if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit REQUIRED)
  endif()

  # Output paths
  set(_ptx "${SPE_OUT_DIR}/${SPE_NAME}.ptx")
  set(_header "${SPE_OUT_DIR}/${SPE_NAME}.ptx.h")
  set(_script "${SPE_OUT_DIR}/${SPE_NAME}_embed.cmake")

  # Build include args
  set(_slang_incs "")
  foreach(_d IN LISTS SPE_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()

  # Dependencies
  set(_deps "${SPE_SLANG_FILE}")
  if(SPE_DEPENDS)
    list(APPEND _deps ${SPE_DEPENDS})
  endif()

  # Step 1: Slang -> PTX (direct compilation)
  add_custom_command(
    OUTPUT "${_ptx}"
    COMMAND "${SLANGC}" "${SPE_SLANG_FILE}"
      -target ptx
      -o "${_ptx}"
      -O3
      -D SLANG_CUDA_ENABLE_OPTIX
      -I "${CUDAToolkit_INCLUDE_DIRS}"
      ${_slang_incs}
      ${SPE_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SPE_NAME}.slang -> ${SPE_NAME}.ptx"
    VERBATIM
  )

  # Step 2: PTX -> C++ header with raw string literal
  set(_script_content
"# Read PTX file
file(READ \"${_ptx}\" _ptx_content)

# Escape any problematic characters for raw string literal
# Raw strings can contain almost anything except )delimiter\"
# We use a unique delimiter to be safe
string(REPLACE \"\${\" \"\\\\\${\" _ptx_content \"\${_ptx_content}\")

# Generate header with inline constexpr
set(_header_content \"// Auto-generated PTX shader - DO NOT EDIT
// Source: ${SPE_SLANG_FILE}
#pragma once

namespace ptx {

inline constexpr const char* ${SPE_NAME} = R\\\"__PTX__(
\${_ptx_content}
)__PTX__\\\";

} // namespace ptx
\")

file(WRITE \"${_header}\" \"\${_header_content}\")
")

  file(GENERATE OUTPUT "${_script}" CONTENT "${_script_content}")

  add_custom_command(
    OUTPUT "${_header}"
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "${_ptx}"
    COMMENT "Generating ${SPE_NAME}.ptx.h"
    VERBATIM
  )

  # Attach to target
  set_source_files_properties("${_header}" PROPERTIES GENERATED TRUE)
  target_sources(${SPE_TARGET} PRIVATE "${_header}")
  target_include_directories(${SPE_TARGET} PRIVATE "${SPE_OUT_DIR}")

  # Custom target for dependencies
  add_custom_target(${SPE_NAME}_ptx_embed DEPENDS "${_header}")
  add_dependencies(${SPE_TARGET} ${SPE_NAME}_ptx_embed)
endfunction()
