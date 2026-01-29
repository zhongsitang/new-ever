#[[
  Slang.cmake - CMake utilities for Slang shader compilation
  
  Provides two main functions:
    - slang_ptx_embed()    : Compile .slang to embedded PTX header
    - slang_add_py_module(): Compile .slang to PyTorch/pybind11 module
]]

include_guard(GLOBAL)

#=============================================================================
# Configuration
#=============================================================================

find_program(SLANGC slangc REQUIRED)

set(SLANG_DEFAULT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/slang"
    CACHE PATH "Default output directory for Slang generated files")

file(MAKE_DIRECTORY "${SLANG_DEFAULT_OUTPUT_DIR}")

#=============================================================================
# Internal Helpers
#=============================================================================

# Validate required arguments exist
macro(_slang_require_args prefix)
  foreach(_arg ${ARGN})
    if(NOT DEFINED ${prefix}_${_arg} OR "${${prefix}_${_arg}}" STREQUAL "")
      message(FATAL_ERROR "${CMAKE_CURRENT_FUNCTION}: missing required argument ${_arg}")
    endif()
  endforeach()
endmacro()

# Set output directory to default if not specified
macro(_slang_default_outdir prefix)
  if(NOT ${prefix}_OUT_DIR)
    set(${prefix}_OUT_DIR "${SLANG_DEFAULT_OUTPUT_DIR}")
  endif()
endmacro()

# Build include arguments for slangc
#   -I <dir>           for regular includes
#   -Xnvrtc -I<dir>    for NVRTC includes
function(_slang_build_include_args out_var prefix)
  set(_args "")
  
  foreach(_dir IN LISTS ${prefix}_INCLUDE_DIRS)
    list(APPEND _args -I "${_dir}")
  endforeach()
  
  foreach(_dir IN LISTS ${prefix}_NVRTC_DIRS)
    list(APPEND _args -Xnvrtc "-I${_dir}")
  endforeach()
  
  set(${out_var} ${_args} PARENT_SCOPE)
endfunction()

#=============================================================================
# Public API
#=============================================================================

#[[
  slang_ptx_embed - Compile Slang to embedded PTX header
  
  Usage:
    slang_ptx_embed(
      TARGET       <target>           # Target to add include directory and dependency
      NAME         <symbol>           # C symbol name (must be valid C identifier)
      SLANG_FILE   <file.slang>       # Input Slang file
      [OUT_DIR     <dir>]             # Output directory (default: SLANG_DEFAULT_OUTPUT_DIR)
      [INCLUDE_DIRS <dir>...]         # Include directories for slangc
      [NVRTC_DIRS   <dir>...]         # Include directories passed to NVRTC
      [DEPENDS      <file>...]        # Additional file dependencies
      [SLANG_FLAGS  <flag>...]        # Additional slangc flags
    )
  
  Output: ${OUT_DIR}/${NAME}.h containing embedded PTX as C array
]]
function(slang_ptx_embed)
  cmake_parse_arguments(PARSE_ARGV 0 ARG
    ""
    "TARGET;NAME;SLANG_FILE;OUT_DIR"
    "INCLUDE_DIRS;NVRTC_DIRS;DEPENDS;SLANG_FLAGS"
  )

  # Validation
  _slang_require_args(ARG TARGET NAME SLANG_FILE)
  
  if(NOT ARG_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "slang_ptx_embed: NAME '${ARG_NAME}' is not a valid C identifier")
  endif()

  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "slang_ptx_embed: TARGET '${ARG_TARGET}' does not exist")
  endif()

  # Setup paths
  _slang_default_outdir(ARG)
  set(_output_header "${ARG_OUT_DIR}/${ARG_NAME}.h")
  
  # Build compiler arguments
  _slang_build_include_args(_include_args ARG)

  # Generate header with embedded PTX
  add_custom_command(
    OUTPUT "${_output_header}"
    COMMAND "${SLANGC}"
      "${ARG_SLANG_FILE}"
      -target ptx
      -o "${_output_header}"
      -source-embed-style default
      -source-embed-name "${ARG_NAME}"
      ${_include_args}
      ${ARG_SLANG_FLAGS}
    DEPENDS "${ARG_SLANG_FILE}" ${ARG_DEPENDS}
    COMMENT "[slang] ${ARG_NAME}: .slang -> .h (embedded PTX)"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  # Create target and wire up dependencies
  set(_gen_target "${ARG_NAME}_ptx_embed")
  add_custom_target(${_gen_target} DEPENDS "${_output_header}")
  
  target_include_directories(${ARG_TARGET} PRIVATE "${ARG_OUT_DIR}")
  add_dependencies(${ARG_TARGET} ${_gen_target})
endfunction()

#[[
  slang_add_py_module - Compile Slang to PyTorch Python module

  Usage:
    slang_add_py_module(<name>
      SLANG_FILES   <file.slang>...   # Input Slang files (compiled together)
      [OUT_DIR      <dir>]            # Generated output directory (default: SLANG_DEFAULT_OUTPUT_DIR)
      [INCLUDE_DIRS <dir>...]         # Include directories for slangc
      [NVRTC_DIRS   <dir>...]         # Include directories passed to NVRTC
      [DEPENDS      <file>...]        # Additional file dependencies
      [SLANG_FLAGS  <flag>...]        # Additional slangc flags
    )

  Creates a pybind11 module target named <name>
  Output: ${OUT_DIR}/${name}.cpp (torch binding) + ${OUT_DIR}/${name}.cu (cuda impl)
]]
function(slang_add_py_module _name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    ""
    "OUT_DIR"
    "SLANG_FILES;INCLUDE_DIRS;NVRTC_DIRS;DEPENDS;SLANG_FLAGS"
  )

  # Validation
  if(NOT ARG_SLANG_FILES)
    message(FATAL_ERROR "slang_add_py_module: SLANG_FILES is required")
  endif()

  # Setup paths
  _slang_default_outdir(ARG)
  set(_output_cpp "${ARG_OUT_DIR}/${_name}.cpp")
  set(_output_cu "${ARG_OUT_DIR}/${_name}.cu")

  # Build compiler arguments
  _slang_build_include_args(_include_args ARG)

  # Generate C++ binding file (-target torch)
  add_custom_command(
    OUTPUT "${_output_cpp}"
    COMMAND "${SLANGC}"
      ${ARG_SLANG_FILES}
      -target torch-binding
      -o "${_output_cpp}"
      ${_include_args}
      ${ARG_SLANG_FLAGS}
    DEPENDS ${ARG_SLANG_FILES} ${ARG_DEPENDS}
    COMMENT "[slang] ${_name}: .slang -> .cpp (torch binding)"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  # Generate CUDA implementation file (-target cuda)
  add_custom_command(
    OUTPUT "${_output_cu}"
    COMMAND "${SLANGC}"
      ${ARG_SLANG_FILES}
      -target cuda
      -o "${_output_cu}"
      ${_include_args}
      ${ARG_SLANG_FLAGS}
    DEPENDS ${ARG_SLANG_FILES} ${ARG_DEPENDS}
    COMMENT "[slang] ${_name}: .slang -> .cu (cuda impl)"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  # Create generation target
  set(_gen_target "${_name}_slang_gen")
  add_custom_target(${_gen_target} DEPENDS "${_output_cpp}" "${_output_cu}")

  # Create pybind11 module with both files
  pybind11_add_module(${_name} "${_output_cpp}" "${_output_cu}")
  add_dependencies(${_name} ${_gen_target})
  target_compile_definitions(${_name} PRIVATE "TORCH_EXTENSION_NAME=${_name}")
  target_compile_options(${_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012,177,550>
  )
endfunction()