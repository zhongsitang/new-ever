include_guard(GLOBAL)

# slang_cuda_embed(
#   TARGET       <target>           # Target to add include directory
#   NAME         <symbol>           # C symbol name (MUST be a valid C identifier)
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OUT_DIR      <dir>              # Optional output directory
#   INCLUDE_DIRS <dir;...>          # Include dirs for slangc (-I)
#   DEPENDS      <file;...>         # Extra dependencies
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
# )
#
# Pipeline: .slang -> .cuh (CUDA header with kernel implementations)
#
function(slang_cuda_embed)
  set(_one TARGET NAME SLANG_FILE OUT_DIR)
  set(_multi INCLUDE_DIRS DEPENDS SLANG_FLAGS)
  cmake_parse_arguments(SCE "" "${_one}" "${_multi}" ${ARGN})

  foreach(_req TARGET NAME SLANG_FILE)
    if(NOT SCE_${_req})
      message(FATAL_ERROR "slang_cuda_embed: missing ${_req}")
    endif()
  endforeach()

  if(NOT SCE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "slang_cuda_embed: NAME must be a valid C identifier, got: ${SCE_NAME}")
  endif()

  if(NOT SCE_OUT_DIR)
    set(SCE_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda")
  endif()
  file(MAKE_DIRECTORY "${SCE_OUT_DIR}")

  find_program(SLANGC slangc REQUIRED)

  set(_cuda_header "${SCE_OUT_DIR}/${SCE_NAME}_cuda.h")

  # Build -I args for slangc
  set(_slang_incs "")
  foreach(_d IN LISTS SCE_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()

  set(_deps "${SCE_SLANG_FILE}")
  if(SCE_DEPENDS)
    list(APPEND _deps ${SCE_DEPENDS})
  endif()

  # Slang -> CUDA source
  add_custom_command(
    OUTPUT "${_cuda_header}"
    COMMAND "${SLANGC}" "${SCE_SLANG_FILE}"
      -target cuda
      -o "${_cuda_header}"
      ${_slang_incs}
      ${SCE_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SCE_NAME}.slang -> ${SCE_NAME}_cuda.h"
    VERBATIM
  )

  # Add include directory to target
  target_include_directories(${SCE_TARGET} PRIVATE "${SCE_OUT_DIR}")

  # Ensure header is generated before compiling
  add_custom_target(${SCE_NAME}_cuda_embed DEPENDS "${_cuda_header}")
  add_dependencies(${SCE_TARGET} ${SCE_NAME}_cuda_embed)
endfunction()
