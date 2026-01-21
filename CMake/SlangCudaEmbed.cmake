include_guard(GLOBAL)

# slang_cuda_compile(
#   TARGET       <target>           # Target to add the generated .cu source to
#   NAME         <name>             # Base name for output files
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OUT_DIR      <dir>              # Optional output directory (default: ${CMAKE_CURRENT_BINARY_DIR}/slang_cuda)
#   INCLUDE_DIRS <dir;...>          # Extra include dirs for slangc
#   DEPENDS      <file;...>         # Extra dependencies (e.g. included .slang files)
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
# )
#
# Pipeline:
#   .slang -> .cu (via slangc -target cuda)
#
# This is for regular CUDA kernels (not OptiX shaders).
# The generated .cu file is added as a source to the target.
#
function(slang_cuda_compile)
  set(_one TARGET NAME SLANG_FILE OUT_DIR)
  set(_multi INCLUDE_DIRS DEPENDS SLANG_FLAGS)
  cmake_parse_arguments(SCC "" "${_one}" "${_multi}" ${ARGN})

  # ---- Validate required arguments ------------------------------------------
  foreach(_req TARGET NAME SLANG_FILE)
    if(NOT SCC_${_req})
      message(FATAL_ERROR "slang_cuda_compile: missing ${_req}")
    endif()
  endforeach()

  # ---- OUT_DIR default -------------------------------------------------------
  if(NOT SCC_OUT_DIR)
    set(SCC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/slang_cuda")
  endif()
  file(MAKE_DIRECTORY "${SCC_OUT_DIR}")

  # ---- Find slangc -----------------------------------------------------------
  find_program(SLANGC slangc REQUIRED)

  # ---- Output path -----------------------------------------------------------
  set(_cu "${SCC_OUT_DIR}/${SCC_NAME}.cu")

  # ---- Build include args for slangc ----------------------------------------
  set(_slang_incs "")
  foreach(_d IN LISTS SCC_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()

  # ---- Dependencies ----------------------------------------------------------
  set(_deps "${SCC_SLANG_FILE}")
  if(SCC_DEPENDS)
    list(APPEND _deps ${SCC_DEPENDS})
  endif()

  # ---- Slang -> CUDA ---------------------------------------------------------
  add_custom_command(
    OUTPUT "${_cu}"
    COMMAND "${SLANGC}" "${SCC_SLANG_FILE}"
      -target cuda
      -o "${_cu}"
      ${_slang_incs}
      ${SCC_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SCC_NAME}.slang -> ${SCC_NAME}.cu"
    VERBATIM
  )

  # ---- Attach generated source to the target --------------------------------
  set_source_files_properties("${_cu}" PROPERTIES
    GENERATED TRUE
    LANGUAGE CUDA
  )
  target_sources(${SCC_TARGET} PRIVATE "${_cu}")

  # Ensure generation runs as part of the build graph.
  add_custom_target(${SCC_NAME}_slang_cuda DEPENDS "${_cu}")
  add_dependencies(${SCC_TARGET} ${SCC_NAME}_slang_cuda)
endfunction()
