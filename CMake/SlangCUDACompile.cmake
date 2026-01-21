# SlangCUDACompile.cmake - Compile Slang to CUDA source
#
# slang_cuda_compile(
#   TARGET       <target>           # Target to add generated source
#   NAME         <name>             # Base name for output files
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OUT_DIR      <dir>              # Output directory
#   INCLUDE_DIRS <dir;...>          # Include dirs for slangc (-I)
#   ENTRY_POINTS <name;...>         # Entry point function names
# )
#
# Pipeline: .slang -> .cu (with CUDA prelude)

include_guard(GLOBAL)

function(slang_cuda_compile)
  set(_one TARGET NAME SLANG_FILE OUT_DIR)
  set(_multi INCLUDE_DIRS ENTRY_POINTS)
  cmake_parse_arguments(SCC "" "${_one}" "${_multi}" ${ARGN})

  foreach(_req TARGET NAME SLANG_FILE)
    if(NOT SCC_${_req})
      message(FATAL_ERROR "slang_cuda_compile: missing ${_req}")
    endif()
  endforeach()

  if(NOT SCC_OUT_DIR)
    set(SCC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/slang_cuda")
  endif()
  file(MAKE_DIRECTORY "${SCC_OUT_DIR}")

  find_package(Slang REQUIRED)

  set(_cu_file "${SCC_OUT_DIR}/${SCC_NAME}.cu")

  # Build -I args for slangc
  set(_slang_incs "")
  foreach(_d IN LISTS SCC_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()

  # Build entry point flags
  set(_entry_flags "")
  foreach(_ep IN LISTS SCC_ENTRY_POINTS)
    list(APPEND _entry_flags -entry "${_ep}")
  endforeach()

  # Compile .slang -> .cu
  add_custom_command(
    OUTPUT "${_cu_file}"
    COMMAND "${Slang_SLANGC}" "${SCC_SLANG_FILE}"
      -target cuda
      -o "${_cu_file}"
      ${_slang_incs}
      ${_entry_flags}
      -line-directive-mode none
    DEPENDS "${SCC_SLANG_FILE}"
    COMMENT "slangc: ${SCC_NAME}.slang -> ${SCC_NAME}.cu"
    VERBATIM
  )

  # The generated .cu file should be #included by another .cu file
  # This is necessary because CUDA kernels must be in the same compilation unit as their callers
  # Add include directories for slang prelude headers and generated files
  target_include_directories(${SCC_TARGET} PRIVATE
    "${Slang_INCLUDE_DIR}/slang"
    "${SCC_OUT_DIR}"
  )

  # Create custom target for dependency tracking
  add_custom_target(${SCC_NAME}_slang_cuda DEPENDS "${_cu_file}")
  add_dependencies(${SCC_TARGET} ${SCC_NAME}_slang_cuda)
endfunction()
