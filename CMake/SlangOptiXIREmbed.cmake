include_guard(GLOBAL)

# slang_optixir_embed(
#   TARGET       <target>           # Target to add the generated .c source to
#   NAME         <symbol>           # C symbol name (MUST be a valid C identifier)
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OPTIX_ROOT   <path>             # OptiX SDK root (REQUIRED; must contain include/)
#   OUT_DIR      <dir>              # Optional output directory (default: ${CMAKE_CURRENT_BINARY_DIR}/optix_ir)
#   INCLUDE_DIRS <dir;...>          # Extra include dirs for slangc
#   DEPENDS      <file;...>         # Extra dependencies (e.g. included .slang/.hlsli)
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
#   NVCC_FLAGS   <flag;...>         # Extra nvcc flags
# )
#
# Pipeline:
#   .slang -> .cu -> .optixir -> .c (via bin2c + post-processing)
#
# Outputs (in OUT_DIR):
#   1) <NAME>.cu
#   2) <NAME>.optixir
#   3) <NAME>.c        (single generated translation unit)
#
# The generated .c contains (WITHOUT static, suitable for extern linkage):
#   const unsigned char <NAME>[];
#   const size_t <NAME>_size;
#
function(slang_optixir_embed)
  set(_one TARGET NAME SLANG_FILE OPTIX_ROOT OUT_DIR)
  set(_multi INCLUDE_DIRS DEPENDS SLANG_FLAGS NVCC_FLAGS)
  cmake_parse_arguments(SOE "" "${_one}" "${_multi}" ${ARGN})

  # ---- Validate required arguments ------------------------------------------
  foreach(_req TARGET NAME SLANG_FILE OPTIX_ROOT)
    if(NOT SOE_${_req})
      message(FATAL_ERROR "slang_optixir_embed: missing ${_req}")
    endif()
  endforeach()

  if(NOT EXISTS "${SOE_OPTIX_ROOT}/include")
    message(FATAL_ERROR
      "slang_optixir_embed: OPTIX_ROOT/include not found: ${SOE_OPTIX_ROOT}")
  endif()

  # ---- NAME is the symbol: must be a valid C identifier ----------------------
  # C identifier: [A-Za-z_][A-Za-z0-9_]*
  if(NOT SOE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "slang_optixir_embed: NAME must be a valid C identifier ([A-Za-z_][A-Za-z0-9_]*), got: ${SOE_NAME}")
  endif()

  # ---- OUT_DIR default -------------------------------------------------------
  if(NOT SOE_OUT_DIR)
    set(SOE_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/optix_ir")
  endif()
  file(MAKE_DIRECTORY "${SOE_OUT_DIR}")

  # ---- Find required tools ---------------------------------------------------
  find_program(SLANGC slangc REQUIRED)
  find_program(BIN2C bin2c REQUIRED)

  if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    find_package(CUDAToolkit REQUIRED)
  endif()

  # ---- Choose NVCC C++ standard based on target (fallback: c++17) -----------
  get_target_property(_tstd "${SOE_TARGET}" CXX_STANDARD)
  if(_tstd AND NOT _tstd STREQUAL "NOTFOUND")
    set(_nvcc_std "c++${_tstd}")
  else()
    set(_nvcc_std "c++17")
  endif()

  # ---- Output paths ----------------------------------------------------------
  set(_cu "${SOE_OUT_DIR}/${SOE_NAME}.cu")
  set(_ir "${SOE_OUT_DIR}/${SOE_NAME}.optixir")
  set(_c "${SOE_OUT_DIR}/${SOE_NAME}.c")
  set(_c_tmp "${SOE_OUT_DIR}/${SOE_NAME}_bin2c_tmp.c")
  set(_script "${SOE_OUT_DIR}/${SOE_NAME}_bin2c.cmake")

  # ---- Build include args for slangc ----------------------------------------
  set(_slang_incs "")
  foreach(_d IN LISTS SOE_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()
  list(APPEND _slang_incs -I "${SOE_OPTIX_ROOT}/include")

  # ---- 1) Slang -> CUDA ------------------------------------------------------
  set(_deps "${SOE_SLANG_FILE}")
  if(SOE_DEPENDS)
    list(APPEND _deps ${SOE_DEPENDS})
  endif()

  add_custom_command(
    OUTPUT "${_cu}"
    COMMAND "${SLANGC}" "${SOE_SLANG_FILE}"
      -target cuda
      -o "${_cu}"
      ${_slang_incs}
      ${SOE_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SOE_NAME}.slang -> ${SOE_NAME}.cu"
    VERBATIM
  )

  # ---- 2) CUDA -> OptiX-IR ---------------------------------------------------
  # Suppress warnings:
  #   177: variable declared but never referenced
  #   550: variable set but never used
  #   20044: extern declaration of the entity treated as static definition
  #          (expected for SLANG_globalParams with OptiX)
  add_custom_command(
    OUTPUT "${_ir}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
      -optix-ir
      -O3
      --std=${_nvcc_std}
      -I "${SOE_OPTIX_ROOT}/include"
      -D SLANG_CUDA_ENABLE_OPTIX
      --diag-suppress=177,550,20044
      ${SOE_NVCC_FLAGS}
      -o "${_ir}" "${_cu}"
    DEPENDS "${_cu}"
    COMMENT "nvcc: ${SOE_NAME}.cu -> ${SOE_NAME}.optixir (std=${_nvcc_std})"
    VERBATIM
  )

  # ---- 3) OptiX-IR -> single .c via bin2c + post-processing -----------------
  # bin2c generates static arrays, but we need extern linkage.
  # We also need to add a size variable for use with optixModuleCreate.
  set(_script_content
"# Run bin2c to generate temporary file
execute_process(
  COMMAND \"${BIN2C}\" --const --name \"${SOE_NAME}\" \"${_ir}\"
  OUTPUT_FILE \"${_c_tmp}\"
  RESULT_VARIABLE _r
)
if(NOT _r EQUAL 0)
  message(FATAL_ERROR \"bin2c failed with code: \${_r}\")
endif()

# Read the temporary file
file(READ \"${_c_tmp}\" _content)

string(REGEX REPLACE
  \"(const unsigned char ${SOE_NAME}\\\\[\\\\] = \\\\{[^}]+\\\\};)\"
  \"\\\\1\\n\\nconst size_t ${SOE_NAME}_size = sizeof(${SOE_NAME});\"
  _content
  \"\${_content}\"
)

# Write the final output
file(WRITE \"${_c}\" \"\${_content}\")

# Clean up temporary file
file(REMOVE \"${_c_tmp}\")
")

  file(GENERATE OUTPUT "${_script}" CONTENT "${_script_content}")

  add_custom_command(
    OUTPUT "${_c}"
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "${_ir}"
    COMMENT "bin2c: ${SOE_NAME}.optixir -> ${SOE_NAME}.c (with extern linkage and size)"
    VERBATIM
  )

  # ---- Attach generated source to the target --------------------------------
  set_source_files_properties("${_c}" PROPERTIES GENERATED TRUE)
  target_sources(${SOE_TARGET} PRIVATE "${_c}")

  # Ensure generation runs as part of the build graph.
  add_custom_target(${SOE_NAME}_optixir_embed DEPENDS "${_c}")
  add_dependencies(${SOE_TARGET} ${SOE_NAME}_optixir_embed)
endfunction()