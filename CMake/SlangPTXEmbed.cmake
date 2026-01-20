include_guard(GLOBAL)

# slang_ptx_embed(
#   TARGET       <target>           # Target to add the generated .cpp source to
#   NAME         <symbol>           # C symbol name (MUST be a valid C identifier)
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OPTIX_ROOT   <path>             # OptiX SDK root (REQUIRED; must contain include/)
#   OUT_DIR      <dir>              # Optional output directory
#   INCLUDE_DIRS <dir;...>          # Extra include dirs for slangc
#   DEPENDS      <file;...>         # Extra dependencies
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
# )
#
# Pipeline: .slang -> .ptx -> .cpp (embedded as const char*)
#
# The generated .cpp contains:
#   extern const char <NAME>[];
#
function(slang_ptx_embed)
  set(_one TARGET NAME SLANG_FILE OPTIX_ROOT OUT_DIR)
  set(_multi INCLUDE_DIRS DEPENDS SLANG_FLAGS)
  cmake_parse_arguments(SPE "" "${_one}" "${_multi}" ${ARGN})

  # Validate required arguments
  foreach(_req TARGET NAME SLANG_FILE OPTIX_ROOT)
    if(NOT SPE_${_req})
      message(FATAL_ERROR "slang_ptx_embed: missing ${_req}")
    endif()
  endforeach()

  if(NOT EXISTS "${SPE_OPTIX_ROOT}/include")
    message(FATAL_ERROR "slang_ptx_embed: OPTIX_ROOT/include not found: ${SPE_OPTIX_ROOT}")
  endif()

  # NAME must be a valid C identifier
  if(NOT SPE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "slang_ptx_embed: NAME must be a valid C identifier, got: ${SPE_NAME}")
  endif()

  # OUT_DIR default
  if(NOT SPE_OUT_DIR)
    set(SPE_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
  endif()
  file(MAKE_DIRECTORY "${SPE_OUT_DIR}")

  # Find slangc
  find_program(SLANGC slangc REQUIRED)

  # Output paths
  set(_ptx "${SPE_OUT_DIR}/${SPE_NAME}.ptx")
  set(_cpp "${SPE_OUT_DIR}/${SPE_NAME}.cpp")
  set(_script "${SPE_OUT_DIR}/${SPE_NAME}_embed.cmake")

  # Build include args for slangc
  set(_slang_incs "")
  foreach(_d IN LISTS SPE_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()
  list(APPEND _slang_incs -I "${SPE_OPTIX_ROOT}/include")

  # Dependencies
  set(_deps "${SPE_SLANG_FILE}")
  if(SPE_DEPENDS)
    list(APPEND _deps ${SPE_DEPENDS})
  endif()

  # 1) Slang -> PTX directly
  add_custom_command(
    OUTPUT "${_ptx}"
    COMMAND "${SLANGC}" "${SPE_SLANG_FILE}"
      -target ptx
      -o "${_ptx}"
      ${_slang_incs}
      ${SPE_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SPE_NAME}.slang -> ${SPE_NAME}.ptx"
    VERBATIM
  )

  # 2) PTX -> C++ string literal
  set(_script_content
"# Convert PTX to C++ string literal
file(READ \"${_ptx}\" _ptx_content)

# Escape backslashes and quotes for C string
string(REPLACE \"\\\\\" \"\\\\\\\\\" _ptx_content \"\${_ptx_content}\")
string(REPLACE \"\\\"\" \"\\\\\\\"\" _ptx_content \"\${_ptx_content}\")

# Convert to raw string literal for cleaner embedding
file(WRITE \"${_cpp}\" \"// Generated from ${SPE_NAME}.slang
extern const char ${SPE_NAME}[] = R\\\"ptx(
\${_ptx_content})ptx\\\";
\")
")

  file(GENERATE OUTPUT "${_script}" CONTENT "${_script_content}")

  add_custom_command(
    OUTPUT "${_cpp}"
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "${_ptx}"
    COMMENT "embed: ${SPE_NAME}.ptx -> ${SPE_NAME}.cpp"
    VERBATIM
  )

  # Attach generated source to the target
  set_source_files_properties("${_cpp}" PROPERTIES GENERATED TRUE)
  target_sources(${SPE_TARGET} PRIVATE "${_cpp}")

  # Ensure generation runs as part of the build graph
  add_custom_target(${SPE_NAME}_ptx_embed DEPENDS "${_cpp}")
  add_dependencies(${SPE_TARGET} ${SPE_NAME}_ptx_embed)
endfunction()
