include_guard(GLOBAL)

# slang_ptx_embed(
#   TARGET       <target>           # Target to add the generated .cpp source to
#   NAME         <symbol>           # C symbol name (MUST be a valid C identifier)
#   SLANG_FILE   <file.slang>       # Input Slang file
#   OUT_DIR      <dir>              # Optional output directory
#   INCLUDE_DIRS <dir;...>          # Include dirs for slangc (-I)
#   NVRTC_DIRS   <dir;...>          # Include dirs for nvrtc (-Xnvrtc -I)
#   DEPENDS      <file;...>         # Extra dependencies
#   SLANG_FLAGS  <flag;...>         # Extra slangc flags
# )
#
# Pipeline: .slang -> .ptx -> .cpp (embedded as const char*)
#
function(slang_ptx_embed)
  set(_one TARGET NAME SLANG_FILE OUT_DIR)
  set(_multi INCLUDE_DIRS NVRTC_DIRS DEPENDS SLANG_FLAGS)
  cmake_parse_arguments(SPE "" "${_one}" "${_multi}" ${ARGN})

  foreach(_req TARGET NAME SLANG_FILE)
    if(NOT SPE_${_req})
      message(FATAL_ERROR "slang_ptx_embed: missing ${_req}")
    endif()
  endforeach()

  if(NOT SPE_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "slang_ptx_embed: NAME must be a valid C identifier, got: ${SPE_NAME}")
  endif()

  if(NOT SPE_OUT_DIR)
    set(SPE_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
  endif()
  file(MAKE_DIRECTORY "${SPE_OUT_DIR}")

  find_program(SLANGC slangc REQUIRED)

  set(_ptx "${SPE_OUT_DIR}/${SPE_NAME}.ptx")
  set(_cpp "${SPE_OUT_DIR}/${SPE_NAME}.cpp")
  set(_script "${SPE_OUT_DIR}/${SPE_NAME}_embed.cmake")

  # Build -I args for slangc
  set(_slang_incs "")
  foreach(_d IN LISTS SPE_INCLUDE_DIRS)
    list(APPEND _slang_incs -I "${_d}")
  endforeach()

  # Build -Xnvrtc -I args
  set(_nvrtc_incs "")
  foreach(_d IN LISTS SPE_NVRTC_DIRS)
    list(APPEND _nvrtc_incs -Xnvrtc -I "${_d}")
  endforeach()

  set(_deps "${SPE_SLANG_FILE}")
  if(SPE_DEPENDS)
    list(APPEND _deps ${SPE_DEPENDS})
  endif()

  # Slang -> PTX
  add_custom_command(
    OUTPUT "${_ptx}"
    COMMAND "${SLANGC}" "${SPE_SLANG_FILE}"
      -target ptx
      -o "${_ptx}"
      ${_slang_incs}
      ${_nvrtc_incs}
      ${SPE_SLANG_FLAGS}
    DEPENDS ${_deps}
    COMMENT "slangc: ${SPE_NAME}.slang -> ${SPE_NAME}.ptx"
    VERBATIM
  )

  # PTX -> C++ raw string literal
  set(_script_content
"file(READ \"${_ptx}\" _ptx)
string(REPLACE \"\\\\\" \"\\\\\\\\\" _ptx \"\${_ptx}\")
string(REPLACE \"\\\"\" \"\\\\\\\"\" _ptx \"\${_ptx}\")
file(WRITE \"${_cpp}\" \"// Generated from ${SPE_NAME}.slang
extern const char ${SPE_NAME}[] = R\\\"ptx(\${_ptx})ptx\\\";
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

  set_source_files_properties("${_cpp}" PROPERTIES GENERATED TRUE)
  target_sources(${SPE_TARGET} PRIVATE "${_cpp}")

  add_custom_target(${SPE_NAME}_ptx_embed DEPENDS "${_cpp}")
  add_dependencies(${SPE_TARGET} ${SPE_NAME}_ptx_embed)
endfunction()
