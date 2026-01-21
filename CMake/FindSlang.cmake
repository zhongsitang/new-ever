# FindSlang.cmake - Find Slang SDK
#
# This module finds the Slang shader compiler and libraries.
# It looks in the slang-sdk directory relative to the project root.
#
# Sets:
#   Slang_FOUND       - True if slang was found
#   Slang_INCLUDE_DIR - Slang include directory
#   Slang_SLANGC      - Path to slangc compiler
#   SLANGC            - Alias for Slang_SLANGC (for compatibility)

include_guard(GLOBAL)

# Look for slang-sdk in project root
set(_SLANG_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/slang-sdk")

# Find include directory
find_path(Slang_INCLUDE_DIR
  NAMES slang.h
  PATHS "${_SLANG_SDK_DIR}/include"
  NO_DEFAULT_PATH
)

# Find slangc compiler - prefer Linux binary, fall back to system PATH
find_program(Slang_SLANGC
  NAMES slangc
  PATHS "${_SLANG_SDK_DIR}/bin"
  NO_DEFAULT_PATH
)

# If not found in SDK, try system PATH
if(NOT Slang_SLANGC)
  find_program(Slang_SLANGC NAMES slangc)
endif()

# Set SLANGC alias for SlangPTXEmbed.cmake compatibility
set(SLANGC "${Slang_SLANGC}" CACHE FILEPATH "Path to slangc compiler")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Slang
  REQUIRED_VARS Slang_INCLUDE_DIR Slang_SLANGC
)

if(Slang_FOUND)
  message(STATUS "Slang SDK found:")
  message(STATUS "  Include: ${Slang_INCLUDE_DIR}")
  message(STATUS "  slangc:  ${Slang_SLANGC}")
endif()
