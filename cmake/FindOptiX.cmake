# =============================================================================
# FindOptiX.cmake - Modern CMake OptiX SDK finder
# =============================================================================
# Finds OptiX SDK and creates an IMPORTED target.
#
# Result Variables:
#   OptiX_FOUND        - True if OptiX was found
#   OptiX_INCLUDE_DIR  - OptiX include directory
#   OptiX_ROOT         - OptiX SDK root directory (parent of include/)
#   OptiX_VERSION      - OptiX version (e.g., "7.4.0")
#
# Imported Targets:
#   OptiX::OptiX       - Header-only interface target (preferred)
#   OptiX::Headers     - Alias for compatibility
#
# Usage:
#   find_package(OptiX REQUIRED)
#   target_link_libraries(myapp PRIVATE OptiX::OptiX)
# =============================================================================

if(TARGET OptiX::OptiX)
    return()
endif()

# Cache variable for SDK path
set(OptiX_INSTALL_DIR "${OptiX_INSTALL_DIR}" CACHE PATH "Path to OptiX SDK installation")

# Search paths
set(_optix_search_paths
    "${OptiX_INSTALL_DIR}"
    "$ENV{OptiX_INSTALL_DIR}"
    "$ENV{OPTIX_PATH}"
    "$ENV{OptiX_ROOT}"
)

# Require 64-bit
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "OptiX requires 64-bit build")
endif()

# Find include directory
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${_optix_search_paths}
    PATH_SUFFIXES include
    DOC "OptiX include directory"
)

# Standard find handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX REQUIRED_VARS OptiX_INCLUDE_DIR)

if(NOT OptiX_FOUND)
    message(FATAL_ERROR
        "OptiX SDK not found!\n"
        "Set OptiX_INSTALL_DIR to your OptiX SDK path:\n"
        "  cmake -DOptiX_INSTALL_DIR=/path/to/optix ..."
    )
endif()

# Derive root directory from include path
get_filename_component(OptiX_ROOT "${OptiX_INCLUDE_DIR}" DIRECTORY)
set(OptiX_ROOT "${OptiX_ROOT}" CACHE PATH "OptiX SDK root directory")

# Extract version from optix.h
if(EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
    file(STRINGS "${OptiX_INCLUDE_DIR}/optix.h" _ver_line REGEX "#define OPTIX_VERSION [0-9]+")
    if(_ver_line)
        string(REGEX REPLACE ".*#define OPTIX_VERSION ([0-9]+).*" "\\1" _ver "${_ver_line}")
        math(EXPR _major "${_ver} / 10000")
        math(EXPR _minor "(${_ver} % 10000) / 100")
        math(EXPR _micro "${_ver} % 100")
        set(OptiX_VERSION "${_major}.${_minor}.${_micro}" CACHE STRING "OptiX version")
    endif()
endif()

# Create imported target
add_library(OptiX::OptiX INTERFACE IMPORTED)
set_target_properties(OptiX::OptiX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE_DIR}"
)

# Compatibility alias
add_library(OptiX::Headers ALIAS OptiX::OptiX)

message(STATUS "Found OptiX ${OptiX_VERSION}: ${OptiX_ROOT}")

mark_as_advanced(OptiX_INCLUDE_DIR OptiX_ROOT)