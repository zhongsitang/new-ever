# FindSlang.cmake
# ----------------
# Find the Slang shader compiler library
#
# This module defines:
#   Slang_FOUND        - True if Slang was found
#   Slang_INCLUDE_DIR  - Include directory for Slang headers
#   Slang_LIBRARY      - The Slang library to link
#   Slang::Slang       - Imported target for Slang
#
# Search paths (in order):
#   1. SLANG_ROOT environment variable
#   2. SLANG_DIR CMake variable
#   3. Common installation paths
#
# Example usage:
#   find_package(Slang REQUIRED)
#   target_link_libraries(myapp PRIVATE Slang::Slang)

include(FindPackageHandleStandardArgs)

# Search paths
set(_slang_search_paths
    "${SLANG_ROOT}"
    "$ENV{SLANG_ROOT}"
    "${SLANG_DIR}"
    "$ENV{SLANG_DIR}"
    "/usr/local"
    "/usr"
    "/opt/slang"
)

# Find include directory
find_path(Slang_INCLUDE_DIR
    NAMES slang.h
    PATHS ${_slang_search_paths}
    PATH_SUFFIXES include
    DOC "Slang include directory"
)

# Find library
find_library(Slang_LIBRARY
    NAMES slang libslang
    PATHS ${_slang_search_paths}
    PATH_SUFFIXES lib lib64 bin
    DOC "Slang library"
)

# Handle standard find_package arguments
find_package_handle_standard_args(Slang
    REQUIRED_VARS Slang_LIBRARY Slang_INCLUDE_DIR
    FAIL_MESSAGE "Could not find Slang. Set SLANG_ROOT to the Slang installation directory."
)

# Create imported target
if(Slang_FOUND AND NOT TARGET Slang::Slang)
    add_library(Slang::Slang UNKNOWN IMPORTED)
    set_target_properties(Slang::Slang PROPERTIES
        IMPORTED_LOCATION "${Slang_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Slang_INCLUDE_DIR}"
    )

    # Mark as advanced
    mark_as_advanced(Slang_INCLUDE_DIR Slang_LIBRARY)

    message(STATUS "Found Slang: ${Slang_LIBRARY}")
    message(STATUS "  Include: ${Slang_INCLUDE_DIR}")
endif()
