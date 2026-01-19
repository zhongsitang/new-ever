# =============================================================================
# FindTorch.cmake - Find PyTorch installation
# =============================================================================
# This module locates PyTorch installation and creates Torch::Torch target.
# =============================================================================

# Include guard to prevent multiple inclusion
# Only return if target already exists (from previous include)
if(TARGET Torch::Torch)
    set(Torch_FOUND TRUE)
    return()
endif()

# -----------------------------------------------------------------------------
# 1. Dependencies: CUDA and Python (REQUIRED)
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# -----------------------------------------------------------------------------
# 2. Locate PyTorch Installation
# -----------------------------------------------------------------------------
if(NOT Torch_ROOT)
  execute_process(
    COMMAND ${Python_EXECUTABLE}
    -c "import os, torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE _Torch_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Torch_ROOT ${_Torch_ROOT} CACHE PATH "Path to the PyTorch installation root" FORCE)
endif()

if(NOT TORCH_CXX11_ABI)
  execute_process(
    COMMAND "${Python_EXECUTABLE}"
    -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE _TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(TORCH_CXX11_ABI ${_TORCH_CXX11_ABI} CACHE STRING "The value of torch._C._GLIBCXX_USE_CXX11_ABI" FORCE)
endif()

message(STATUS "Torch found at: ${Torch_ROOT}")
message(STATUS "Torch CXX11 ABI: ${TORCH_CXX11_ABI}")

# -----------------------------------------------------------------------------
# 3. Extract PyBind11 Definitions from PyTorch
# -----------------------------------------------------------------------------
if(NOT PYBIND11_DEFINES)
  foreach(name COMPILER_TYPE STDLIB BUILD_ABI)
    execute_process(
      COMMAND ${Python_EXECUTABLE}
      -c "import torch; print(getattr(torch._C,'_PYBIND11_'+'${name}',''))"
      OUTPUT_VARIABLE value
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(value)
      list(APPEND _PYBIND11_DEFINES "-DPYBIND11_${name}=\"${value}\"")
    endif()
  endforeach()
  set(PYBIND11_DEFINES ${_PYBIND11_DEFINES} CACHE STRING "List of -DPYBIND11_* definitions from PyTorch" FORCE)
endif()

message(STATUS "Torch pybind11 defines: ${PYBIND11_DEFINES}")

# -----------------------------------------------------------------------------
# 4. Locate PyTorch Include Directories and Libraries
# -----------------------------------------------------------------------------

# Get include paths from torch.utils.cpp_extension (primary source)
execute_process(
    COMMAND ${Python_EXECUTABLE}
    -c "import torch.utils.cpp_extension; import os; paths = torch.utils.cpp_extension.include_paths(False); valid = [p for p in paths if os.path.exists(p)]; print(';'.join(valid), end='')"
    OUTPUT_VARIABLE _torch_ext_include_dirs
    RESULT_VARIABLE _result
    ERROR_QUIET
)
if(_result EQUAL 0 AND _torch_ext_include_dirs)
    # Split by semicolon (Windows) or newline
    string(REPLACE ";" " " _torch_ext_include_dirs "${_torch_ext_include_dirs}")
    string(REPLACE "\n" " " _torch_ext_include_dirs "${_torch_ext_include_dirs}")
    string(REPLACE " " ";" _torch_ext_include_dirs "${_torch_ext_include_dirs}")
    foreach(_dir IN LISTS _torch_ext_include_dirs)
        string(STRIP "${_dir}" _dir)
        if(_dir AND EXISTS "${_dir}" AND NOT "${_dir}" IN_LIST TORCH_INCLUDE_DIRS)
            list(APPEND TORCH_INCLUDE_DIRS "${_dir}")
        endif()
    endforeach()
endif()

if(NOT TORCH_INCLUDE_DIRS)
    message(FATAL_ERROR "No Torch include directories found. Please check your PyTorch installation.")
endif()

# Find library path - prefer torch.utils.cpp_extension
execute_process(
    COMMAND ${Python_EXECUTABLE}
    -c "import torch.utils.cpp_extension; import os; paths = torch.utils.cpp_extension.library_paths(False); valid = [p for p in paths if os.path.exists(p)]; print(valid[0] if valid else '', end='')"
    OUTPUT_VARIABLE _torch_lib_path_from_ext
    RESULT_VARIABLE _result
    ERROR_QUIET
)
if(_result EQUAL 0 AND _torch_lib_path_from_ext)
    string(STRIP "${_torch_lib_path_from_ext}" _torch_lib_path_from_ext)
    if(_torch_lib_path_from_ext AND EXISTS "${_torch_lib_path_from_ext}")
        set(TORCH_LIB_PATH "${_torch_lib_path_from_ext}")
    else()
        set(TORCH_LIB_PATH "${Torch_ROOT}/lib")
    endif()
else()
    set(TORCH_LIB_PATH "${Torch_ROOT}/lib")
endif()

# Library names - use standard PyTorch library names
set(TORCH_LIB_NAMES
    c10
    torch
    torch_cpu
    torch_python
    c10_cuda
    torch_cuda
)
if(WIN32)
    list(APPEND TORCH_LIB_NAMES sleef)
endif()

# Find all libraries
set(TORCH_LIBRARIES)
foreach(name IN LISTS TORCH_LIB_NAMES)
  find_library(lib_${name}
    NAMES ${name}
    HINTS ${TORCH_LIB_PATH}
    REQUIRED
  )
  list(APPEND TORCH_LIBRARIES ${lib_${name}})
endforeach()

# Ensure we have at least some libraries
if(NOT TORCH_LIBRARIES)
    message(FATAL_ERROR "No Torch libraries found. Please check your PyTorch installation.")
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

# -----------------------------------------------------------------------------
# 5. Common Compiler, CUDA Flags and Definitions
# -----------------------------------------------------------------------------
set(COMMON_MSVC_FLAGS
    /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068
)

set(MSVC_IGNORE_CUDAFE_WARNINGS
    --diag-suppress 1388,1390,1394
)

set(COMMON_NVCC_FLAGS
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    --expt-relaxed-constexpr
)

set(COMMON_DEFINES
    TORCH_API_INCLUDE_EXTENSION_H
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)
if(UNIX)
    list(APPEND COMMON_DEFINES ${PYBIND11_DEFINES})
endif()

# -----------------------------------------------------------------------------
# 6. CUDA Compiler Flags for Different Platforms
# -----------------------------------------------------------------------------
string(REPLACE ";" "," WIN_CUDA_CXX_FLAGS "-Xcompiler=${COMMON_MSVC_FLAGS}")

set(WIN_CUDA_FLAGS
  --use-local-env
  ${WIN_CUDA_CXX_FLAGS}
  ${MSVC_IGNORE_CUDAFE_WARNINGS}
  ${COMMON_NVCC_FLAGS}
)
set(WIN_LINK_FLAGS
  -INCLUDE:?warp_size@cuda@at@@YAHXZ
)

set(UNIX_CUDA_FLAGS
  ${COMMON_NVCC_FLAGS}
)
set(UNIX_LINK_FLAGS
  -Wl,-rpath,${TORCH_LIB_PATH}
)

# -----------------------------------------------------------------------------
# 7. Configure Torch Target (INTERFACE Library)
# -----------------------------------------------------------------------------
add_library(Torch::Torch INTERFACE IMPORTED)

set_target_properties(Torch::Torch PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS};${CUDAToolkit_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
)

target_compile_options(Torch::Torch INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${COMMON_MSVC_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:${WIN_CUDA_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:${UNIX_CUDA_FLAGS}>
)

target_compile_definitions(Torch::Torch INTERFACE ${COMMON_DEFINES})

target_link_libraries(Torch::Torch INTERFACE CUDA::cudart Python::Python)
target_link_options(Torch::Torch INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:${WIN_LINK_FLAGS}>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${UNIX_LINK_FLAGS}>
)

# Mark as found
set(Torch_FOUND TRUE CACHE INTERNAL "Torch found" FORCE)
