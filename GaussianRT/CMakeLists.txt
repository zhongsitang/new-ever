# =============================================================================
# GaussianRT - Clean slang-rhi Ray Tracing Implementation
# =============================================================================
# A modern, well-structured implementation of ellipsoid volume rendering
# using slang-rhi hardware ray tracing.
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#
# Test:
#   ./bin/gaussianrt_test
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Check for CUDA before declaring project
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(gaussianrt
        VERSION 1.0.0
        DESCRIPTION "Modern slang-rhi ray tracing for ellipsoid volume rendering"
        LANGUAGES C CXX CUDA
    )
    set(HAVE_CUDA TRUE)
else()
    project(gaussianrt
        VERSION 1.0.0
        DESCRIPTION "Modern slang-rhi ray tracing for ellipsoid volume rendering"
        LANGUAGES C CXX
    )
    set(HAVE_CUDA FALSE)
    message(WARNING "CUDA not found. Building without GPU support (compile-only mode).")
endif()

# -----------------------------------------------------------------------------
# Build Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------------
# Find Dependencies
# -----------------------------------------------------------------------------
if(HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Use parent project's cmake modules
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# -----------------------------------------------------------------------------
# Slang and slang-rhi (Stub mode without actual slang for compile testing)
# -----------------------------------------------------------------------------
# For full functionality, this needs slang and slang-rhi.
# In compile-only mode, we create stub targets.

option(USE_SLANG_STUBS "Use stub implementations instead of real slang-rhi" OFF)

if(NOT TARGET slang-rhi OR USE_SLANG_STUBS)
    # Check if we can use the parent slang-master
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../slang-master/CMakeLists.txt" AND NOT USE_SLANG_STUBS)
        message(STATUS "Using slang from parent directory")

        include(FetchContent)
        set(SLANG_ENABLE_SLANG_RHI ON CACHE BOOL "" FORCE)
        set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            slang_rhi
            GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
            GIT_TAG v0.19.0
        )
        FetchContent_GetProperties(slang_rhi)
        if(NOT slang_rhi_POPULATED)
            FetchContent_Populate(slang_rhi)
        endif()

        set(SLANG_OVERRIDE_SLANG_RHI_PATH "${slang_rhi_SOURCE_DIR}" CACHE PATH "" FORCE)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../slang-master" slang EXCLUDE_FROM_ALL)
    else()
        # Create stub interface libraries for compile testing
        message(STATUS "Creating stub slang libraries for compile testing")
        set(USE_SLANG_STUBS ON)

        # Create stub header directory
        set(STUB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/stubs/include")
        file(MAKE_DIRECTORY "${STUB_INCLUDE_DIR}")

        # Create minimal stub headers
        file(WRITE "${STUB_INCLUDE_DIR}/slang-com-ptr.h" [[
#pragma once
namespace Slang {
    template<typename T> class ComPtr {
    public:
        ComPtr() : ptr_(nullptr) {}
        ComPtr(T* p) : ptr_(p) {}
        T* get() const { return ptr_; }
        T** writeRef() { return &ptr_; }
        T* operator->() const { return ptr_; }
        operator bool() const { return ptr_ != nullptr; }
    private:
        T* ptr_;
    };
    using Result = long;
    template<typename T> struct List {
        void add(T) {}
        T* getBuffer() { return nullptr; }
        int getCount() { return 0; }
    };
}
#define SLANG_FAILED(x) ((x) < 0)
#define SLANG_RETURN_ON_FAIL(x) { auto _r = (x); if (SLANG_FAILED(_r)) return _r; }
#define SLANG_OK 0
#define SLANG_FAIL -1
]])

        file(WRITE "${STUB_INCLUDE_DIR}/slang-rhi.h" [[
#pragma once
#include <cstdint>
#include <cstddef>
namespace rhi {
    using Size = size_t;
    enum class DeviceType { CUDA };
    enum class QueueType { Graphics };
    enum class BufferUsage : uint32_t {
        None = 0,
        ShaderResource = 1,
        UnorderedAccess = 2,
        AccelerationStructureBuildInput = 4
    };
    inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
        return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    inline BufferUsage operator&(BufferUsage a, BufferUsage b) {
        return static_cast<BufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }
    enum class ResourceState { ShaderResource, UnorderedAccess, AccelerationStructureBuildInput };
    enum class AccelerationStructureBuildInputType { Triangles, Instances, ProceduralPrimitives };
    enum class AccelerationStructureBuildFlags : uint32_t {
        None = 0, AllowCompaction = 1, PreferFastBuild = 2, PreferFastTrace = 4
    };
    inline AccelerationStructureBuildFlags operator|(AccelerationStructureBuildFlags a, AccelerationStructureBuildFlags b) {
        return static_cast<AccelerationStructureBuildFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    enum class AccelerationStructureGeometryFlags { None, Opaque };
    enum class AccelerationStructureCopyMode { Compact };
    enum class AccelerationStructureInstanceFlags { TriangleFacingCullDisable };
    enum class AccelerationStructureInstanceDescType { Generic };
    enum class QueryType { AccelerationStructureCompactedSize };

    struct DeviceDesc { DeviceType deviceType; bool enableValidation; int adapterIndex; };
    struct BufferDesc { size_t size; size_t elementSize; BufferUsage usage; ResourceState defaultState; };
    struct ShaderProgramDesc { void* slangGlobalScope; };
    struct HitGroupDesc { const char* hitGroupName; const char* anyHitEntryPoint; const char* intersectionEntryPoint; const char* closestHitEntryPoint; };
    struct RayTracingPipelineDesc { void* program; int hitGroupCount; HitGroupDesc* hitGroups; size_t maxRayPayloadSize; int maxRecursion; };
    struct ShaderTableDesc { void* program; int hitGroupCount; const char** hitGroupNames; int rayGenShaderCount; const char** rayGenShaderEntryPointNames; int missShaderCount; const char** missShaderEntryPointNames; };
    struct AccelerationStructureDesc { size_t size; };
    struct AccelerationStructureSizes { size_t accelerationStructureSize; size_t scratchSize; };
    struct QueryPoolDesc { int count; QueryType type; };
    struct AccelerationStructureQueryDesc { void* queryPool; QueryType queryType; };

    struct BufferOffsetPair {
        void* buffer; size_t offset;
        BufferOffsetPair(void* b, size_t o) : buffer(b), offset(o) {}
    };

    struct ProceduralPrimitivesInput {
        BufferOffsetPair aabbBuffer = {nullptr, 0};
        uint32_t primitiveCount;
        size_t aabbStride;
        AccelerationStructureGeometryFlags flags;
    };
    struct InstancesInput {
        BufferOffsetPair instanceBuffer = {nullptr, 0};
        uint32_t instanceCount;
        size_t instanceStride;
    };
    struct AccelerationStructureBuildInput {
        AccelerationStructureBuildInputType type;
        ProceduralPrimitivesInput proceduralPrimitives;
        InstancesInput instances;
    };
    struct AccelerationStructureBuildDesc {
        AccelerationStructureBuildInput* inputs;
        int inputCount;
        AccelerationStructureBuildFlags flags;
    };
    struct AccelerationStructureInstanceDescGeneric {
        float transform[3][4];
        uint32_t instanceID;
        uint32_t instanceMask;
        uint32_t instanceContributionToHitGroupIndex;
        AccelerationStructureInstanceFlags flags;
        uint64_t accelerationStructure;
    };

    struct IBuffer { virtual ~IBuffer() = default; };
    struct IShaderProgram { virtual ~IShaderProgram() = default; };
    struct IRayTracingPipeline { virtual ~IRayTracingPipeline() = default; };
    struct IShaderTable { virtual ~IShaderTable() = default; };
    struct IQueryPool { virtual ~IQueryPool() = default; virtual void reset() {} virtual void getResult(int, int, uint64_t*) {} };
    struct IAccelerationStructure { virtual ~IAccelerationStructure() = default; virtual uint64_t getHandle() { return 0; } };
    struct ICommandEncoder { virtual ~ICommandEncoder() = default; virtual void* finish() { return nullptr; } virtual void buildAccelerationStructure(AccelerationStructureBuildDesc&, void*, void*, BufferOffsetPair, int, AccelerationStructureQueryDesc*) {} virtual void copyAccelerationStructure(void*, void*, AccelerationStructureCopyMode) {} };
    struct IRayTracingPassEncoder { virtual ~IRayTracingPassEncoder() = default; virtual void* bindPipeline(void*, void*) { return nullptr; } virtual void dispatchRays(int, uint32_t, uint32_t, uint32_t) {} virtual void end() {} };
    struct ICommandQueue {
        virtual ~ICommandQueue() = default;
        virtual ICommandEncoder* createCommandEncoder() { return nullptr; }
        virtual void submit(void*) {}
        virtual void waitOnHost() {}
    };
    struct IDevice {
        virtual ~IDevice() = default;
        virtual ICommandQueue* getQueue(QueueType) { return nullptr; }
        virtual void* getSlangSession() { return nullptr; }
        virtual IBuffer* createBuffer(BufferDesc&, const void* = nullptr) { return nullptr; }
        virtual long getAccelerationStructureSizes(AccelerationStructureBuildDesc&, AccelerationStructureSizes*) { return 0; }
        virtual long createAccelerationStructure(AccelerationStructureDesc&, IAccelerationStructure**) { return 0; }
        virtual long createQueryPool(QueryPoolDesc&, IQueryPool**) { return 0; }
        virtual long createShaderProgram(ShaderProgramDesc&, IShaderProgram**) { return 0; }
        virtual long createRayTracingPipeline(RayTracingPipelineDesc&, IRayTracingPipeline**) { return 0; }
        virtual long createShaderTable(ShaderTableDesc&, IShaderTable**) { return 0; }
    };
    struct IRHI {
        virtual ~IRHI() = default;
        virtual IDevice* createDevice(DeviceDesc&) { return nullptr; }
    };
    inline IRHI* getRHI() { static IRHI rhi; return &rhi; }
    inline AccelerationStructureInstanceDescType getAccelerationStructureInstanceDescType(IDevice*) { return AccelerationStructureInstanceDescType::Generic; }
    inline Size getAccelerationStructureInstanceDescSize(AccelerationStructureInstanceDescType) { return sizeof(AccelerationStructureInstanceDescGeneric); }
    inline void convertAccelerationStructureInstanceDescs(size_t, AccelerationStructureInstanceDescType, void*, size_t, void*, size_t) {}
}
namespace slang {
    struct ISession { virtual ~ISession() = default; virtual void* loadModule(const char*, void**) { return nullptr; } virtual long createCompositeComponentType(void**, int, void**, void**) { return 0; } };
    struct IModule { virtual ~IModule() = default; virtual long findEntryPointByName(const char*, void**) { return 0; } };
    struct IEntryPoint { virtual ~IEntryPoint() = default; };
    struct IComponentType { virtual ~IComponentType() = default; };
    struct IBlob { virtual ~IBlob() = default; virtual const void* getBufferPointer() { return nullptr; } };
}
]])

        file(WRITE "${STUB_INCLUDE_DIR}/slang-rhi/acceleration-structure-utils.h" [[
#pragma once
#include "slang-rhi.h"
]])

        file(WRITE "${STUB_INCLUDE_DIR}/slang-rhi/shader-cursor.h" [[
#pragma once
namespace rhi {
    struct ShaderCursor {
        ShaderCursor(void*) {}
        ShaderCursor operator[](const char*) { return *this; }
        void setBinding(void*) {}
        void setData(const void*, size_t) {}
    };
}
]])

        file(WRITE "${STUB_INCLUDE_DIR}/slang.h" [[
#pragma once
#include "slang-rhi.h"
typedef long SlangResult;
]])

        # Create interface libraries
        add_library(slang INTERFACE)
        target_include_directories(slang INTERFACE "${STUB_INCLUDE_DIR}")

        add_library(slang-rhi INTERFACE)
        target_include_directories(slang-rhi INTERFACE "${STUB_INCLUDE_DIR}")
    endif()
endif()

# -----------------------------------------------------------------------------
# Source Files
# -----------------------------------------------------------------------------
set(GAUSSIANRT_HEADERS
    include/Types.h
    include/Device.h
    include/AccelerationStructure.h
    include/RayTracer.h
)

set(GAUSSIANRT_SOURCES
    src/Device.cpp
    src/AccelerationStructure.cpp
    src/RayTracer.cpp
)

set(GAUSSIANRT_SHADERS
    shaders/safe_math.slang
    shaders/sh.slang
    shaders/volume.slang
    shaders/ellipsoid.slang
    shaders/raytracer.slang
)

# -----------------------------------------------------------------------------
# Main Library
# -----------------------------------------------------------------------------
add_library(gaussianrt STATIC
    ${GAUSSIANRT_HEADERS}
    ${GAUSSIANRT_SOURCES}
)

target_include_directories(gaussianrt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gaussianrt
    PUBLIC
        slang
        slang-rhi
)

if(HAVE_CUDA)
    target_link_libraries(gaussianrt PUBLIC CUDA::cudart CUDA::cuda_driver)
else()
    # Define stubs for CUDA functions when CUDA is not available
    target_compile_definitions(gaussianrt PUBLIC GAUSSIANRT_NO_CUDA)
endif()

# Copy shaders to build directory
add_custom_command(TARGET gaussianrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${CMAKE_BINARY_DIR}/shaders"
    COMMENT "Copying shaders to build directory"
)

# -----------------------------------------------------------------------------
# Test Executable
# -----------------------------------------------------------------------------
add_executable(gaussianrt_test
    test/main.cpp
)

target_link_libraries(gaussianrt_test
    PRIVATE
        gaussianrt
)

# Ensure shaders are available for tests
add_dependencies(gaussianrt_test gaussianrt)

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(gaussianrt PRIVATE /W4)
    target_compile_options(gaussianrt_test PRIVATE /W4)
else()
    target_compile_options(gaussianrt PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_options(gaussianrt_test PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# CUDA specific flags
if(HAVE_CUDA AND CMAKE_CUDA_COMPILER)
    target_compile_options(gaussianrt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS gaussianrt
    EXPORT gaussianrt-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/gaussianrt
)

install(DIRECTORY shaders/
    DESTINATION share/gaussianrt/shaders
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "GaussianRT Configuration")
message(STATUS "========================================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
if(HAVE_CUDA)
    message(STATUS "  CUDA:           ${CUDAToolkit_VERSION}")
else()
    message(STATUS "  CUDA:           Not available (stub mode)")
endif()
message(STATUS "  Slang stubs:    ${USE_SLANG_STUBS}")
message(STATUS "  Output dir:     ${CMAKE_BINARY_DIR}")
message(STATUS "========================================")
