#include "gaussian_rt/AccelStruct.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

namespace gaussian_rt {

//------------------------------------------------------------------------------
// OptiX error checking
//------------------------------------------------------------------------------

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            fprintf(stderr, "OptiX error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    optixGetErrorString(res));                                 \
            return Result::ErrorAccelStructBuild;                              \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            return Result::ErrorCUDA;                                          \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
// Static OptiX context management
//------------------------------------------------------------------------------

static OptixDeviceContext g_optixContext = nullptr;
static bool g_optixInitialized = false;

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
    fprintf(stderr, "[OptiX %u][%s]: %s\n", level, tag, message);
}

static Result ensureOptixInitialized(int cudaDeviceId) {
    if (g_optixInitialized) {
        return Result::Success;
    }

    // Initialize CUDA
    cudaError_t cudaErr = cudaSetDevice(cudaDeviceId);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device\n");
        return Result::ErrorCUDA;
    }

    cudaErr = cudaFree(0);  // Force context creation
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return Result::ErrorCUDA;
    }

    // Initialize OptiX
    OptixResult optixRes = optixInit();
    if (optixRes != OPTIX_SUCCESS) {
        fprintf(stderr, "Failed to initialize OptiX: %s\n", optixGetErrorString(optixRes));
        return Result::ErrorAccelStructBuild;
    }

    // Get CUDA context
    CUcontext cuCtx = nullptr;
    CUresult cuRes = cuCtxGetCurrent(&cuCtx);
    if (cuRes != CUDA_SUCCESS || cuCtx == nullptr) {
        fprintf(stderr, "Failed to get CUDA context\n");
        return Result::ErrorCUDA;
    }

    // Create OptiX context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;

    optixRes = optixDeviceContextCreate(cuCtx, &options, &g_optixContext);
    if (optixRes != OPTIX_SUCCESS) {
        fprintf(stderr, "Failed to create OptiX context: %s\n", optixGetErrorString(optixRes));
        return Result::ErrorAccelStructBuild;
    }

    g_optixInitialized = true;
    return Result::Success;
}

//------------------------------------------------------------------------------
// AccelStruct implementation
//------------------------------------------------------------------------------

AccelStruct::AccelStruct(Device& device)
    : m_device(device) {
}

AccelStruct::~AccelStruct() {
    freeBuffers();
}

AccelStruct::AccelStruct(AccelStruct&& other) noexcept
    : m_device(other.m_device)
    , m_built(other.m_built)
    , m_allowUpdate(other.m_allowUpdate)
    , m_numPrimitives(other.m_numPrimitives)
    , m_blasBuffer(other.m_blasBuffer)
    , m_tlasBuffer(other.m_tlasBuffer)
    , m_instanceBuffer(other.m_instanceBuffer)
    , m_scratchBuffer(other.m_scratchBuffer)
    , m_traversableHandle(other.m_traversableHandle)
    , m_blasSize(other.m_blasSize)
    , m_tlasSize(other.m_tlasSize)
    , m_scratchSize(other.m_scratchSize) {
    other.m_blasBuffer = nullptr;
    other.m_tlasBuffer = nullptr;
    other.m_instanceBuffer = nullptr;
    other.m_scratchBuffer = nullptr;
    other.m_traversableHandle = nullptr;
    other.m_built = false;
}

AccelStruct& AccelStruct::operator=(AccelStruct&& other) noexcept {
    if (this != &other) {
        freeBuffers();

        m_built = other.m_built;
        m_allowUpdate = other.m_allowUpdate;
        m_numPrimitives = other.m_numPrimitives;
        m_blasBuffer = other.m_blasBuffer;
        m_tlasBuffer = other.m_tlasBuffer;
        m_instanceBuffer = other.m_instanceBuffer;
        m_scratchBuffer = other.m_scratchBuffer;
        m_traversableHandle = other.m_traversableHandle;
        m_blasSize = other.m_blasSize;
        m_tlasSize = other.m_tlasSize;
        m_scratchSize = other.m_scratchSize;

        other.m_blasBuffer = nullptr;
        other.m_tlasBuffer = nullptr;
        other.m_instanceBuffer = nullptr;
        other.m_scratchBuffer = nullptr;
        other.m_traversableHandle = nullptr;
        other.m_built = false;
    }
    return *this;
}

void AccelStruct::freeBuffers() {
    if (m_blasBuffer) m_device.freeBuffer(m_blasBuffer);
    if (m_tlasBuffer) m_device.freeBuffer(m_tlasBuffer);
    if (m_instanceBuffer) m_device.freeBuffer(m_instanceBuffer);
    if (m_scratchBuffer) m_device.freeBuffer(m_scratchBuffer);

    m_blasBuffer = nullptr;
    m_tlasBuffer = nullptr;
    m_instanceBuffer = nullptr;
    m_scratchBuffer = nullptr;
    m_traversableHandle = nullptr;
    m_built = false;
}

Result AccelStruct::build(const PrimitiveSet& primitives, bool allowUpdate, bool fastBuild) {
    if (!primitives.isValid()) {
        return Result::ErrorInvalidArgument;
    }

    // Ensure OptiX is initialized
    Result result = ensureOptixInitialized(m_device.getCudaDeviceId());
    if (result != Result::Success) {
        return result;
    }

    freeBuffers();

    m_numPrimitives = primitives.getNumPrimitives();
    m_allowUpdate = allowUpdate;

    // Build BLAS
    result = buildBLAS(primitives, fastBuild);
    if (result != Result::Success) {
        return result;
    }

    // Build TLAS
    result = buildTLAS(fastBuild);
    if (result != Result::Success) {
        return result;
    }

    m_built = true;
    return Result::Success;
}

Result AccelStruct::buildBLAS(const PrimitiveSet& primitives, bool fastBuild) {
    // Setup AABB build input
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    auto& customInput = buildInput.customPrimitiveArray;

    // AABB data is stored as [min0, min1, min2, ..., max0, max1, max2, ...]
    // Need to interleave for OptiX format
    CUdeviceptr d_aabbBuffer = reinterpret_cast<CUdeviceptr>(primitives.getAABBsDevice());

    customInput.aabbBuffers = &d_aabbBuffer;
    customInput.numPrimitives = static_cast<unsigned int>(m_numPrimitives);
    customInput.strideInBytes = 0;  // Tightly packed
    customInput.primitiveIndexOffset = 0;

    // Build flags
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    customInput.flags = &flags;
    customInput.numSbtRecords = 1;

    // Build options
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    if (m_allowUpdate) {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    if (fastBuild) {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    } else {
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        g_optixContext,
        &accelOptions,
        &buildInput,
        1,
        &bufferSizes
    ));

    // Allocate buffers
    m_scratchSize = bufferSizes.tempSizeInBytes;
    m_blasSize = bufferSizes.outputSizeInBytes;

    m_scratchBuffer = m_device.createBuffer(m_scratchSize);
    if (!m_scratchBuffer) return Result::ErrorOutOfMemory;

    m_blasBuffer = m_device.createBuffer(m_blasSize);
    if (!m_blasBuffer) return Result::ErrorOutOfMemory;

    // Compaction buffer
    void* d_compactedSize = m_device.createBuffer(sizeof(size_t));
    if (!d_compactedSize) return Result::ErrorOutOfMemory;

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = reinterpret_cast<CUdeviceptr>(d_compactedSize);

    // Build acceleration structure
    OptixTraversableHandle blasHandle;
    OPTIX_CHECK(optixAccelBuild(
        g_optixContext,
        static_cast<CUstream>(m_device.getCudaStream()),
        &accelOptions,
        &buildInput,
        1,
        reinterpret_cast<CUdeviceptr>(m_scratchBuffer),
        m_scratchSize,
        reinterpret_cast<CUdeviceptr>(m_blasBuffer),
        m_blasSize,
        &blasHandle,
        &emitDesc,
        1
    ));

    // Synchronize and get compacted size
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(m_device.getCudaStream())));

    size_t compactedSize;
    CUDA_CHECK(cudaMemcpy(&compactedSize, d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost));
    m_device.freeBuffer(d_compactedSize);

    // Compact if beneficial
    if (compactedSize < m_blasSize) {
        void* compactedBuffer = m_device.createBuffer(compactedSize);
        if (compactedBuffer) {
            OPTIX_CHECK(optixAccelCompact(
                g_optixContext,
                static_cast<CUstream>(m_device.getCudaStream()),
                blasHandle,
                reinterpret_cast<CUdeviceptr>(compactedBuffer),
                compactedSize,
                &blasHandle
            ));

            m_device.freeBuffer(m_blasBuffer);
            m_blasBuffer = compactedBuffer;
            m_blasSize = compactedSize;
        }
    }

    m_traversableHandle = reinterpret_cast<void*>(blasHandle);
    return Result::Success;
}

Result AccelStruct::buildTLAS(bool fastBuild) {
    // For single BLAS, we can use the BLAS handle directly
    // If we need instance transforms, we'd build a TLAS here

    // For now, the traversable handle is the BLAS handle
    // In full implementation, we'd create instances and build TLAS
    return Result::Success;
}

Result AccelStruct::update(const PrimitiveSet& primitives) {
    if (!m_built || !m_allowUpdate) {
        return Result::ErrorInvalidArgument;
    }

    if (primitives.getNumPrimitives() != m_numPrimitives) {
        // Topology changed, need full rebuild
        return rebuild(primitives);
    }

    // Refit BLAS
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    auto& customInput = buildInput.customPrimitiveArray;
    CUdeviceptr d_aabbBuffer = reinterpret_cast<CUdeviceptr>(primitives.getAABBsDevice());

    customInput.aabbBuffers = &d_aabbBuffer;
    customInput.numPrimitives = static_cast<unsigned int>(m_numPrimitives);
    customInput.strideInBytes = 0;
    customInput.primitiveIndexOffset = 0;

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    customInput.flags = &flags;
    customInput.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixTraversableHandle blasHandle = reinterpret_cast<OptixTraversableHandle>(m_traversableHandle);

    OPTIX_CHECK(optixAccelBuild(
        g_optixContext,
        static_cast<CUstream>(m_device.getCudaStream()),
        &accelOptions,
        &buildInput,
        1,
        reinterpret_cast<CUdeviceptr>(m_scratchBuffer),
        m_scratchSize,
        reinterpret_cast<CUdeviceptr>(m_blasBuffer),
        m_blasSize,
        &blasHandle,
        nullptr,
        0
    ));

    m_traversableHandle = reinterpret_cast<void*>(blasHandle);
    return Result::Success;
}

Result AccelStruct::rebuild(const PrimitiveSet& primitives) {
    return build(primitives, m_allowUpdate, false);
}

} // namespace gaussian_rt
