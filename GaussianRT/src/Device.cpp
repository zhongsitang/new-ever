#include "gaussian_rt/Device.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <memory>
#include <mutex>

// Try to include Slang-RHI if available
#if __has_include(<slang-rhi.h>)
#include <slang-rhi.h>
#include <slang.h>
#define GAUSSIANRT_HAS_SLANG_RHI 1
#else
#define GAUSSIANRT_HAS_SLANG_RHI 0
#endif

namespace gaussian_rt {

//------------------------------------------------------------------------------
// CUDA error checking
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            return Result::ErrorCUDA;                                          \
        }                                                                      \
    } while (0)

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult err = call;                                                   \
        if (err != CUDA_SUCCESS) {                                             \
            const char* errStr;                                                \
            cuGetErrorString(err, &errStr);                                    \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__,      \
                    __LINE__, errStr ? errStr : "Unknown");                    \
            return Result::ErrorCUDA;                                          \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
// Device::Impl
//------------------------------------------------------------------------------

struct Device::Impl {
#if GAUSSIANRT_HAS_SLANG_RHI
    Slang::ComPtr<rhi::IDevice> rhiDevice;
    Slang::ComPtr<rhi::ICommandQueue> commandQueue;
    Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;
    Slang::ComPtr<slang::ISession> slangSession;
#endif

    CUcontext cudaContext = nullptr;
    cudaStream_t cudaStream = nullptr;
};

//------------------------------------------------------------------------------
// Device implementation
//------------------------------------------------------------------------------

Device::Device()
    : m_impl(std::make_unique<Impl>()) {
}

Device::~Device() {
    shutdown();
}

Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

Result Device::initialize(int cudaDeviceId, bool enableValidation) {
    if (m_initialized) {
        return Result::Success;
    }

    m_cudaDeviceId = cudaDeviceId;

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(cudaDeviceId));
    CUDA_CHECK(cudaFree(0));  // Force context creation

    // Get CUDA context
    CU_CHECK(cuCtxGetCurrent(&m_impl->cudaContext));
    m_cudaContext = m_impl->cudaContext;

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&m_impl->cudaStream));
    m_cudaStream = m_impl->cudaStream;

#if GAUSSIANRT_HAS_SLANG_RHI
    // Initialize Slang global session
    slang::createGlobalSession(m_impl->slangGlobalSession.writeRef());
    if (!m_impl->slangGlobalSession) {
        fprintf(stderr, "Failed to create Slang global session\n");
        return Result::ErrorShaderCompilation;
    }
    m_slangGlobalSession = m_impl->slangGlobalSession.get();

    // Create RHI device with CUDA backend
    rhi::DeviceDesc deviceDesc = {};
    deviceDesc.deviceType = rhi::DeviceType::CUDA;
    deviceDesc.slang.slangGlobalSession = m_impl->slangGlobalSession.get();

    if (enableValidation) {
        deviceDesc.enableValidation = true;
    }

    // Try to create CUDA device
    rhi::Result rhiResult = rhi::createDevice(deviceDesc, m_impl->rhiDevice.writeRef());

    if (SLANG_FAILED(rhiResult) || !m_impl->rhiDevice) {
        // Fallback: try Vulkan with ray tracing
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        rhiResult = rhi::createDevice(deviceDesc, m_impl->rhiDevice.writeRef());
    }

    if (SLANG_FAILED(rhiResult) || !m_impl->rhiDevice) {
        fprintf(stderr, "Failed to create RHI device\n");
        return Result::ErrorDeviceNotInitialized;
    }

    m_rhiDevice = m_impl->rhiDevice.get();

    // Create command queue
    rhi::CommandQueueDesc queueDesc = {};
    queueDesc.type = rhi::QueueType::Graphics;
    rhiResult = m_impl->rhiDevice->createCommandQueue(queueDesc, m_impl->commandQueue.writeRef());

    if (SLANG_FAILED(rhiResult) || !m_impl->commandQueue) {
        fprintf(stderr, "Failed to create command queue\n");
        return Result::ErrorDeviceNotInitialized;
    }

    m_commandQueue = m_impl->commandQueue.get();

    // Create Slang session
    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = m_impl->slangGlobalSession->findProfile("spirv_1_5");

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    m_impl->slangGlobalSession->createSession(sessionDesc, m_impl->slangSession.writeRef());
    m_slangSession = m_impl->slangSession.get();
#endif

    m_initialized = true;
    return Result::Success;
}

void Device::shutdown() {
    if (!m_initialized) {
        return;
    }

#if GAUSSIANRT_HAS_SLANG_RHI
    m_impl->slangSession = nullptr;
    m_impl->commandQueue = nullptr;
    m_impl->rhiDevice = nullptr;
    m_impl->slangGlobalSession = nullptr;
#endif

    if (m_impl->cudaStream) {
        cudaStreamDestroy(m_impl->cudaStream);
        m_impl->cudaStream = nullptr;
    }

    m_rhiDevice = nullptr;
    m_commandQueue = nullptr;
    m_slangSession = nullptr;
    m_slangGlobalSession = nullptr;
    m_cudaStream = nullptr;
    m_cudaContext = nullptr;
    m_initialized = false;
}

void Device::synchronize() {
    if (m_impl->cudaStream) {
        cudaStreamSynchronize(m_impl->cudaStream);
    }

#if GAUSSIANRT_HAS_SLANG_RHI
    if (m_impl->commandQueue) {
        m_impl->commandQueue->waitOnHost();
    }
#endif
}

//------------------------------------------------------------------------------
// Buffer operations
//------------------------------------------------------------------------------

void* Device::createBuffer(size_t size, const void* initialData) {
    if (!m_initialized || size == 0) {
        return nullptr;
    }

    void* devicePtr = nullptr;
    cudaError_t err = cudaMalloc(&devicePtr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }

    if (initialData) {
        cudaMemcpy(devicePtr, initialData, size, cudaMemcpyHostToDevice);
    }

    return devicePtr;
}

void* Device::createBufferWithUsage(size_t size, uint32_t usage, const void* initialData) {
    // For CUDA, usage flags don't matter - just create a regular buffer
    return createBuffer(size, initialData);
}

void Device::freeBuffer(void* buffer) {
    if (buffer) {
        cudaFree(buffer);
    }
}

Result Device::uploadToBuffer(void* buffer, const void* data, size_t size, size_t offset) {
    if (!buffer || !data) {
        return Result::ErrorInvalidArgument;
    }

    cudaError_t err = cudaMemcpyAsync(
        static_cast<char*>(buffer) + offset,
        data,
        size,
        cudaMemcpyHostToDevice,
        m_impl->cudaStream
    );

    return (err == cudaSuccess) ? Result::Success : Result::ErrorCUDA;
}

Result Device::downloadFromBuffer(void* buffer, void* data, size_t size, size_t offset) {
    if (!buffer || !data) {
        return Result::ErrorInvalidArgument;
    }

    cudaError_t err = cudaMemcpyAsync(
        data,
        static_cast<char*>(buffer) + offset,
        size,
        cudaMemcpyDeviceToHost,
        m_impl->cudaStream
    );

    if (err == cudaSuccess) {
        cudaStreamSynchronize(m_impl->cudaStream);
        return Result::Success;
    }

    return Result::ErrorCUDA;
}

//------------------------------------------------------------------------------
// Global device management
//------------------------------------------------------------------------------

static std::unique_ptr<Device> g_globalDevice;
static std::mutex g_deviceMutex;

Device& getGlobalDevice() {
    std::lock_guard<std::mutex> lock(g_deviceMutex);
    if (!g_globalDevice) {
        g_globalDevice = std::make_unique<Device>();
    }
    return *g_globalDevice;
}

Result initializeGlobalDevice(int cudaDeviceId, bool enableValidation) {
    return getGlobalDevice().initialize(cudaDeviceId, enableValidation);
}

void shutdownGlobalDevice() {
    std::lock_guard<std::mutex> lock(g_deviceMutex);
    if (g_globalDevice) {
        g_globalDevice->shutdown();
        g_globalDevice.reset();
    }
}

} // namespace gaussian_rt
