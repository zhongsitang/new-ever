// GaussianRT - Device Implementation
// Apache License 2.0

#include "Device.h"

#ifndef GAUSSIANRT_NO_CUDA
#include <cuda_runtime.h>
#endif

namespace gaussianrt {

namespace {

#ifndef GAUSSIANRT_NO_CUDA
void check_cuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        throw DeviceError(std::string(msg) + ": " + cudaGetErrorString(result));
    }
}
#endif

} // namespace

Device::Device(int cuda_device_index) : cuda_device_index_(cuda_device_index) {
#ifndef GAUSSIANRT_NO_CUDA
    // Initialize CUDA device
    check_cuda(cudaSetDevice(cuda_device_index), "Failed to set CUDA device");
    check_cuda(cudaFree(nullptr), "Failed to initialize CUDA context");
#endif

    // Create RHI device
    rhi::DeviceDesc desc = {};
    desc.deviceType = rhi::DeviceType::CUDA;
    desc.enableValidation = false;
    desc.adapterIndex = cuda_device_index;

    device_ = rhi::getRHI()->createDevice(desc);
    if (!device_) {
        throw DeviceError("Failed to create slang-rhi CUDA device");
    }

    // Get the graphics queue (used for ray tracing)
    queue_ = device_->getQueue(rhi::QueueType::Graphics);
    if (!queue_) {
        throw DeviceError("Failed to get graphics queue");
    }
}

std::unique_ptr<Device> Device::create(int cuda_device_index) {
    return std::unique_ptr<Device>(new Device(cuda_device_index));
}

slang::ISession* Device::slang_session() const {
    return static_cast<slang::ISession*>(device_->getSlangSession());
}

Slang::ComPtr<rhi::ICommandEncoder> Device::create_command_encoder() const {
    return Slang::ComPtr<rhi::ICommandEncoder>(queue_->createCommandEncoder());
}

void Device::submit_and_wait(rhi::ICommandEncoder* encoder) const {
    queue_->submit(encoder->finish());
    queue_->waitOnHost();
}

Slang::ComPtr<rhi::IBuffer> Device::create_buffer(
    size_t size,
    rhi::BufferUsage usage,
    const void* initial_data) const {

    rhi::BufferDesc desc = {};
    desc.size = size;
    desc.usage = usage;
    desc.defaultState = (usage & rhi::BufferUsage::UnorderedAccess) != rhi::BufferUsage::None
        ? rhi::ResourceState::UnorderedAccess
        : rhi::ResourceState::ShaderResource;

    return Slang::ComPtr<rhi::IBuffer>(device_->createBuffer(desc, initial_data));
}

Slang::ComPtr<rhi::IBuffer> Device::create_structured_buffer(
    size_t element_count,
    size_t element_size,
    rhi::BufferUsage usage,
    const void* initial_data) const {

    rhi::BufferDesc desc = {};
    desc.size = element_count * element_size;
    desc.elementSize = element_size;
    desc.usage = usage;
    desc.defaultState = (usage & rhi::BufferUsage::UnorderedAccess) != rhi::BufferUsage::None
        ? rhi::ResourceState::UnorderedAccess
        : rhi::ResourceState::ShaderResource;

    return Slang::ComPtr<rhi::IBuffer>(device_->createBuffer(desc, initial_data));
}

} // namespace gaussianrt
