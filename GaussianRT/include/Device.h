// GaussianRT - RHI Device Management
// Clean abstraction over slang-rhi device creation
// Apache License 2.0

#pragma once

#include <memory>
#include <stdexcept>
#include "slang-com-ptr.h"
#include "slang-rhi.h"

namespace gaussianrt {

// RAII wrapper for slang-rhi device and queue
class Device {
public:
    // Create a CUDA-backed RHI device
    static std::unique_ptr<Device> create(int cuda_device_index = 0);

    // Accessors
    rhi::IDevice* get() const { return device_.get(); }
    rhi::ICommandQueue* queue() const { return queue_.get(); }
    slang::ISession* slang_session() const;
    int cuda_device_index() const { return cuda_device_index_; }

    // Command encoder creation
    Slang::ComPtr<rhi::ICommandEncoder> create_command_encoder() const;

    // Submit and wait
    void submit_and_wait(rhi::ICommandEncoder* encoder) const;

    // Buffer creation helpers
    Slang::ComPtr<rhi::IBuffer> create_buffer(
        size_t size,
        rhi::BufferUsage usage,
        const void* initial_data = nullptr) const;

    Slang::ComPtr<rhi::IBuffer> create_structured_buffer(
        size_t element_count,
        size_t element_size,
        rhi::BufferUsage usage,
        const void* initial_data = nullptr) const;

    ~Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

private:
    Device(int cuda_device_index);

    Slang::ComPtr<rhi::IDevice> device_;
    Slang::ComPtr<rhi::ICommandQueue> queue_;
    int cuda_device_index_;
};

// Exception for device errors
class DeviceError : public std::runtime_error {
public:
    explicit DeviceError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace gaussianrt
