#pragma once

#include <slang.h>
#include <slang-rhi.h>
#include <slang-com-ptr.h>
#include <memory>
#include <string>
#include <vector>

namespace gaussian_rt {

// Slang-RHI device wrapper for ray tracing
class Device {
public:
    Device();
    ~Device();

    // Initialize device with CUDA backend
    bool initialize(int device_index = 0);
    void shutdown();

    // Accessors
    rhi::IDevice* get_device() const { return device_.get(); }
    rhi::ICommandQueue* get_queue() const { return queue_.get(); }
    slang::ISession* get_slang_session() const { return slang_session_.get(); }

    // Buffer creation helpers
    Slang::ComPtr<rhi::IBuffer> create_buffer(
        size_t size,
        rhi::BufferUsage usage,
        rhi::MemoryType memory_type = rhi::MemoryType::DeviceLocal,
        const void* initial_data = nullptr
    );

    Slang::ComPtr<rhi::ITexture> create_texture_2d(
        uint32_t width,
        uint32_t height,
        rhi::Format format,
        rhi::TextureUsage usage
    );

    // Shader compilation
    Slang::ComPtr<rhi::IShaderProgram> load_shader_program(
        const std::string& module_path,
        const std::vector<std::string>& entry_points
    );

    // Command execution
    void submit_and_wait(rhi::ICommandBuffer* cmd);
    void submit(rhi::ICommandBuffer* cmd);
    void wait_for_gpu();

    // Check ray tracing support
    bool supports_ray_tracing() const;

private:
    Slang::ComPtr<rhi::IDevice> device_;
    Slang::ComPtr<rhi::ICommandQueue> queue_;
    Slang::ComPtr<slang::IGlobalSession> slang_global_session_;
    Slang::ComPtr<slang::ISession> slang_session_;

    bool initialized_ = false;
    int device_index_ = 0;
};

// Singleton accessor
Device& get_device();

} // namespace gaussian_rt
