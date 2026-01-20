#include "device.h"
#include <stdexcept>
#include <iostream>

namespace gaussian_rt {

static Device* g_device_instance = nullptr;

Device::Device() = default;

Device::~Device() {
    shutdown();
}

bool Device::initialize(int device_index) {
    if (initialized_) return true;

    device_index_ = device_index;

    // Create Slang global session
    slang::createGlobalSession(slang_global_session_.writeRef());
    if (!slang_global_session_) {
        std::cerr << "Failed to create Slang global session\n";
        return false;
    }

    // Create RHI device with CUDA backend
    rhi::DeviceDesc device_desc = {};
    device_desc.deviceType = rhi::DeviceType::CUDA;
    device_desc.slang.slangGlobalSession = slang_global_session_;
    device_desc.adapterIndex = device_index;

    if (SLANG_FAILED(rhi::createDevice(device_desc, device_.writeRef()))) {
        std::cerr << "Failed to create RHI device with CUDA backend\n";
        return false;
    }

    // Create command queue
    rhi::CommandQueueDesc queue_desc = {};
    queue_desc.type = rhi::QueueType::Graphics;
    if (SLANG_FAILED(device_->createCommandQueue(queue_desc, queue_.writeRef()))) {
        std::cerr << "Failed to create command queue\n";
        return false;
    }

    // Create Slang session for shader compilation
    slang::SessionDesc session_desc = {};

    slang::TargetDesc target_desc = {};
    target_desc.format = SLANG_SPIRV;  // Will be translated to CUDA
    target_desc.profile = slang_global_session_->findProfile("sm_7_5");
    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;

    if (SLANG_FAILED(slang_global_session_->createSession(session_desc, slang_session_.writeRef()))) {
        std::cerr << "Failed to create Slang session\n";
        return false;
    }

    initialized_ = true;
    return true;
}

void Device::shutdown() {
    if (!initialized_) return;

    wait_for_gpu();

    slang_session_ = nullptr;
    queue_ = nullptr;
    device_ = nullptr;
    slang_global_session_ = nullptr;

    initialized_ = false;
}

Slang::ComPtr<rhi::IBuffer> Device::create_buffer(
    size_t size,
    rhi::BufferUsage usage,
    rhi::MemoryType memory_type,
    const void* initial_data
) {
    rhi::BufferDesc desc = {};
    desc.size = size;
    desc.usage = usage;
    desc.memoryType = memory_type;

    Slang::ComPtr<rhi::IBuffer> buffer;
    if (SLANG_FAILED(device_->createBuffer(desc, initial_data, buffer.writeRef()))) {
        return nullptr;
    }
    return buffer;
}

Slang::ComPtr<rhi::ITexture> Device::create_texture_2d(
    uint32_t width,
    uint32_t height,
    rhi::Format format,
    rhi::TextureUsage usage
) {
    rhi::TextureDesc desc = {};
    desc.type = rhi::TextureType::Texture2D;
    desc.size.width = width;
    desc.size.height = height;
    desc.size.depth = 1;
    desc.format = format;
    desc.usage = usage;
    desc.numMipLevels = 1;
    desc.arrayLength = 1;
    desc.sampleCount = 1;

    Slang::ComPtr<rhi::ITexture> texture;
    if (SLANG_FAILED(device_->createTexture(desc, nullptr, texture.writeRef()))) {
        return nullptr;
    }
    return texture;
}

Slang::ComPtr<rhi::IShaderProgram> Device::load_shader_program(
    const std::string& module_path,
    const std::vector<std::string>& entry_points
) {
    Slang::ComPtr<slang::IBlob> diagnostics;

    // Load shader module
    slang::IModule* module = slang_session_->loadModule(
        module_path.c_str(),
        diagnostics.writeRef()
    );

    if (!module) {
        if (diagnostics) {
            std::cerr << "Shader compilation error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer())
                      << std::endl;
        }
        return nullptr;
    }

    // Collect entry points
    std::vector<slang::IComponentType*> components;
    components.push_back(module);

    for (const auto& entry_name : entry_points) {
        Slang::ComPtr<slang::IEntryPoint> entry_point;
        if (SLANG_FAILED(module->findEntryPointByName(entry_name.c_str(), entry_point.writeRef()))) {
            std::cerr << "Entry point not found: " << entry_name << std::endl;
            return nullptr;
        }
        components.push_back(entry_point.get());
    }

    // Link program
    Slang::ComPtr<slang::IComponentType> linked_program;
    if (SLANG_FAILED(slang_session_->createCompositeComponentType(
            components.data(),
            components.size(),
            linked_program.writeRef(),
            diagnostics.writeRef()))) {
        if (diagnostics) {
            std::cerr << "Shader linking error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer())
                      << std::endl;
        }
        return nullptr;
    }

    // Create shader program
    rhi::ShaderProgramDesc program_desc = {};
    program_desc.slangGlobalScope = linked_program;

    Slang::ComPtr<rhi::IShaderProgram> program;
    if (SLANG_FAILED(device_->createShaderProgram(program_desc, program.writeRef()))) {
        std::cerr << "Failed to create shader program\n";
        return nullptr;
    }

    return program;
}

void Device::submit_and_wait(rhi::ICommandBuffer* cmd) {
    queue_->submit(cmd);
    queue_->waitOnHost();
}

void Device::submit(rhi::ICommandBuffer* cmd) {
    queue_->submit(cmd);
}

void Device::wait_for_gpu() {
    if (queue_) {
        queue_->waitOnHost();
    }
}

bool Device::supports_ray_tracing() const {
    if (!device_) return false;
    const auto& features = device_->getDeviceInfo().features;
    return features.rayTracing;
}

Device& get_device() {
    if (!g_device_instance) {
        g_device_instance = new Device();
    }
    return *g_device_instance;
}

} // namespace gaussian_rt
