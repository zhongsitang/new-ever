// GaussianRT - Ray Tracing Pipeline
// Core ray tracing implementation using slang-rhi
// Apache License 2.0

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include "slang-com-ptr.h"
#include "slang-rhi.h"
#include "slang-rhi/shader-cursor.h"
#include "Types.h"
#include "Device.h"
#include "AccelerationStructure.h"

namespace gaussianrt {

// Ray tracer configuration
struct RayTracerConfig {
    uint max_ray_payload_size = 128;   // Bytes for ray payload
    uint max_recursion = 1;            // No recursive rays for volume rendering
    std::filesystem::path shader_path; // Path to shader file
};

// Main ray tracing class
class RayTracer {
public:
    // Create ray tracer with given configuration
    static std::unique_ptr<RayTracer> create(
        const Device& device,
        const RayTracerConfig& config);

    // Upload primitive data to GPU buffers
    void set_primitives(
        const Device& device,
        const float3* means, size_t count,
        const float3* scales,
        const float4* quats,
        const float* densities,
        const float* features, size_t feature_size);

    // Trace rays against acceleration structure
    void trace_rays(
        const Device& device,
        const AccelerationStructure& accel,
        const float3* ray_origins,
        const float3* ray_directions,
        size_t ray_count,
        const TraceParams& params,
        RenderOutput& output);

    // Trace rays with camera (generates rays internally)
    void render(
        const Device& device,
        const AccelerationStructure& accel,
        const Camera& camera,
        const TraceParams& params,
        RenderOutput& output);

    ~RayTracer() = default;
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

private:
    RayTracer() = default;

    void load_shader_program(const Device& device, const std::filesystem::path& path);
    void create_pipeline(const Device& device, const RayTracerConfig& config);

    // Shader program and pipeline
    Slang::ComPtr<rhi::IShaderProgram> shader_program_;
    Slang::ComPtr<rhi::IRayTracingPipeline> pipeline_;
    Slang::ComPtr<rhi::IShaderTable> shader_table_;

    // Primitive data buffers
    Slang::ComPtr<rhi::IBuffer> means_buffer_;
    Slang::ComPtr<rhi::IBuffer> scales_buffer_;
    Slang::ComPtr<rhi::IBuffer> quats_buffer_;
    Slang::ComPtr<rhi::IBuffer> densities_buffer_;
    Slang::ComPtr<rhi::IBuffer> features_buffer_;

    size_t primitive_count_ = 0;
    size_t feature_size_ = 0;
};

} // namespace gaussianrt
