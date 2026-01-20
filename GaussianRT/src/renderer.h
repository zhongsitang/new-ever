#pragma once

#include "device.h"
#include "acceleration_structure.h"
#include "types.h"
#include <memory>
#include <vector>

namespace gaussian_rt {

// Forward rendering pipeline using slang-rhi ray tracing
class Renderer {
public:
    Renderer(Device& device);
    ~Renderer();

    // Initialize ray tracing pipeline
    bool initialize(const std::string& shader_path);

    // Render a frame
    void render(
        AccelerationStructure& accel,
        const SceneData& scene,
        const float3* ray_origins,
        const float3* ray_directions,
        uint32_t num_rays,
        const RenderParams& params,
        ForwardOutput& output
    );

    // Get ray tracing pipeline
    rhi::IRayTracingPipeline* get_pipeline() const { return pipeline_.get(); }

private:
    Device& device_;

    // Ray tracing pipeline
    Slang::ComPtr<rhi::IRayTracingPipeline> pipeline_;
    Slang::ComPtr<rhi::IShaderTable> shader_table_;
    Slang::ComPtr<rhi::IShaderProgram> shader_program_;

    // Buffers
    Slang::ComPtr<rhi::IBuffer> uniform_buffer_;
    Slang::ComPtr<rhi::IBuffer> ray_origin_buffer_;
    Slang::ComPtr<rhi::IBuffer> ray_direction_buffer_;
    Slang::ComPtr<rhi::IBuffer> output_color_buffer_;
    Slang::ComPtr<rhi::IBuffer> output_state_buffer_;
    Slang::ComPtr<rhi::IBuffer> output_point_buffer_;
    Slang::ComPtr<rhi::IBuffer> output_count_buffer_;
    Slang::ComPtr<rhi::IBuffer> output_index_buffer_;
    Slang::ComPtr<rhi::IBuffer> touch_count_buffer_;

    // Scene data buffers (references to external data)
    Slang::ComPtr<rhi::IBuffer> position_buffer_;
    Slang::ComPtr<rhi::IBuffer> scale_buffer_;
    Slang::ComPtr<rhi::IBuffer> rotation_buffer_;
    Slang::ComPtr<rhi::IBuffer> opacity_buffer_;
    Slang::ComPtr<rhi::IBuffer> feature_buffer_;

    uint32_t max_rays_ = 0;
    uint32_t max_elements_ = 0;
    uint32_t max_samples_per_ray_ = 0;

    void create_buffers(uint32_t num_rays, uint32_t num_elements, uint32_t max_samples);
    void update_scene_buffers(const SceneData& scene);
    void update_ray_buffers(const float3* origins, const float3* directions, uint32_t num_rays);
    bool build_pipeline(const std::string& shader_path);
};

} // namespace gaussian_rt
