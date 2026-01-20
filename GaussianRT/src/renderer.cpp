#include "renderer.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace gaussian_rt {

// Uniform structure matching the shader
struct RenderUniforms {
    uint32_t width;
    uint32_t height;
    float t_min;
    float t_max;
    uint32_t max_samples;
    uint32_t sh_degree;
    float background_color[3];
    uint32_t num_elements;
};

Renderer::Renderer(Device& device)
    : device_(device) {}

Renderer::~Renderer() = default;

bool Renderer::initialize(const std::string& shader_path) {
    return build_pipeline(shader_path);
}

bool Renderer::build_pipeline(const std::string& shader_path) {
    auto rhi_device = device_.get_device();

    // Load shader program
    shader_program_ = device_.load_shader_program(
        shader_path,
        {"raygen_shader", "miss_shader", "anyhit_shader", "intersection_shader"}
    );

    if (!shader_program_) {
        std::cerr << "Failed to load shader program\n";
        return false;
    }

    // Create ray tracing pipeline
    rhi::RayTracingPipelineDesc pipeline_desc = {};
    pipeline_desc.program = shader_program_;

    // Hit group configuration
    rhi::HitGroupDesc hit_group = {};
    hit_group.anyHitEntryPoint = "anyhit_shader";
    hit_group.intersectionEntryPoint = "intersection_shader";
    hit_group.hitGroupName = "ellipsoid_hit_group";

    pipeline_desc.hitGroups = &hit_group;
    pipeline_desc.hitGroupCount = 1;
    pipeline_desc.maxRayPayloadSize = sizeof(float) * 64;  // Conservative estimate
    pipeline_desc.maxRecursion = 1;
    pipeline_desc.maxAttributeSize = sizeof(float) * 2;

    if (SLANG_FAILED(rhi_device->createRayTracingPipeline(pipeline_desc, pipeline_.writeRef()))) {
        std::cerr << "Failed to create ray tracing pipeline\n";
        return false;
    }

    // Create shader table
    const char* ray_gen_names[] = {"raygen_shader"};
    const char* miss_names[] = {"miss_shader"};
    const char* hit_group_names[] = {"ellipsoid_hit_group"};

    rhi::ShaderTableDesc table_desc = {};
    table_desc.rayGenShaderNames = ray_gen_names;
    table_desc.rayGenShaderCount = 1;
    table_desc.missShaderNames = miss_names;
    table_desc.missShaderCount = 1;
    table_desc.hitGroupNames = hit_group_names;
    table_desc.hitGroupCount = 1;

    if (SLANG_FAILED(pipeline_->createShaderTable(table_desc, shader_table_.writeRef()))) {
        std::cerr << "Failed to create shader table\n";
        return false;
    }

    return true;
}

void Renderer::create_buffers(uint32_t num_rays, uint32_t num_elements, uint32_t max_samples) {
    bool need_recreate = (num_rays > max_rays_) ||
                         (num_elements > max_elements_) ||
                         (max_samples > max_samples_per_ray_);

    if (!need_recreate) return;

    max_rays_ = std::max(max_rays_, num_rays);
    max_elements_ = std::max(max_elements_, num_elements);
    max_samples_per_ray_ = std::max(max_samples_per_ray_, max_samples);

    auto usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess;

    // Ray buffers
    ray_origin_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(float3), usage, rhi::MemoryType::DeviceLocal);
    ray_direction_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(float3), usage, rhi::MemoryType::DeviceLocal);

    // Output buffers
    output_color_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(float4), usage, rhi::MemoryType::DeviceLocal);
    output_state_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(RenderState), usage, rhi::MemoryType::DeviceLocal);
    output_point_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(ControlPoint), usage, rhi::MemoryType::DeviceLocal);
    output_count_buffer_ = device_.create_buffer(
        max_rays_ * sizeof(uint32_t), usage, rhi::MemoryType::DeviceLocal);
    output_index_buffer_ = device_.create_buffer(
        max_rays_ * max_samples_per_ray_ * sizeof(int32_t), usage, rhi::MemoryType::DeviceLocal);
    touch_count_buffer_ = device_.create_buffer(
        max_elements_ * sizeof(uint32_t), usage, rhi::MemoryType::DeviceLocal);

    // Uniform buffer
    uniform_buffer_ = device_.create_buffer(
        sizeof(RenderUniforms),
        rhi::BufferUsage::ConstantBuffer,
        rhi::MemoryType::Upload
    );
}

void Renderer::update_scene_buffers(const SceneData& scene) {
    auto usage = rhi::BufferUsage::ShaderResource;

    // Create/update scene data buffers
    size_t position_size = scene.num_elements * sizeof(float3);
    size_t scale_size = scene.num_elements * sizeof(float3);
    size_t rotation_size = scene.num_elements * sizeof(float4);
    size_t opacity_size = scene.num_elements * sizeof(float);
    size_t feature_size = scene.num_elements * scene.feature_dim * sizeof(float);

    position_buffer_ = device_.create_buffer(position_size, usage, rhi::MemoryType::DeviceLocal, scene.positions);
    scale_buffer_ = device_.create_buffer(scale_size, usage, rhi::MemoryType::DeviceLocal, scene.scales);
    rotation_buffer_ = device_.create_buffer(rotation_size, usage, rhi::MemoryType::DeviceLocal, scene.rotations);
    opacity_buffer_ = device_.create_buffer(opacity_size, usage, rhi::MemoryType::DeviceLocal, scene.opacities);
    feature_buffer_ = device_.create_buffer(feature_size, usage, rhi::MemoryType::DeviceLocal, scene.features);
}

void Renderer::update_ray_buffers(const float3* origins, const float3* directions, uint32_t num_rays) {
    // Copy ray data to GPU
    size_t origin_size = num_rays * sizeof(float3);
    size_t dir_size = num_rays * sizeof(float3);

    auto staging_origin = device_.create_buffer(
        origin_size, rhi::BufferUsage::CopySource, rhi::MemoryType::Upload, origins);
    auto staging_dir = device_.create_buffer(
        dir_size, rhi::BufferUsage::CopySource, rhi::MemoryType::Upload, directions);

    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    device_.get_queue()->createCommandEncoder(encoder.writeRef());

    encoder->copyBuffer(ray_origin_buffer_, 0, staging_origin, 0, origin_size);
    encoder->copyBuffer(ray_direction_buffer_, 0, staging_dir, 0, dir_size);

    Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
    encoder->finish(cmd_buffer.writeRef());
    device_.submit_and_wait(cmd_buffer);
}

void Renderer::render(
    AccelerationStructure& accel,
    const SceneData& scene,
    const float3* ray_origins,
    const float3* ray_directions,
    uint32_t num_rays,
    const RenderParams& params,
    ForwardOutput& output
) {
    // Ensure buffers are large enough
    create_buffers(num_rays, scene.num_elements, params.max_samples_per_ray);

    // Update scene buffers
    update_scene_buffers(scene);

    // Update ray buffers
    update_ray_buffers(ray_origins, ray_directions, num_rays);

    // Update uniforms
    RenderUniforms uniforms;
    uniforms.width = params.width;
    uniforms.height = params.height;
    uniforms.t_min = params.t_min;
    uniforms.t_max = params.t_max;
    uniforms.max_samples = params.max_samples_per_ray;
    uniforms.sh_degree = params.sh_degree;
    uniforms.background_color[0] = 0.0f;
    uniforms.background_color[1] = 0.0f;
    uniforms.background_color[2] = 0.0f;
    uniforms.num_elements = scene.num_elements;

    void* uniform_ptr = nullptr;
    uniform_buffer_->map(nullptr, &uniform_ptr);
    if (uniform_ptr) {
        memcpy(uniform_ptr, &uniforms, sizeof(uniforms));
        uniform_buffer_->unmap(nullptr);
    }

    // Clear touch count buffer
    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    device_.get_queue()->createCommandEncoder(encoder.writeRef());

    // Begin ray tracing pass
    auto rt_encoder = encoder->beginRayTracingPass();

    // Bind pipeline
    auto root_object = rt_encoder->bindPipeline(pipeline_, shader_table_);
    rhi::ShaderCursor cursor(root_object);

    // Bind resources
    cursor["uniforms"].setData(&uniforms, sizeof(uniforms));
    cursor["scene_bvh"].setBinding(accel.get_handle());
    cursor["positions"].setBinding(position_buffer_);
    cursor["scales"].setBinding(scale_buffer_);
    cursor["rotations"].setBinding(rotation_buffer_);
    cursor["opacities"].setBinding(opacity_buffer_);
    cursor["features"].setBinding(feature_buffer_);
    cursor["ray_origins"].setBinding(ray_origin_buffer_);
    cursor["ray_directions"].setBinding(ray_direction_buffer_);
    cursor["output_colors"].setBinding(output_color_buffer_);
    cursor["output_states"].setBinding(output_state_buffer_);
    cursor["output_last_points"].setBinding(output_point_buffer_);
    cursor["output_sample_counts"].setBinding(output_count_buffer_);
    cursor["output_sample_indices"].setBinding(output_index_buffer_);
    cursor["element_touch_counts"].setBinding(touch_count_buffer_);

    // Dispatch rays
    rt_encoder->dispatchRays(0, params.width, params.height, 1);

    rt_encoder->end();

    Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
    encoder->finish(cmd_buffer.writeRef());
    device_.submit_and_wait(cmd_buffer);

    // Copy results back (for now, just set pointers - actual implementation
    // would need proper buffer readback)
    // In PyTorch integration, we'll use CUDA interop instead
}

} // namespace gaussian_rt
