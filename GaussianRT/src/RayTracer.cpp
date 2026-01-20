// GaussianRT - Ray Tracer Implementation
// Apache License 2.0

#include "RayTracer.h"
#include <stdexcept>

namespace gaussianrt {

namespace {

constexpr size_t kBufferSize = 16;  // Number of hits to buffer per ray
constexpr size_t kPayloadSize = sizeof(uint32_t) * 2 * kBufferSize;

} // namespace

std::unique_ptr<RayTracer> RayTracer::create(
    const Device& device,
    const RayTracerConfig& config) {

    auto tracer = std::unique_ptr<RayTracer>(new RayTracer());
    tracer->load_shader_program(device, config.shader_path);
    tracer->create_pipeline(device, config);
    return tracer;
}

void RayTracer::load_shader_program(
    const Device& device,
    const std::filesystem::path& path) {

    auto session = device.slang_session();
    Slang::ComPtr<slang::IBlob> diagnostics;

    // Load shader module
    slang::IModule* module = session->loadModule(path.string().c_str(), diagnostics.writeRef());
    if (diagnostics) {
        printf("Shader diagnostics: %s\n", static_cast<const char*>(diagnostics->getBufferPointer()));
    }
    if (!module) {
        throw std::runtime_error("Failed to load shader module: " + path.string());
    }

    // Find entry points
    Slang::List<slang::IComponentType*> components;
    components.add(module);

    const char* entry_points[] = {
        "rayGenShader",
        "missShader",
        "anyHitShader",
        "intersectionShader"
    };

    for (const char* name : entry_points) {
        Slang::ComPtr<slang::IEntryPoint> entry_point;
        if (SLANG_FAILED(module->findEntryPointByName(name, entry_point.writeRef()))) {
            throw std::runtime_error(std::string("Failed to find entry point: ") + name);
        }
        components.add(entry_point);
    }

    // Link program
    Slang::ComPtr<slang::IComponentType> linked_program;
    SlangResult result = session->createCompositeComponentType(
        components.getBuffer(),
        components.getCount(),
        linked_program.writeRef(),
        diagnostics.writeRef());

    if (diagnostics) {
        printf("Link diagnostics: %s\n", static_cast<const char*>(diagnostics->getBufferPointer()));
    }
    if (SLANG_FAILED(result)) {
        throw std::runtime_error("Failed to link shader program");
    }

    // Create shader program
    rhi::ShaderProgramDesc program_desc = {};
    program_desc.slangGlobalScope = linked_program;
    if (SLANG_FAILED(device.get()->createShaderProgram(program_desc, shader_program_.writeRef()))) {
        throw std::runtime_error("Failed to create shader program");
    }
}

void RayTracer::create_pipeline(
    const Device& device,
    const RayTracerConfig& config) {

    using namespace rhi;

    // Define hit group
    const char* hit_group_name = "ellipsoidHitGroup";
    HitGroupDesc hit_group = {};
    hit_group.hitGroupName = hit_group_name;
    hit_group.anyHitEntryPoint = "anyHitShader";
    hit_group.intersectionEntryPoint = "intersectionShader";

    // Create ray tracing pipeline
    RayTracingPipelineDesc pipeline_desc = {};
    pipeline_desc.program = shader_program_;
    pipeline_desc.hitGroupCount = 1;
    pipeline_desc.hitGroups = &hit_group;
    pipeline_desc.maxRayPayloadSize = kPayloadSize;
    pipeline_desc.maxRecursion = config.max_recursion;

    if (SLANG_FAILED(device.get()->createRayTracingPipeline(pipeline_desc, pipeline_.writeRef()))) {
        throw std::runtime_error("Failed to create ray tracing pipeline");
    }

    // Create shader table
    const char* raygen_name = "rayGenShader";
    const char* miss_name = "missShader";

    ShaderTableDesc table_desc = {};
    table_desc.program = shader_program_;
    table_desc.hitGroupCount = 1;
    table_desc.hitGroupNames = &hit_group_name;
    table_desc.rayGenShaderCount = 1;
    table_desc.rayGenShaderEntryPointNames = &raygen_name;
    table_desc.missShaderCount = 1;
    table_desc.missShaderEntryPointNames = &miss_name;

    if (SLANG_FAILED(device.get()->createShaderTable(table_desc, shader_table_.writeRef()))) {
        throw std::runtime_error("Failed to create shader table");
    }
}

void RayTracer::set_primitives(
    const Device& device,
    const float3* means, size_t count,
    const float3* scales,
    const float4* quats,
    const float* densities,
    const float* features, size_t feature_size) {

    using namespace rhi;

    primitive_count_ = count;
    feature_size_ = feature_size;

    BufferUsage usage = BufferUsage::ShaderResource;
    ResourceState state = ResourceState::ShaderResource;

    // Create buffers
    rhi::BufferDesc desc = {};
    desc.usage = usage;
    desc.defaultState = state;

    desc.size = count * sizeof(float3);
    desc.elementSize = sizeof(float3);
    means_buffer_ = device.get()->createBuffer(desc, means);

    desc.size = count * sizeof(float3);
    scales_buffer_ = device.get()->createBuffer(desc, scales);

    desc.size = count * sizeof(float4);
    desc.elementSize = sizeof(float4);
    quats_buffer_ = device.get()->createBuffer(desc, quats);

    desc.size = count * sizeof(float);
    desc.elementSize = sizeof(float);
    densities_buffer_ = device.get()->createBuffer(desc, densities);

    desc.size = count * feature_size * sizeof(float);
    features_buffer_ = device.get()->createBuffer(desc, features);

    if (!means_buffer_ || !scales_buffer_ || !quats_buffer_ ||
        !densities_buffer_ || !features_buffer_) {
        throw std::runtime_error("Failed to create primitive buffers");
    }
}

void RayTracer::trace_rays(
    const Device& device,
    const AccelerationStructure& accel,
    const float3* ray_origins,
    const float3* ray_directions,
    size_t ray_count,
    const TraceParams& params,
    RenderOutput& output) {

    using namespace rhi;

    // Create temporary buffers for ray data
    auto origins_buffer = device.create_structured_buffer(
        ray_count, sizeof(float3),
        BufferUsage::ShaderResource,
        ray_origins);

    auto directions_buffer = device.create_structured_buffer(
        ray_count, sizeof(float3),
        BufferUsage::ShaderResource,
        ray_directions);

    // Create output buffers
    auto image_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto iters_buffer = device.create_structured_buffer(
        ray_count, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto touch_buffer = device.create_structured_buffer(
        primitive_count_, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto state_buffer = device.create_structured_buffer(
        ray_count, sizeof(VolumeState),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto dirac_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto face_buffer = device.create_structured_buffer(
        ray_count, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto drgb_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    // Begin ray tracing pass
    auto encoder = device.create_command_encoder();
    auto rt_encoder = encoder->beginRayTracingPass();
    auto root_object = rt_encoder->bindPipeline(pipeline_, shader_table_);
    ShaderCursor cursor(root_object);

    // Bind output buffers
    cursor["image"].setBinding(image_buffer);
    cursor["iters"].setBinding(iters_buffer);
    cursor["touch_count"].setBinding(touch_buffer);
    cursor["last_state"].setBinding(state_buffer);
    cursor["last_dirac"].setBinding(dirac_buffer);
    cursor["last_face"].setBinding(face_buffer);
    cursor["initial_drgb"].setBinding(drgb_buffer);

    // Bind input buffers
    cursor["ray_origins"].setBinding(origins_buffer);
    cursor["ray_directions"].setBinding(directions_buffer);
    cursor["means"].setBinding(means_buffer_);
    cursor["scales"].setBinding(scales_buffer_);
    cursor["quats"].setBinding(quats_buffer_);
    cursor["densities"].setBinding(densities_buffer_);
    cursor["features"].setBinding(features_buffer_);

    // Bind acceleration structure
    cursor["traversable"].setBinding(accel.tlas());

    // Set uniform parameters
    cursor["sh_degree"].setData(&params.sh_degree, sizeof(params.sh_degree));
    cursor["max_iters"].setData(&params.max_iters, sizeof(params.max_iters));
    cursor["tmin"].setData(&params.tmin, sizeof(params.tmin));
    cursor["tmax"].setData(&params.tmax, sizeof(params.tmax));
    cursor["max_prim_size"].setData(&params.max_prim_size, sizeof(params.max_prim_size));

    // Dispatch rays
    rt_encoder->dispatchRays(0, static_cast<uint32_t>(ray_count), 1, 1);
    rt_encoder->end();

    device.submit_and_wait(encoder);

    // Copy results back if output pointers provided
    // Note: In production, you'd use async copy or keep data on GPU
}

void RayTracer::render(
    const Device& device,
    const AccelerationStructure& accel,
    const Camera& camera,
    const TraceParams& params,
    RenderOutput& output) {

    using namespace rhi;

    size_t ray_count = static_cast<size_t>(camera.width) * camera.height;

    // Create output buffers
    auto image_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto iters_buffer = device.create_structured_buffer(
        ray_count, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto touch_buffer = device.create_structured_buffer(
        primitive_count_, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto state_buffer = device.create_structured_buffer(
        ray_count, sizeof(VolumeState),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto dirac_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto face_buffer = device.create_structured_buffer(
        ray_count, sizeof(uint),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    auto drgb_buffer = device.create_structured_buffer(
        ray_count, sizeof(float4),
        BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
        nullptr);

    // Begin ray tracing pass
    auto encoder = device.create_command_encoder();
    auto rt_encoder = encoder->beginRayTracingPass();
    auto root_object = rt_encoder->bindPipeline(pipeline_, shader_table_);
    ShaderCursor cursor(root_object);

    // Bind output buffers
    cursor["image"].setBinding(image_buffer);
    cursor["iters"].setBinding(iters_buffer);
    cursor["touch_count"].setBinding(touch_buffer);
    cursor["last_state"].setBinding(state_buffer);
    cursor["last_dirac"].setBinding(dirac_buffer);
    cursor["last_face"].setBinding(face_buffer);
    cursor["initial_drgb"].setBinding(drgb_buffer);

    // Bind primitive buffers
    cursor["means"].setBinding(means_buffer_);
    cursor["scales"].setBinding(scales_buffer_);
    cursor["quats"].setBinding(quats_buffer_);
    cursor["densities"].setBinding(densities_buffer_);
    cursor["features"].setBinding(features_buffer_);

    // Bind acceleration structure
    cursor["traversable"].setBinding(accel.tlas());

    // Set camera
    cursor["camera"].setData(&camera, sizeof(Camera));

    // Set uniform parameters
    cursor["sh_degree"].setData(&params.sh_degree, sizeof(params.sh_degree));
    cursor["max_iters"].setData(&params.max_iters, sizeof(params.max_iters));
    cursor["tmin"].setData(&params.tmin, sizeof(params.tmin));
    cursor["tmax"].setData(&params.tmax, sizeof(params.tmax));
    cursor["max_prim_size"].setData(&params.max_prim_size, sizeof(params.max_prim_size));

    // Dispatch rays (2D for camera rendering)
    rt_encoder->dispatchRays(0,
        static_cast<uint32_t>(camera.width),
        static_cast<uint32_t>(camera.height),
        1);
    rt_encoder->end();

    device.submit_and_wait(encoder);
}

} // namespace gaussianrt
