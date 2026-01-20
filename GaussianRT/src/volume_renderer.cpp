#include "volume_renderer.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace gaussian_rt {

static VolumeRenderer* g_volume_renderer = nullptr;

// External CUDA kernels
extern void launch_compute_aabbs(
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    AABB* aabbs,
    uint32_t num_elements,
    cudaStream_t stream
);

extern void launch_initialize_render_state(
    const float3* ray_origins,
    const float3* ray_directions,
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float* features,
    uint32_t num_rays,
    uint32_t num_elements,
    uint32_t feature_dim,
    float t_min,
    RenderState* states,
    float4* initial_contributions,
    cudaStream_t stream
);

// Backward kernel launcher (compiled from Slang)
extern "C" void launch_backward_kernel(
    // Forward outputs
    const float* final_states,
    const float* last_points,
    const int* sample_counts,
    const int* sample_indices,
    // Scene data
    const float* positions,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const float* features,
    uint32_t num_elements,
    uint32_t feature_dim,
    // Ray data
    const float* ray_origins,
    const float* ray_dirs,
    uint32_t num_rays,
    // Upstream gradients
    const float* grad_colors,
    const float* grad_depths,
    const float* grad_distortions,
    // Params
    float t_min,
    float t_max,
    uint32_t max_samples,
    uint32_t sh_degree,
    // Output gradients
    float* grad_positions,
    float* grad_scales,
    float* grad_rotations,
    float* grad_opacities,
    float* grad_features,
    float* grad_ray_origins,
    float* grad_ray_dirs,
    cudaStream_t stream
);

VolumeRenderer::VolumeRenderer() = default;

VolumeRenderer::~VolumeRenderer() {
    accel_.reset();
    renderer_.reset();
    device_.shutdown();
}

bool VolumeRenderer::initialize(int device_index, const std::string& shader_dir) {
    if (initialized_) return true;

    shader_dir_ = shader_dir.empty() ? "." : shader_dir;

    // Initialize device
    if (!device_.initialize(device_index)) {
        std::cerr << "Failed to initialize device\n";
        return false;
    }

    // Check ray tracing support
    if (!device_.supports_ray_tracing()) {
        std::cerr << "Device does not support ray tracing\n";
        return false;
    }

    // Create renderer
    renderer_ = std::make_unique<Renderer>(device_);
    std::string shader_path = shader_dir_ + "/forward.slang";

    if (!renderer_->initialize(shader_path)) {
        std::cerr << "Failed to initialize renderer\n";
        return false;
    }

    initialized_ = true;
    return true;
}

void VolumeRenderer::build_accel(
    const float* positions,
    const float* scales,
    const float* rotations,
    uint32_t num_elements,
    bool fast_build
) {
    if (!initialized_) {
        throw std::runtime_error("VolumeRenderer not initialized");
    }

    // Create acceleration structure if needed
    if (!accel_) {
        accel_ = std::make_unique<AccelerationStructure>(device_);
    }

    // Build/rebuild acceleration structure
    accel_->build(
        reinterpret_cast<const float3*>(positions),
        reinterpret_cast<const float3*>(scales),
        reinterpret_cast<const float4*>(rotations),
        num_elements,
        fast_build
    );

    // Cache for potential rebuilds
    cached_positions_ = positions;
    cached_scales_ = scales;
    cached_rotations_ = rotations;
    cached_num_elements_ = num_elements;
}

void VolumeRenderer::forward(
    const float* positions,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const float* features,
    uint32_t num_elements,
    uint32_t feature_dim,
    const float* ray_origins,
    const float* ray_dirs,
    uint32_t num_rays,
    float t_min,
    float t_max,
    uint32_t max_samples,
    uint32_t sh_degree,
    float* out_colors,
    float* out_states,
    float* out_last_points,
    int* out_sample_counts,
    int* out_sample_indices,
    int* out_touch_counts
) {
    if (!initialized_ || !accel_) {
        throw std::runtime_error("VolumeRenderer not initialized or accel not built");
    }

    // Setup scene data
    SceneData scene;
    scene.positions = const_cast<float3*>(reinterpret_cast<const float3*>(positions));
    scene.scales = const_cast<float3*>(reinterpret_cast<const float3*>(scales));
    scene.rotations = const_cast<float4*>(reinterpret_cast<const float4*>(rotations));
    scene.opacities = const_cast<float*>(opacities);
    scene.features = const_cast<float*>(features);
    scene.num_elements = num_elements;
    scene.feature_dim = feature_dim;

    // Setup render params
    RenderParams params;
    params.width = num_rays;  // Treat as 1D array of rays
    params.height = 1;
    params.t_min = t_min;
    params.t_max = t_max;
    params.max_samples_per_ray = max_samples;
    params.sh_degree = sh_degree;
    params.transmittance_threshold = 0.004f;

    // Setup output
    ForwardOutput output;
    output.colors = reinterpret_cast<float4*>(out_colors);
    output.final_states = reinterpret_cast<RenderState*>(out_states);
    output.last_points = reinterpret_cast<ControlPoint*>(out_last_points);
    output.sample_counts = reinterpret_cast<uint32_t*>(out_sample_counts);
    output.sample_indices = out_sample_indices;
    output.element_touch_counts = reinterpret_cast<uint32_t*>(out_touch_counts);

    // Render
    renderer_->render(
        *accel_,
        scene,
        reinterpret_cast<const float3*>(ray_origins),
        reinterpret_cast<const float3*>(ray_dirs),
        num_rays,
        params,
        output
    );
}

void VolumeRenderer::backward(
    const float* final_states,
    const float* last_points,
    const int* sample_counts,
    const int* sample_indices,
    const float* positions,
    const float* scales,
    const float* rotations,
    const float* opacities,
    const float* features,
    uint32_t num_elements,
    uint32_t feature_dim,
    const float* ray_origins,
    const float* ray_dirs,
    uint32_t num_rays,
    const float* grad_colors,
    const float* grad_depths,
    const float* grad_distortions,
    float t_min,
    float t_max,
    uint32_t max_samples,
    uint32_t sh_degree,
    float* grad_positions,
    float* grad_scales,
    float* grad_rotations,
    float* grad_opacities,
    float* grad_features,
    float* grad_ray_origins,
    float* grad_ray_dirs
) {
    if (!initialized_) {
        throw std::runtime_error("VolumeRenderer not initialized");
    }

    // Zero gradients
    cudaMemset(grad_positions, 0, num_elements * 3 * sizeof(float));
    cudaMemset(grad_scales, 0, num_elements * 3 * sizeof(float));
    cudaMemset(grad_rotations, 0, num_elements * 4 * sizeof(float));
    cudaMemset(grad_opacities, 0, num_elements * sizeof(float));
    cudaMemset(grad_features, 0, num_elements * feature_dim * sizeof(float));
    cudaMemset(grad_ray_origins, 0, num_rays * 3 * sizeof(float));
    cudaMemset(grad_ray_dirs, 0, num_rays * 3 * sizeof(float));

    // Launch backward kernel
    launch_backward_kernel(
        final_states,
        last_points,
        sample_counts,
        sample_indices,
        positions,
        scales,
        rotations,
        opacities,
        features,
        num_elements,
        feature_dim,
        ray_origins,
        ray_dirs,
        num_rays,
        grad_colors,
        grad_depths,
        grad_distortions,
        t_min,
        t_max,
        max_samples,
        sh_degree,
        grad_positions,
        grad_scales,
        grad_rotations,
        grad_opacities,
        grad_features,
        grad_ray_origins,
        grad_ray_dirs,
        nullptr  // default stream
    );

    cudaDeviceSynchronize();
}

VolumeRenderer& get_volume_renderer() {
    if (!g_volume_renderer) {
        g_volume_renderer = new VolumeRenderer();
    }
    return *g_volume_renderer;
}

} // namespace gaussian_rt
