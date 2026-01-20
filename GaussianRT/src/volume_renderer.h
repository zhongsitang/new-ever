#pragma once

#include "device.h"
#include "acceleration_structure.h"
#include "renderer.h"
#include "types.h"
#include <memory>
#include <string>

namespace gaussian_rt {

// High-level volume renderer for differentiable rendering
// Manages forward and backward passes for PyTorch integration
class VolumeRenderer {
public:
    VolumeRenderer();
    ~VolumeRenderer();

    // Initialize renderer
    bool initialize(int device_index = 0, const std::string& shader_dir = "");

    // Build/update acceleration structure
    void build_accel(
        const float* positions,    // [N, 3]
        const float* scales,       // [N, 3]
        const float* rotations,    // [N, 4]
        uint32_t num_elements,
        bool fast_build = false
    );

    // Forward pass: trace rays and compute colors
    // Returns: colors [num_rays, 4], states [num_rays, state_size]
    void forward(
        // Scene data (on GPU)
        const float* positions,    // [N, 3]
        const float* scales,       // [N, 3]
        const float* rotations,    // [N, 4]
        const float* opacities,    // [N]
        const float* features,     // [N, feature_dim]
        uint32_t num_elements,
        uint32_t feature_dim,
        // Ray data (on GPU)
        const float* ray_origins,  // [num_rays, 3]
        const float* ray_dirs,     // [num_rays, 3]
        uint32_t num_rays,
        // Render params
        float t_min,
        float t_max,
        uint32_t max_samples,
        uint32_t sh_degree,
        // Outputs (on GPU)
        float* out_colors,         // [num_rays, 4]
        float* out_states,         // [num_rays, state_size]
        float* out_last_points,    // [num_rays, point_size]
        int* out_sample_counts,    // [num_rays]
        int* out_sample_indices,   // [num_rays * max_samples]
        int* out_touch_counts      // [N]
    );

    // Backward pass: compute gradients
    void backward(
        // Forward pass outputs
        const float* final_states,     // [num_rays, state_size]
        const float* last_points,      // [num_rays, point_size]
        const int* sample_counts,      // [num_rays]
        const int* sample_indices,     // [num_rays * max_samples]
        // Scene data
        const float* positions,        // [N, 3]
        const float* scales,           // [N, 3]
        const float* rotations,        // [N, 4]
        const float* opacities,        // [N]
        const float* features,         // [N, feature_dim]
        uint32_t num_elements,
        uint32_t feature_dim,
        // Ray data
        const float* ray_origins,      // [num_rays, 3]
        const float* ray_dirs,         // [num_rays, 3]
        uint32_t num_rays,
        // Upstream gradients
        const float* grad_colors,      // [num_rays, 4]
        const float* grad_depths,      // [num_rays]
        const float* grad_distortions, // [num_rays]
        // Render params
        float t_min,
        float t_max,
        uint32_t max_samples,
        uint32_t sh_degree,
        // Output gradients
        float* grad_positions,         // [N, 3]
        float* grad_scales,            // [N, 3]
        float* grad_rotations,         // [N, 4]
        float* grad_opacities,         // [N]
        float* grad_features,          // [N, feature_dim]
        float* grad_ray_origins,       // [num_rays, 3]
        float* grad_ray_dirs           // [num_rays, 3]
    );

    // Accessors
    Device& get_device() { return device_; }
    bool is_initialized() const { return initialized_; }

private:
    Device device_;
    std::unique_ptr<Renderer> renderer_;
    std::unique_ptr<AccelerationStructure> accel_;

    std::string shader_dir_;
    bool initialized_ = false;

    // Cached pointers for accel rebuild
    const float* cached_positions_ = nullptr;
    const float* cached_scales_ = nullptr;
    const float* cached_rotations_ = nullptr;
    uint32_t cached_num_elements_ = 0;
};

// Global renderer instance
VolumeRenderer& get_volume_renderer();

} // namespace gaussian_rt
