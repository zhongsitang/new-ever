#include "../types.h"
#include <cuda_runtime.h>

namespace gaussian_rt {

// Check if a point is inside an ellipsoid
__device__ bool point_inside_ellipsoid(
    float3 point,
    float3 center,
    float3 scale,
    float4 rotation
) {
    // Transform point to ellipsoid local coordinates
    float3 local_point = point - center;
    local_point = rotate_by_quaternion_inverse(local_point, rotation);

    // Check if inside unit sphere (after scaling)
    float3 normalized = make_float3(
        local_point.x / scale.x,
        local_point.y / scale.y,
        local_point.z / scale.z
    );

    float dist_sq = dot(normalized, normalized);
    return dist_sq <= 1.0f;
}

// Compute the contribution of an ellipsoid at a given point
// Returns the density and color contribution
__device__ void compute_initial_contribution(
    float3 ray_origin,
    float3 ray_direction,
    float3 center,
    float3 scale,
    float4 rotation,
    float opacity,
    const float* features,
    uint32_t sh_degree,
    float& out_density,
    float3& out_color
) {
    // For rays starting inside an ellipsoid, compute initial contribution
    // This handles the case where camera is inside a volume element

    // Transform ray origin to ellipsoid local coordinates
    float3 local_origin = ray_origin - center;
    local_origin = rotate_by_quaternion_inverse(local_origin, rotation);

    // Normalize by scale
    float3 normalized = make_float3(
        local_origin.x / scale.x,
        local_origin.y / scale.y,
        local_origin.z / scale.z
    );

    float dist_sq = dot(normalized, normalized);

    if (dist_sq > 1.0f) {
        out_density = 0.0f;
        out_color = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    // Inside the ellipsoid - compute Gaussian density
    // Using Gaussian falloff: exp(-0.5 * r^2) where r is normalized distance
    float gaussian_weight = expf(-0.5f * dist_sq);
    out_density = opacity * gaussian_weight;

    // Evaluate spherical harmonics for color
    // For simplicity, just use DC component (first SH coefficient)
    // Full SH evaluation is done in the Slang shader
    out_color = make_float3(
        features[0],  // R
        features[1],  // G
        features[2]   // B
    );
}

__global__ void kernel_initialize_render_state(
    const float3* __restrict__ ray_origins,
    const float3* __restrict__ ray_directions,
    const float3* __restrict__ positions,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float* __restrict__ features,
    uint32_t num_rays,
    uint32_t num_elements,
    uint32_t feature_dim,
    float t_min,
    RenderState* __restrict__ states,
    float4* __restrict__ initial_contributions
) {
    uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    float3 ray_origin = ray_origins[ray_idx];
    float3 ray_direction = ray_directions[ray_idx];

    // Initialize render state
    RenderState state;
    state.log_transmittance = 0.0f;  // T = exp(0) = 1
    state.accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    state.accumulated_depth = 0.0f;
    state.depth_weight = 0.0f;
    state.t_current = t_min;
    state.distortion_accum = make_float2(0.0f, 0.0f);
    state.weight_accum = make_float2(0.0f, 0.0f);

    // Accumulate contributions from ellipsoids containing the ray origin
    float4 initial = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t i = 0; i < num_elements; ++i) {
        if (point_inside_ellipsoid(ray_origin, positions[i], scales[i], rotations[i])) {
            float density;
            float3 color;
            compute_initial_contribution(
                ray_origin,
                ray_direction,
                positions[i],
                scales[i],
                rotations[i],
                opacities[i],
                features + i * feature_dim,
                0,  // Use DC component only for initialization
                density,
                color
            );

            initial.x += density;
            initial.y += density * color.x;
            initial.z += density * color.y;
            initial.w += density * color.z;
        }
    }

    states[ray_idx] = state;
    initial_contributions[ray_idx] = initial;
}

void launch_initialize_render_state(
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
) {
    constexpr uint32_t BLOCK_SIZE = 256;
    uint32_t num_blocks = (num_rays + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_initialize_render_state<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        ray_origins,
        ray_directions,
        positions,
        scales,
        rotations,
        opacities,
        features,
        num_rays,
        num_elements,
        feature_dim,
        t_min,
        states,
        initial_contributions
    );
}

} // namespace gaussian_rt
