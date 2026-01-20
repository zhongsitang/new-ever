#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gaussian_rt {

// ============================================================================
// Volume Element Types (Primitives)
// ============================================================================

// Base properties shared by all volume element types
struct VolumeElementBase {
    float3 position;    // Center position in world space
    float opacity;      // Opacity/density coefficient (sigma)
};

// Ellipsoid-specific properties
struct EllipsoidParams {
    float3 scale;       // Semi-axes lengths (a, b, c)
    float4 rotation;    // Rotation quaternion (w, x, y, z)
};

// Complete ellipsoid volume element
struct EllipsoidElement {
    VolumeElementBase base;
    EllipsoidParams params;
    uint32_t feature_offset;  // Offset into feature buffer (for SH coefficients)
};

// ============================================================================
// Ray and Intersection Types
// ============================================================================

struct Ray {
    float3 origin;
    float3 direction;
    float t_min;
    float t_max;
};

// Intersection result
struct RayHit {
    float t_near;           // Entry distance
    float t_far;            // Exit distance
    uint32_t element_id;    // Index of hit element
    bool valid;
};

// ============================================================================
// Volume Rendering State
// ============================================================================

// Accumulated state along a ray during volume rendering
struct RenderState {
    // Transmittance and color accumulation
    float log_transmittance;    // log(T) where T = exp(-integral of sigma*dt)
    float3 accumulated_color;   // C = integral of T * sigma * c dt

    // For depth computation
    float accumulated_depth;    // Weighted depth
    float depth_weight;

    // Current ray parameter
    float t_current;

    // Distortion loss components (for regularization)
    float2 distortion_accum;
    float2 weight_accum;
};

// Control point: represents a sample along the ray
struct ControlPoint {
    float t;                // Distance along ray
    float sigma;            // Density at this point
    float3 color;           // Color at this point (from SH evaluation)
    uint32_t element_id;    // Source element
};

// ============================================================================
// Spherical Harmonics
// ============================================================================

static constexpr uint32_t SH_MAX_DEGREE = 3;
static constexpr uint32_t SH_COEFF_COUNT = (SH_MAX_DEGREE + 1) * (SH_MAX_DEGREE + 1);  // 16

struct SHCoefficients {
    float3 coeffs[SH_COEFF_COUNT];  // 16 RGB triplets for degree-3 SH
};

// ============================================================================
// Scene Data (GPU Buffers)
// ============================================================================

struct SceneData {
    // Element data
    float3* positions;          // [N] Center positions
    float3* scales;             // [N] Scale factors
    float4* rotations;          // [N] Rotation quaternions
    float* opacities;           // [N] Opacity values
    float* features;            // [N * feature_dim] SH coefficients

    // Acceleration structure data
    void* aabbs;                // OptixAabb or equivalent

    // Counts
    uint32_t num_elements;
    uint32_t feature_dim;       // Typically 16*3 for SH degree 3
};

// ============================================================================
// Rendering Parameters
// ============================================================================

struct RenderParams {
    uint32_t width;
    uint32_t height;
    float t_min;
    float t_max;
    uint32_t max_samples_per_ray;   // Maximum intersections to process
    uint32_t sh_degree;             // SH evaluation degree (0-3)
    float transmittance_threshold;  // Early termination threshold
};

// Camera parameters
struct Camera {
    float3 position;
    float3 forward;
    float3 up;
    float3 right;
    float focal_length;
    float sensor_width;
    float sensor_height;
};

// ============================================================================
// Gradient Buffers (for backward pass)
// ============================================================================

struct GradientBuffers {
    float3* d_positions;    // [N]
    float3* d_scales;       // [N]
    float4* d_rotations;    // [N]
    float* d_opacities;     // [N]
    float* d_features;      // [N * feature_dim]
    float3* d_ray_origins;  // [num_rays]
    float3* d_ray_dirs;     // [num_rays]
};

// ============================================================================
// Forward Pass Output
// ============================================================================

struct ForwardOutput {
    float4* colors;                 // [H*W] RGBA output
    RenderState* final_states;      // [H*W] Final render states
    ControlPoint* last_points;      // [H*W] Last control points
    uint32_t* sample_counts;        // [H*W] Samples per ray
    int32_t* sample_indices;        // [H*W * max_samples] Element indices
    uint32_t* element_touch_counts; // [N] Per-element hit counts
};

// ============================================================================
// Utility Functions
// ============================================================================

inline __host__ __device__ float3 make_float3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}

inline __host__ __device__ float4 make_float4(float x, float y, float z, float w) {
    float4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ float length(float3 v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 normalize(float3 v) {
    float len = length(v);
    return len > 0.0f ? v * (1.0f / len) : v;
}

// Quaternion rotation
inline __host__ __device__ float3 rotate_by_quaternion(float3 v, float4 q) {
    // q = (w, x, y, z)
    float3 u = make_float3(q.y, q.z, q.w);
    float s = q.x;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

inline __host__ __device__ float3 rotate_by_quaternion_inverse(float3 v, float4 q) {
    // Conjugate: (w, -x, -y, -z)
    float4 q_conj = make_float4(q.x, -q.y, -q.z, -q.w);
    return rotate_by_quaternion(v, q_conj);
}

} // namespace gaussian_rt
