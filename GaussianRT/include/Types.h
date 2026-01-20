// GaussianRT - Type Definitions
// Hardware ray tracing for Gaussian/ellipsoid volume rendering
// Apache License 2.0

#pragma once

#include <cstdint>
#include <cstddef>

namespace gaussianrt {

using uint = uint32_t;

// 2D vector type
struct float2 {
    float x, y;
    float2() : x(0), y(0) {}
    float2(float x, float y) : x(x), y(y) {}
};

// 3D vector type
struct alignas(16) float3 {
    float x, y, z;
    float _pad;

    float3() : x(0), y(0), z(0), _pad(0) {}
    float3(float x, float y, float z) : x(x), y(y), z(z), _pad(0) {}
};

// 4D vector type
struct alignas(16) float4 {
    float x, y, z, w;

    float4() : x(0), y(0), z(0), w(0) {}
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

// Axis-aligned bounding box
struct AABB {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

// Camera parameters - matches GPU layout
struct alignas(16) Camera {
    float fx, fy;           // Focal lengths
    int width, height;      // Image dimensions
    float2 _pad0;           // Padding for alignment
    float3 U, V, W;         // Camera basis vectors
    float3 eye;             // Camera position
};

// Volume rendering state - for tracking ray march progress
struct alignas(16) VolumeState {
    float2 distortion_parts;
    float2 cum_sum;
    float3 padding;
    float t;                // Current ray parameter
    float4 drgb;            // Accumulated density and color gradients
    float logT;             // Log transmittance
    float3 C;               // Accumulated color
};

// Ellipsoid primitive data
struct Ellipsoid {
    float3 mean;            // Center position
    float3 scale;           // Semi-axis lengths
    float4 quaternion;      // Rotation quaternion
    float density;          // Volume density
};

// Primitive collection on GPU
struct Primitives {
    float3* means;
    float3* scales;
    float4* quats;
    float* densities;
    float* features;
    AABB* aabbs;
    size_t count;
    size_t feature_size;
};

// Ray tracing parameters
struct TraceParams {
    uint sh_degree;         // Spherical harmonics degree
    uint max_iters;         // Maximum ray march iterations
    float tmin;             // Near plane
    float tmax;             // Far plane
    float max_prim_size;    // Maximum primitive size
};

// Rendering output
struct RenderOutput {
    float4* image;          // RGBA output
    uint* iteration_counts; // Debug: iterations per pixel
    uint* touch_counts;     // Debug: primitives touched
    VolumeState* states;    // Final volume states
};

// Forward pass saved state for backward
struct ForwardSavedState {
    VolumeState* states;        // Final volume states per ray
    float4* diracs;             // Final dirac values per ray
    int* iters;                 // Iteration counts per ray
    int* tri_collection;        // Triangle hit history (ray_count * max_iters)
    float4* initial_drgb;       // Initial drgb values
    int* initial_touch_inds;    // Initial touch indices
    int initial_touch_count;    // Number of initial touches
    size_t ray_count;
    size_t max_iters;
};

// Gradients for backward pass
struct Gradients {
    float3* dL_dmeans;          // Gradient w.r.t. means
    float3* dL_dscales;         // Gradient w.r.t. scales
    float4* dL_dquats;          // Gradient w.r.t. quaternions
    float* dL_ddensities;       // Gradient w.r.t. densities
    float* dL_dfeatures;        // Gradient w.r.t. features
    float3* dL_dray_origins;    // Gradient w.r.t. ray origins
    float3* dL_dray_dirs;       // Gradient w.r.t. ray directions
    float2* dL_dmeans2D;        // Gradient w.r.t. 2D means (for splatting)
};

// Backward pass input
struct BackwardInput {
    float* dL_doutputs;         // Gradient w.r.t. outputs (ray_count x 5)
    float* wcts;                // World-to-clip transforms (for 2D gradient)
    ForwardSavedState saved;    // Saved state from forward pass
};

} // namespace gaussianrt
