// OptiX curve shaders for volume rendering
// Compile with: nvcc -ptx -I<OptiX_SDK>/include curve_shaders.cu -o curve_shaders.ptx

#include <optix.h>
#include <cuda_runtime.h>

// Launch parameters (set by host)
struct LaunchParams {
    // Output buffers
    float4* colors;          // [num_rays]
    float* depths;           // [num_rays]
    int* hit_indices;        // [num_rays]
    float* hit_params;       // [num_rays] - curve parameter u

    // Input buffers
    float3* ray_origins;     // [num_rays]
    float3* ray_directions;  // [num_rays]

    // Curve attributes
    float* curve_colors;     // [num_curves, 3]
    float* curve_opacities;  // [num_curves]

    // Scene
    OptixTraversableHandle traversable;

    // Parameters
    float t_min;
    float t_max;
    uint32_t num_rays;
};

extern "C" {
    __constant__ LaunchParams params;
}

// Ray payload
struct RayPayload {
    float3 color;
    float alpha;
    float depth;
    int hit_index;
    float hit_param;
};

// ============================================================================
// Ray Generation
// ============================================================================

extern "C" __global__ void __raygen__curve() {
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t ray_idx = idx.x;

    if (ray_idx >= params.num_rays) return;

    // Get ray
    float3 origin = params.ray_origins[ray_idx];
    float3 direction = params.ray_directions[ray_idx];

    // Initialize payload
    RayPayload payload;
    payload.color = make_float3(0.0f, 0.0f, 0.0f);
    payload.alpha = 0.0f;
    payload.depth = 0.0f;
    payload.hit_index = -1;
    payload.hit_param = 0.0f;

    // Pack payload pointer
    uint32_t p0, p1;
    uint64_t ptr = reinterpret_cast<uint64_t>(&payload);
    p0 = static_cast<uint32_t>(ptr);
    p1 = static_cast<uint32_t>(ptr >> 32);

    // Trace ray
    optixTrace(
        params.traversable,
        origin,
        direction,
        params.t_min,
        params.t_max,
        0.0f,                          // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                             // SBT offset
        1,                             // SBT stride
        0,                             // miss SBT index
        p0, p1
    );

    // Write outputs
    params.colors[ray_idx] = make_float4(payload.color.x, payload.color.y, payload.color.z, payload.alpha);
    params.depths[ray_idx] = payload.depth;
    params.hit_indices[ray_idx] = payload.hit_index;
    params.hit_params[ray_idx] = payload.hit_param;
}

// ============================================================================
// Closest Hit - Curves
// ============================================================================

extern "C" __global__ void __closesthit__curve() {
    // Get payload
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    uint64_t ptr = (static_cast<uint64_t>(p1) << 32) | p0;
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);

    // Get curve parameter u ∈ [0, 1] along the segment
    const float u = optixGetCurveParameter();

    // Get primitive (segment) index
    const uint32_t prim_idx = optixGetPrimitiveIndex();

    // Get hit distance
    const float t_hit = optixGetRayTmax();

    // Get curve control points based on curve type
    // For cubic B-spline:
    float4 cp[4];
    optixGetCubicBSplineVertexData(cp);

    // Compute position on curve (de Boor's algorithm already done by OptiX)
    // We can use the control points for attribute interpolation

    // Get curve attributes
    float3 curve_color = make_float3(
        params.curve_colors[prim_idx * 3 + 0],
        params.curve_colors[prim_idx * 3 + 1],
        params.curve_colors[prim_idx * 3 + 2]
    );
    float curve_opacity = params.curve_opacities[prim_idx];

    // Volume rendering: accumulate contribution
    // Simple alpha blending for now
    float alpha = 1.0f - expf(-curve_opacity);
    float weight = alpha * (1.0f - payload->alpha);

    payload->color.x += weight * curve_color.x;
    payload->color.y += weight * curve_color.y;
    payload->color.z += weight * curve_color.z;
    payload->alpha += weight;
    payload->depth = t_hit;
    payload->hit_index = prim_idx;
    payload->hit_param = u;
}

// ============================================================================
// Any Hit - for transparency/volume integration
// ============================================================================

extern "C" __global__ void __anyhit__curve() {
    // Get payload
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    uint64_t ptr = (static_cast<uint64_t>(p1) << 32) | p0;
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);

    // Early termination if ray is saturated
    if (payload->alpha > 0.99f) {
        optixTerminateRay();
        return;
    }

    // Accept hit and continue to closest hit
    // (or use optixIgnoreIntersection() to continue through)
}

// ============================================================================
// Miss
// ============================================================================

extern "C" __global__ void __miss__curve() {
    // Get payload
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    uint64_t ptr = (static_cast<uint64_t>(p1) << 32) | p0;
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);

    // Background color (black)
    // Could blend with remaining transmittance
    float3 bg = make_float3(0.0f, 0.0f, 0.0f);
    float remaining = 1.0f - payload->alpha;
    payload->color.x += remaining * bg.x;
    payload->color.y += remaining * bg.y;
    payload->color.z += remaining * bg.z;
}

// ============================================================================
// Alternative: Linear curve (capsule) closest hit
// ============================================================================

extern "C" __global__ void __closesthit__linear_curve() {
    // Get payload
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    uint64_t ptr = (static_cast<uint64_t>(p1) << 32) | p0;
    RayPayload* payload = reinterpret_cast<RayPayload*>(ptr);

    // Get curve parameter
    const float u = optixGetCurveParameter();
    const uint32_t prim_idx = optixGetPrimitiveIndex();
    const float t_hit = optixGetRayTmax();

    // Get linear curve (capsule) data: 2 float4 values
    float4 data[2];
    optixGetLinearCurveVertexData(data);
    // data[0] = (x0, y0, z0, r0) - start point and radius
    // data[1] = (x1, y1, z1, r1) - end point and radius

    // Interpolate position
    float3 p0_pos = make_float3(data[0].x, data[0].y, data[0].z);
    float3 p1_pos = make_float3(data[1].x, data[1].y, data[1].z);
    float3 hit_pos = make_float3(
        p0_pos.x * (1.0f - u) + p1_pos.x * u,
        p0_pos.y * (1.0f - u) + p1_pos.y * u,
        p0_pos.z * (1.0f - u) + p1_pos.z * u
    );

    // Interpolate radius
    float r = data[0].w * (1.0f - u) + data[1].w * u;

    // Get attributes and render
    float3 curve_color = make_float3(
        params.curve_colors[prim_idx * 3 + 0],
        params.curve_colors[prim_idx * 3 + 1],
        params.curve_colors[prim_idx * 3 + 2]
    );
    float curve_opacity = params.curve_opacities[prim_idx];

    float alpha = 1.0f - expf(-curve_opacity);
    float weight = alpha * (1.0f - payload->alpha);

    payload->color.x += weight * curve_color.x;
    payload->color.y += weight * curve_color.y;
    payload->color.z += weight * curve_color.z;
    payload->alpha += weight;
    payload->depth = t_hit;
    payload->hit_index = prim_idx;
    payload->hit_param = u;
}
