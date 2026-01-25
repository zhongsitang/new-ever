// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#include "ray_tracer.h"
#include <cuda_runtime.h>
#include "cuda_math.h"  // dot, length, normalize, etc.

// =============================================================================
// Constants
// =============================================================================

namespace {
    constexpr size_t BLOCK_SIZE = 1024;
    constexpr size_t RAY_BLOCK  = 64;
    constexpr size_t HIT_BLOCK  = 16;
    __device__ constexpr float SH_C0 = 0.28209479177387814f;
}

// =============================================================================
// Safe Tensor Access Helpers
// =============================================================================
// These functions safely load float3/float4 from contiguous float arrays,
// avoiding unsafe reinterpret_cast from torch tensors to CUDA vector types.

__device__ __forceinline__
float3 load_float3(const float* __restrict__ data, size_t idx) {
    const float* p = data + idx * 3;
    return make_float3(p[0], p[1], p[2]);
}

__device__ __forceinline__
float4 load_float4(const float* __restrict__ data, size_t idx) {
    const float* p = data + idx * 4;
    return make_float4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__
void store_float3(float* __restrict__ data, size_t idx, float3 v) {
    float* p = data + idx * 3;
    p[0] = v.x; p[1] = v.y; p[2] = v.z;
}

__device__ __forceinline__
void store_float4(float* __restrict__ data, size_t idx, float4 v) {
    float* p = data + idx * 4;
    p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
}

// =============================================================================
// Mat3 (column-major)
// =============================================================================

struct mat3 {
    float3 c0, c1, c2;

    __host__ __device__ __forceinline__ mat3() {}

    // From three column vectors
    __host__ __device__ __forceinline__
    mat3(float3 col0, float3 col1, float3 col2)
        : c0(col0), c1(col1), c2(col2) {}

    // Row-major input, column-major storage
    __host__ __device__ __forceinline__
    mat3(float m00, float m01, float m02,
         float m10, float m11, float m12,
         float m20, float m21, float m22)
        : c0(make_float3(m00, m10, m20)),
          c1(make_float3(m01, m11, m21)),
          c2(make_float3(m02, m12, m22)) {}
};

__host__ __device__ __forceinline__
float3 operator*(const mat3& M, const float3& v) {
    return M.c0 * v.x + M.c1 * v.y + M.c2 * v.z;
}

__host__ __device__ __forceinline__
mat3 operator*(const mat3& A, const mat3& B) {
    return mat3{A * B.c0, A * B.c1, A * B.c2};
}

// =============================================================================
// Transform Utilities
// =============================================================================

namespace {

// Safe normalize (returns zero/identity for zero-length input)
__host__ __device__ __forceinline__
float3 safe_normalize(const float3& v) {
    float len2 = dot(v, v);
    return len2 > 0.f ? v * rsqrtf(len2) : make_float3(0.f);
}

__host__ __device__ __forceinline__
float4 safe_normalize(const float4& q) {
    float len2 = dot(q, q);
    return len2 > 0.f ? q * rsqrtf(len2) : make_float4(1.f, 0.f, 0.f, 0.f);
}

// Quaternion (wxyz: w=q.x, x=q.y, y=q.z, z=q.w) -> transposed rotation matrix
__device__ __forceinline__
mat3 quat_to_rotation_matrix_t(const float4& quat) {
    float4 q = safe_normalize(quat);
    float w = q.x, x = q.y, y = q.z, z = q.w;

    return mat3{
        1.f - 2.f*(y*y + z*z),  2.f*(x*y + w*z),        2.f*(x*z - w*y),
        2.f*(x*y - w*z),        1.f - 2.f*(x*x + z*z),  2.f*(y*z + w*x),
        2.f*(x*z + w*y),        2.f*(y*z - w*x),        1.f - 2.f*(x*x + y*y)
    };
}

// World -> ellipsoid local space (normalized: unit sphere = surface)
__device__ __forceinline__
float3 world_to_ellipsoid(const float3& point, const float3& center,
                          const float4& quat, const float3& scale) {
    mat3 Rt = quat_to_rotation_matrix_t(quat);
    float3 p = Rt * (point - center);
    return make_float3(p.x / scale.x, p.y / scale.y, p.z / scale.z);
}

// Compute (density, density*r, density*g, density*b)
__device__ __forceinline__
float4 compute_contribution(float density, const float* features) {
    float r = features[0] * SH_C0 + 0.5f;
    float g = features[1] * SH_C0 + 0.5f;
    float b = features[2] * SH_C0 + 0.5f;
    return make_float4(density, density * r, density * g, density * b);
}

}  // namespace

// =============================================================================
// AABB Computation
// =============================================================================

__global__ void compute_primitive_bounds_kernel(
    const float* __restrict__ means,
    const float* __restrict__ scales,
    const float* __restrict__ quats,
    size_t num_prims,
    OptixAabb* __restrict__ aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)num_prims) return;

    float3 center = load_float3(means, i);
    float3 size = load_float3(scales, i);
    mat3 Rt = quat_to_rotation_matrix_t(load_float4(quats, i));

    // M = S * Rt, extents = column norms
    mat3 S(size.x, 0.f, 0.f, 0.f, size.y, 0.f, 0.f, 0.f, size.z);
    mat3 M = S * Rt;

    float ex = length(M.c0);
    float ey = length(M.c1);
    float ez = length(M.c2);

    aabbs[i] = OptixAabb{
        center.x - ex, center.y - ey, center.z - ez,
        center.x + ex, center.y + ey, center.z + ez
    };
}

void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs) {
    size_t grid = (prims.num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_primitive_bounds_kernel<<<grid, BLOCK_SIZE>>>(
        prims.means, prims.scales, prims.quats,
        prims.num_prims, aabbs);
    CUDA_SYNC_CHECK();
}

// =============================================================================
// Initial Sample Accumulation
// =============================================================================

namespace {

// Find primitives within tmin distance of ray origin
__global__ void find_enclosing_primitives_kernel(
    const OptixAabb* __restrict__ aabbs,
    size_t num_prims,
    float tmin,
    const float* __restrict__ ray_origins,
    int* __restrict__ hit_indices,
    int* __restrict__ hit_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)num_prims) return;

    OptixAabb aabb = aabbs[i];
    float3 o = load_float3(ray_origins, 0);
    float tmin_sq = tmin * tmin;

    // Squared distance point-to-AABB (Arvo, Graphics Gems)
    float d2 = 0.f;
    if      (o.x < aabb.minX) d2 += (o.x - aabb.minX) * (o.x - aabb.minX);
    else if (o.x > aabb.maxX) d2 += (o.x - aabb.maxX) * (o.x - aabb.maxX);
    if      (o.y < aabb.minY) d2 += (o.y - aabb.minY) * (o.y - aabb.minY);
    else if (o.y > aabb.maxY) d2 += (o.y - aabb.maxY) * (o.y - aabb.maxY);
    if      (o.z < aabb.minZ) d2 += (o.z - aabb.minZ) * (o.z - aabb.minZ);
    else if (o.z > aabb.maxZ) d2 += (o.z - aabb.maxZ) * (o.z - aabb.maxZ);

    if (d2 <= tmin_sq) {
        hit_indices[atomicAdd(hit_count, 1)] = i;
    }
}

// Accumulate contributions for rays inside primitives (multi-ray)
__global__ void accumulate_initial_samples_kernel(
    const float* __restrict__ means,
    const float* __restrict__ scales,
    const float* __restrict__ quats,
    const float* __restrict__ densities,
    const float* __restrict__ features,
    size_t num_rays,
    float tmin,
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_directions,
    float* __restrict__ initial_contrib,
    const int* __restrict__ hit_indices,
    const int* __restrict__ hit_count)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hit_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (hit_idx >= *hit_count || ray_idx >= (int)num_rays) return;

    float3 ray_o = load_float3(ray_origins, ray_idx);
    float3 ray_d = load_float3(ray_directions, ray_idx);
    float3 ray_start = ray_o + tmin * safe_normalize(ray_d);
    int prim = hit_indices[hit_idx];
    float3 local = world_to_ellipsoid(ray_start, load_float3(means, prim), load_float4(quats, prim), load_float3(scales, prim));

    if (dot(local, local) <= 1.f) {
        float4 c = compute_contribution(densities[prim], &features[prim * 3]);
        atomicAdd(initial_contrib + 4 * ray_idx + 0, c.x);
        atomicAdd(initial_contrib + 4 * ray_idx + 1, c.y);
        atomicAdd(initial_contrib + 4 * ray_idx + 2, c.z);
        atomicAdd(initial_contrib + 4 * ray_idx + 3, c.w);
    }
}

// Single-ray version: check all primitives directly
__global__ void accumulate_initial_samples_single_kernel(
    const float* __restrict__ means,
    const float* __restrict__ scales,
    const float* __restrict__ quats,
    const float* __restrict__ densities,
    const float* __restrict__ features,
    size_t num_prims,
    const float* __restrict__ ray_origin,
    float* __restrict__ initial_contrib)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)num_prims) return;

    float3 o = load_float3(ray_origin, 0);
    float3 local = world_to_ellipsoid(o, load_float3(means, i), load_float4(quats, i), load_float3(scales, i));

    if (dot(local, local) <= 1.f) {
        float4 c = compute_contribution(densities[i], &features[i * 3]);
        atomicAdd(initial_contrib + 0, c.x);
        atomicAdd(initial_contrib + 1, c.y);
        atomicAdd(initial_contrib + 2, c.z);
        atomicAdd(initial_contrib + 3, c.w);
    }
}

}  // namespace

// =============================================================================
// Public API
// =============================================================================

void init_ray_start_samples(
    const Primitives& prims,
    size_t num_rays,
    float tmin,
    const float* ray_origins,
    const float* ray_directions,
    float* initial_contrib,
    OptixAabb* aabbs,
    int* d_hit_count,
    int* d_hit_inds)
{
    size_t num_prims = prims.num_prims;

    bool alloc_temp = (d_hit_count == nullptr);
    if (alloc_temp) {
        cudaMalloc(&d_hit_inds,  num_prims * sizeof(int));
        cudaMalloc(&d_hit_count, sizeof(int));
    }
    cudaMemset(d_hit_count, 0, sizeof(int));

    // Phase 1: find enclosing primitives
    size_t grid = (num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_enclosing_primitives_kernel<<<grid, BLOCK_SIZE>>>(
        aabbs, num_prims, tmin,
        ray_origins,
        d_hit_inds, d_hit_count);

    int hit_count = 0;
    cudaMemcpy(&hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Phase 2: accumulate contributions
    if (hit_count > 0) {
        dim3 grid2((num_rays + RAY_BLOCK - 1) / RAY_BLOCK,
                   (hit_count + HIT_BLOCK - 1) / HIT_BLOCK);
        dim3 block(RAY_BLOCK, HIT_BLOCK);

        accumulate_initial_samples_kernel<<<grid2, block>>>(
            prims.means, prims.scales, prims.quats,
            prims.densities, prims.features,
            num_rays, tmin,
            ray_origins, ray_directions,
            initial_contrib,
            d_hit_inds, d_hit_count);
        CUDA_SYNC_CHECK();
    }

    if (alloc_temp) {
        cudaFree(d_hit_inds);
        cudaFree(d_hit_count);
    }
}
