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

// Safe normalize for quaternion
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

}  // namespace

// =============================================================================
// AABB Computation
// =============================================================================

__global__ void compute_primitive_bounds_kernel(
    const float* __restrict__ means,
    const float* __restrict__ scales,
    const float* __restrict__ quats,
    int num_prims,
    OptixAabb* __restrict__ aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_prims) return;

    // Load float3 from scalar array
    float3 center = make_float3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]);
    float3 size = make_float3(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
    // Load float4 quaternion from scalar array
    float4 quat = make_float4(quats[i * 4 + 0], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]);
    mat3 Rt = quat_to_rotation_matrix_t(quat);

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
    int grid = (prims.num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_primitive_bounds_kernel<<<grid, BLOCK_SIZE>>>(
        prims.means,
        prims.scales,
        prims.quats,
        prims.num_prims, aabbs);
    CUDA_SYNC_CHECK();
}
