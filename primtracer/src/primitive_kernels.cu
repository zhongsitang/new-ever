// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray_tracer.h"
#include "glm/glm.hpp"

// =============================================================================
// Device Utilities
// =============================================================================

namespace {

constexpr size_t BLOCK_SIZE = 1024;

/// Spherical harmonics constant for DC term
__device__ constexpr float SH_C0 = 0.28209479177387814f;

/// Compute rotation matrix from quaternion (wxyz format).
/// Returns transposed rotation matrix for efficient column-major access.
__device__ glm::mat3 quat_to_rotation_matrix_t(const glm::vec4& quat) {
    const glm::vec4 q = glm::normalize(quat);
    const float w = q.x, x = q.y, y = q.z, z = q.w;

    return glm::mat3{
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z), 2.0f * (x * z + w * y),
        2.0f * (x * y + w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
        2.0f * (x * z - w * y), 2.0f * (y * z + w * x), 1.0f - 2.0f * (x * x + y * y)
    };
}

/// Transform point from world space to ellipsoid-local space.
/// Returns normalized coordinates where unit sphere = ellipsoid surface.
__device__ glm::vec3 world_to_ellipsoid(
    const glm::vec3& point,
    const glm::vec3& center,
    const glm::vec4& quat,
    const glm::vec3& scale)
{
    const glm::mat3 Rt = quat_to_rotation_matrix_t(quat);
    return (Rt * (point - center)) / scale;
}

/// Squared distance from origin (for point-in-ellipsoid test)
__device__ float squared_norm(const glm::vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

}  // namespace

// =============================================================================
// AABB Computation
// =============================================================================

__global__ void compute_primitive_bounds_kernel(
    const glm::vec3* __restrict__ means,
    const glm::vec3* __restrict__ scales,
    const glm::vec4* __restrict__ quats,
    const size_t num_prims,
    OptixAabb* __restrict__ aabbs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_prims) return;

    const glm::vec3 center = means[i];
    const glm::vec3 size = scales[i];
    const glm::mat3 Rt = quat_to_rotation_matrix_t(quats[i]);

    // Compute transformation matrix M = S * R^T
    const glm::mat3 S = glm::mat3(
        size.x, 0.0f, 0.0f,
        0.0f, size.y, 0.0f,
        0.0f, 0.0f, size.z
    );
    const glm::mat3 M = S * Rt;

    // AABB extent = row norms of M (maximum extent along each axis)
    const float extent_x = sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2]);
    const float extent_y = sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2]);
    const float extent_z = sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2]);

    aabbs[i] = OptixAabb{
        center.x - extent_x, center.y - extent_y, center.z - extent_z,
        center.x + extent_x, center.y + extent_y, center.z + extent_z
    };
}

void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs) {
    const size_t grid_size = (prims.num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_primitive_bounds_kernel<<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const glm::vec3*>(prims.means),
        reinterpret_cast<const glm::vec3*>(prims.scales),
        reinterpret_cast<const glm::vec4*>(prims.quats),
        prims.num_prims,
        aabbs);
    CUDA_SYNC_CHECK();
}

// =============================================================================
// Initial Sample Accumulation (rays starting inside primitives)
// =============================================================================

namespace {

/// Find primitives whose AABB contains the ray origin (within tmin distance).
__global__ void find_enclosing_primitives_kernel(
    const OptixAabb* __restrict__ aabbs,
    const size_t num_prims,
    const float tmin,
    const glm::vec3* __restrict__ ray_origins,
    int* __restrict__ hit_indices,
    int* __restrict__ hit_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_prims) return;

    const OptixAabb aabb = aabbs[i];
    const glm::vec3 rayo = ray_origins[0];
    const float tmin_sq = tmin * tmin;

    // Squared distance from point to AABB (Jim Arvo, Graphics Gems)
    float dist_sq = 0.0f;
    if (rayo.x < aabb.minX) dist_sq += (rayo.x - aabb.minX) * (rayo.x - aabb.minX);
    else if (rayo.x > aabb.maxX) dist_sq += (rayo.x - aabb.maxX) * (rayo.x - aabb.maxX);

    if (rayo.y < aabb.minY) dist_sq += (rayo.y - aabb.minY) * (rayo.y - aabb.minY);
    else if (rayo.y > aabb.maxY) dist_sq += (rayo.y - aabb.maxY) * (rayo.y - aabb.maxY);

    if (rayo.z < aabb.minZ) dist_sq += (rayo.z - aabb.minZ) * (rayo.z - aabb.minZ);
    else if (rayo.z > aabb.maxZ) dist_sq += (rayo.z - aabb.maxZ) * (rayo.z - aabb.maxZ);

    if (dist_sq <= tmin_sq) {
        const int pos = atomicAdd(hit_count, 1);
        hit_indices[pos] = i;
    }
}

/// Accumulate density/color contributions for rays inside primitives.
__global__ void accumulate_initial_samples_kernel(
    const glm::vec3* __restrict__ means,
    const glm::vec3* __restrict__ scales,
    const glm::vec4* __restrict__ quats,
    const float* __restrict__ densities,
    const float* __restrict__ features,
    const size_t num_rays,
    const float tmin,
    const glm::vec3* __restrict__ ray_origins,
    const glm::vec3* __restrict__ ray_directions,
    float* __restrict__ initial_contrib,
    const int* __restrict__ hit_indices,
    const int* __restrict__ hit_count)
{
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hit_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (hit_idx >= *hit_count || ray_idx >= num_rays) return;

    // Get ray start point (offset by tmin along direction)
    const glm::vec3 ray_start = ray_origins[ray_idx] + tmin * glm::normalize(ray_directions[ray_idx]);

    // Get primitive data
    const int prim_idx = hit_indices[hit_idx];
    const glm::vec3 local_pos = world_to_ellipsoid(ray_start, means[prim_idx], quats[prim_idx], scales[prim_idx]);

    // Check if point is inside ellipsoid (|local_pos| <= 1)
    if (squared_norm(local_pos) <= 1.0f) {
        const float density = densities[prim_idx];
        const glm::vec3 color = {
            features[prim_idx * 3 + 0] * SH_C0 + 0.5f,
            features[prim_idx * 3 + 1] * SH_C0 + 0.5f,
            features[prim_idx * 3 + 2] * SH_C0 + 0.5f,
        };

        // Accumulate: (density, density*r, density*g, density*b)
        atomicAdd(initial_contrib + 4 * ray_idx + 0, density);
        atomicAdd(initial_contrib + 4 * ray_idx + 1, density * color.x);
        atomicAdd(initial_contrib + 4 * ray_idx + 2, density * color.y);
        atomicAdd(initial_contrib + 4 * ray_idx + 3, density * color.z);
    }
}

/// Single-ray version: check all primitives directly
__global__ void accumulate_initial_samples_single_kernel(
    const glm::vec3* __restrict__ means,
    const glm::vec3* __restrict__ scales,
    const glm::vec4* __restrict__ quats,
    const float* __restrict__ densities,
    const float* __restrict__ features,
    const size_t num_prims,
    const glm::vec3* __restrict__ ray_origin,
    float* __restrict__ initial_contrib)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_prims) return;

    const glm::vec3 local_pos = world_to_ellipsoid(*ray_origin, means[i], quats[i], scales[i]);

    if (squared_norm(local_pos) <= 1.0f) {
        const float density = densities[i];
        const glm::vec3 color = {
            features[i * 3 + 0] * SH_C0 + 0.5f,
            features[i * 3 + 1] * SH_C0 + 0.5f,
            features[i * 3 + 2] * SH_C0 + 0.5f,
        };

        atomicAdd(initial_contrib + 0, density);
        atomicAdd(initial_contrib + 1, density * color.x);
        atomicAdd(initial_contrib + 2, density * color.y);
        atomicAdd(initial_contrib + 3, density * color.z);
    }
}

}  // namespace

// =============================================================================
// Public API
// =============================================================================

void init_ray_start_samples(Params* params, OptixAabb* aabbs, int* d_hit_count, int* d_hit_inds) {
    const int num_prims = params->means.count / 3;  // means uses scalar packing (xyzxyz...)
    const int num_rays = params->initial_contrib.count;

    // Allocate temporary buffers if not provided
    const bool alloc_temp = (d_hit_count == nullptr);
    if (alloc_temp) {
        cudaMalloc(&d_hit_inds, num_prims * sizeof(int));
        cudaMalloc(&d_hit_count, sizeof(int));
    }
    cudaMemset(d_hit_count, 0, sizeof(int));

    // Phase 1: Find primitives near ray origins
    const size_t grid_size = (num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_enclosing_primitives_kernel<<<grid_size, BLOCK_SIZE>>>(
        aabbs, num_prims, params->tmin,
        reinterpret_cast<const glm::vec3*>(params->ray_origins.data),
        d_hit_inds, d_hit_count);

    // Get hit count
    int hit_count;
    cudaMemcpy(&hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Phase 2: Accumulate contributions
    if (hit_count > 0) {
        constexpr size_t RAY_BLOCK = 64;
        constexpr size_t HIT_BLOCK = 16;
        dim3 grid((num_rays + RAY_BLOCK - 1) / RAY_BLOCK, (hit_count + HIT_BLOCK - 1) / HIT_BLOCK);
        dim3 block(RAY_BLOCK, HIT_BLOCK);

        accumulate_initial_samples_kernel<<<grid, block>>>(
            reinterpret_cast<const glm::vec3*>(params->means.data),
            reinterpret_cast<const glm::vec3*>(params->scales.data),
            reinterpret_cast<const glm::vec4*>(params->quats.data),
            params->densities.data,
            params->features.data,
            num_rays, params->tmin,
            reinterpret_cast<const glm::vec3*>(params->ray_origins.data),
            reinterpret_cast<const glm::vec3*>(params->ray_directions.data),
            reinterpret_cast<float*>(params->initial_contrib.data),
            d_hit_inds, d_hit_count);
        CUDA_SYNC_CHECK();
    }

    if (alloc_temp) {
        cudaFree(d_hit_inds);
        cudaFree(d_hit_count);
    }
}

void init_ray_start_samples_single(Params* params) {
    const int num_prims = params->means.count / 3;  // means uses scalar packing (xyzxyz...)
    const size_t grid_size = (num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;

    accumulate_initial_samples_single_kernel<<<grid_size, BLOCK_SIZE>>>(
        reinterpret_cast<const glm::vec3*>(params->means.data),
        reinterpret_cast<const glm::vec3*>(params->scales.data),
        reinterpret_cast<const glm::vec4*>(params->quats.data),
        params->densities.data,
        params->features.data,
        num_prims,
        reinterpret_cast<const glm::vec3*>(params->ray_origins.data),
        reinterpret_cast<float*>(params->initial_contrib.data));
    CUDA_SYNC_CHECK();
}

void init_ray_start_samples_zero(Params* params) {
    // No-op: placeholder when initial samples are not needed
}
