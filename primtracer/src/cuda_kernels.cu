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

#include "cuda_kernels.h"
#include "optix_error.h"
#include "glm/glm.hpp"
#include "volume_types.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// =============================================================================
// Primitive Bounding Box Construction
// =============================================================================

__global__ void compute_primitive_bounds_kernel(
    const glm::vec3 *means,
    const glm::vec3 *scales,
    const glm::vec4 *quats,
    const float *densities,
    const size_t num_prims,
    OptixAabb *aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    const glm::vec4 quat = glm::normalize(quats[i]);
    const glm::vec3 center = means[i];
    const glm::vec3 size = scales[i];

    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    const glm::mat3 Rt = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    };

    float s = 1.0;
    glm::mat3 S = glm::mat3(1.0);
    S[0][0] = s * size.x;
    S[1][1] = s * size.y;
    S[2][2] = s * size.z;

    glm::mat4 M = glm::mat4(S * Rt);
    M[0][3] = center.x;
    M[1][3] = center.y;
    M[2][3] = center.z;

    float row0_norm = sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2]);
    float row1_norm = sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2]);
    float row2_norm = sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2]);

    OptixAabb aabb;
    aabb.minX = center.x - row0_norm;
    aabb.minY = center.y - row1_norm;
    aabb.minZ = center.z - row2_norm;
    aabb.maxX = center.x + row0_norm;
    aabb.maxY = center.y + row1_norm;
    aabb.maxZ = center.z + row2_norm;
    aabbs[i] = aabb;
}

void build_primitive_aabbs(Primitives &prims) {
    const size_t block_size = 1024;
    if (prims.prev_alloc_size < prims.num_prims) {
        if (prims.prev_alloc_size > 0) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(prims.aabbs)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prims.aabbs),
                              prims.num_prims * sizeof(OptixAabb)));
    }
    compute_primitive_bounds_kernel<<<(prims.num_prims + block_size - 1) / block_size, block_size>>>(
        (glm::vec3 *)prims.means,
        (glm::vec3 *)prims.scales,
        (glm::vec4 *)prims.quats,
        prims.densities,
        prims.num_prims,
        prims.aabbs);
    CUDA_SYNC_CHECK();
}

// =============================================================================
// Initial Ray Sample Accumulation
// =============================================================================

#define SQR(x) (x)*(x)

__device__ static const float SH_C0 = 0.28209479177387814f;

__device__ glm::vec3 transform_to_ellipsoid_space(
    const glm::vec3 center,
    const glm::vec4 quat,
    const glm::vec3 size,
    const glm::vec3 rayo)
{
    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    const glm::mat3 Rt = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    };

    const glm::vec3 Trayo = (Rt * (rayo - center)) / size;
    return Trayo;
}

__global__ void find_enclosing_primitives_kernel(
    const OptixAabb *aabbs,
    const size_t num_prims,
    const float tmin,
    const glm::vec3 *rayos,
    int *hit_indices,
    int *hit_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    OptixAabb aabb = aabbs[i];
    const glm::vec3 rayo = rayos[0];

    // Jim Arvo, Graphics Gems.
    float dmin = 0;
    if (rayo.x < aabb.minX)
        dmin += SQR(rayo.x - aabb.minX);
    else if (rayo.x > aabb.maxX)
        dmin += SQR(rayo.x - aabb.maxX);

    if (rayo.y < aabb.minY)
        dmin += SQR(rayo.y - aabb.minY);
    else if (rayo.y > aabb.maxY)
        dmin += SQR(rayo.y - aabb.maxY);

    if (rayo.z < aabb.minZ)
        dmin += SQR(rayo.z - aabb.minZ);
    else if (rayo.z > aabb.maxZ)
        dmin += SQR(rayo.z - aabb.maxZ);

    if (dmin <= tmin * tmin) {
        int pos = atomicAdd(hit_count, 1);
        hit_indices[pos] = i;
    }
}

__global__ void accumulate_initial_samples_kernel(
    const glm::vec3 *means,
    const glm::vec3 *scales,
    const glm::vec4 *quats,
    const float *densities,
    const float *features,
    const size_t num_prims,
    const size_t num_rays,
    const float tmin,
    const glm::vec3 *rayos,
    const glm::vec3 *rayds,
    float *initial_contrib,
    int *hit_indices,
    int *hit_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= *hit_count) return;
    if (j >= num_rays) return;

    glm::vec3 rayo = rayos[j] + tmin * glm::normalize(rayds[j]);

    const int prim_ind = hit_indices[i];
    const glm::vec4 quat = glm::normalize(quats[prim_ind]);
    const glm::vec3 center = means[prim_ind];
    const glm::vec3 size = scales[prim_ind];

    const glm::vec3 Trayo = transform_to_ellipsoid_space(center, quat, size, rayo);
    const float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;

    if (dist <= 1) {
        const float density = densities[prim_ind];
        const glm::vec3 color = {
            features[prim_ind * 3 + 0] * SH_C0 + 0.5,
            features[prim_ind * 3 + 1] * SH_C0 + 0.5,
            features[prim_ind * 3 + 2] * SH_C0 + 0.5,
        };
        atomicAdd(initial_contrib + 4 * j + 0, density);
        atomicAdd(initial_contrib + 4 * j + 1, density * color.x);
        atomicAdd(initial_contrib + 4 * j + 2, density * color.y);
        atomicAdd(initial_contrib + 4 * j + 3, density * color.z);
    }
}

void init_ray_start_samples(Params *params, OptixAabb *aabbs, int *d_hit_count, int *d_hit_inds) {
    const size_t block_size = 1024;
    const size_t ray_block_size = 64;
    const size_t second_block_size = 16;
    int num_prims = params->means.size;
    int num_rays = params->initial_contrib.size;

    dim3 grid_dim(
        (num_prims + block_size - 1) / block_size,
        (num_rays + ray_block_size - 1) / ray_block_size
    );
    dim3 block_dim(block_size, ray_block_size);

    bool initialize_tensors = d_hit_count == NULL;
    if (initialize_tensors) {
        cudaMalloc((void**)&d_hit_inds, num_prims * sizeof(int));
        cudaMalloc((void**)&d_hit_count, sizeof(int));
    }
    cudaMemset(d_hit_count, 0, sizeof(int));

    find_enclosing_primitives_kernel<<<grid_dim.x, block_dim.x>>>(
        aabbs,
        num_prims,
        params->tmin,
        (glm::vec3 *)(params->ray_origins.data),
        d_hit_inds,
        d_hit_count);

    int hit_count;
    cudaMemcpy(&hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (hit_count > 0) {
        dim3 init_grid_dim(
            (num_rays + ray_block_size - 1) / ray_block_size,
            (hit_count + second_block_size - 1) / second_block_size,
            1
        );
        dim3 init_block_dim(ray_block_size, second_block_size, 1);

        accumulate_initial_samples_kernel<<<init_grid_dim, init_block_dim>>>(
            (glm::vec3 *)(params->means.data),
            (glm::vec3 *)(params->scales.data),
            (glm::vec4 *)(params->quats.data),
            (float *)(params->densities.data),
            (float *)(params->features.data),
            num_prims,
            num_rays,
            params->tmin,
            (glm::vec3 *)(params->ray_origins.data),
            (glm::vec3 *)(params->ray_directions.data),
            (float *)(params->initial_contrib.data),
            d_hit_inds,
            d_hit_count);

        CUDA_SYNC_CHECK();
    }

    if (initialize_tensors) {
        cudaFree(d_hit_inds);
        cudaFree(d_hit_count);
    }
}

__global__ void accumulate_initial_samples_single_kernel(
    const glm::vec3 *means,
    const glm::vec3 *scales,
    const glm::vec4 *quats,
    const float *densities,
    const float *features,
    const size_t num_prims,
    const glm::vec3 *rayo,
    float *initial_contrib)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    const glm::vec4 quat = glm::normalize(quats[i]);
    const float density = densities[i];
    const glm::vec3 center = means[i];
    const glm::vec3 size = scales[i];

    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    const glm::mat3 Rt = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    };

    const glm::vec3 Trayo = (Rt * (rayo[0] - center)) / size;
    float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;

    if (dist <= 1) {
        glm::vec3 color = {
            features[i * 3 + 0] * SH_C0 + 0.5,
            features[i * 3 + 1] * SH_C0 + 0.5,
            features[i * 3 + 2] * SH_C0 + 0.5,
        };
        atomicAdd(initial_contrib + 0, density);
        atomicAdd(initial_contrib + 1, density * color.x);
        atomicAdd(initial_contrib + 2, density * color.y);
        atomicAdd(initial_contrib + 3, density * color.z);
    }
}

void init_ray_start_samples_single(Params *params) {
    const size_t block_size = 1024;
    int num_prims = params->means.size;

    accumulate_initial_samples_single_kernel<<<(num_prims + block_size - 1) / block_size, block_size>>>(
        (glm::vec3 *)(params->means.data),
        (glm::vec3 *)(params->scales.data),
        (glm::vec4 *)(params->quats.data),
        (float *)(params->densities.data),
        (float *)(params->features.data),
        num_prims,
        (glm::vec3 *)(params->ray_origins.data),
        (float *)(params->initial_contrib.data));

    CUDA_SYNC_CHECK();
}

void init_ray_start_samples_zero(Params *params) {
    // No-op: used when no initial samples needed
}
