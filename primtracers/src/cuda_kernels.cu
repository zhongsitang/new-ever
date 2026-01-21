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
#include "volume_types.h"
#include <cuda.h>
#include <cuda_runtime.h>

// =============================================================================
// Vector Math Utilities
// =============================================================================

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ __forceinline__ float3 operator/(const float3& v, const float3& s) {
    return make_float3(v.x / s.x, v.y / s.y, v.z / s.z);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float length(const float4& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

__device__ __forceinline__ float4 normalize(const float4& v) {
    float inv_len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    return make_float4(v.x * inv_len, v.y * inv_len, v.z * inv_len, v.w * inv_len);
}

// Quaternion to rotation matrix (returns row-major 3x3 as 3 float3 rows)
__device__ __forceinline__ void quat_to_rotation_matrix(
    const float4& q,
    float3& row0, float3& row1, float3& row2)
{
    float r = q.x, x = q.y, y = q.z, z = q.w;
    row0 = make_float3(
        1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - r * z),
        2.0f * (x * z + r * y));
    row1 = make_float3(
        2.0f * (x * y + r * z),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - r * x));
    row2 = make_float3(
        2.0f * (x * z - r * y),
        2.0f * (y * z + r * x),
        1.0f - 2.0f * (x * x + y * y));
}

// Matrix-vector multiply: result = R * v (R is row-major)
__device__ __forceinline__ float3 mat3_mul_vec3(
    const float3& row0, const float3& row1, const float3& row2,
    const float3& v)
{
    return make_float3(
        dot(row0, v),
        dot(row1, v),
        dot(row2, v));
}

// =============================================================================
// Primitive Bounding Box Construction
// =============================================================================

__global__ void compute_primitive_bounds_kernel(
    const float3 *means,
    const float3 *scales,
    const float4 *quats,
    const float *densities,
    const size_t num_prims,
    OptixAabb *aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    const float4 quat = normalize(quats[i]);
    const float3 center = means[i];
    const float3 size = scales[i];

    // Get rotation matrix
    float3 R0, R1, R2;
    quat_to_rotation_matrix(quat, R0, R1, R2);

    // Scale * Rotation: each row of S*R is scale[row] * R[row]
    float3 SR0 = make_float3(size.x * R0.x, size.x * R0.y, size.x * R0.z);
    float3 SR1 = make_float3(size.y * R1.x, size.y * R1.y, size.y * R1.z);
    float3 SR2 = make_float3(size.z * R2.x, size.z * R2.y, size.z * R2.z);

    // Row norms give the extent in each axis direction
    float row0_norm = length(SR0);
    float row1_norm = length(SR1);
    float row2_norm = length(SR2);

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
        prims.means,
        prims.scales,
        prims.quats,
        prims.densities,
        prims.num_prims,
        prims.aabbs);
    CUDA_SYNC_CHECK();
}

// =============================================================================
// Initial Ray Sample Accumulation
// =============================================================================

#define SQR(x) ((x)*(x))

__device__ static const float SH_C0 = 0.28209479177387814f;

__device__ float3 transform_to_ellipsoid_space(
    const float3 center,
    const float4 quat,
    const float3 size,
    const float3 rayo)
{
    float3 R0, R1, R2;
    quat_to_rotation_matrix(quat, R0, R1, R2);

    float3 local_pos = rayo - center;
    float3 rotated = mat3_mul_vec3(R0, R1, R2, local_pos);
    return rotated / size;
}

__global__ void find_enclosing_primitives_kernel(
    const OptixAabb *aabbs,
    const size_t num_prims,
    const float tmin,
    const float3 *rayos,
    int *touch_indices,
    int *touch_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    OptixAabb aabb = aabbs[i];
    const float3 rayo = rayos[0];

    // Jim Arvo, Graphics Gems - squared distance to AABB
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
        int pos = atomicAdd(touch_count, 1);
        touch_indices[pos] = i;
    }
}

__global__ void accumulate_initial_samples_kernel(
    const float3 *means,
    const float3 *scales,
    const float4 *quats,
    const float *densities,
    const float *features,
    const size_t num_prims,
    const size_t num_rays,
    const float tmin,
    const float3 *rayos,
    const float3 *rayds,
    float *initial_drgb,
    int *touch_indices,
    int *touch_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= *touch_count) return;
    if (j >= num_rays) return;

    float3 rayo = rayos[j] + tmin * normalize(rayds[j]);

    const int prim_ind = touch_indices[i];
    const float4 quat = normalize(quats[prim_ind]);
    const float3 center = means[prim_ind];
    const float3 size = scales[prim_ind];

    const float3 Trayo = transform_to_ellipsoid_space(center, quat, size, rayo);
    const float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;

    if (dist <= 1) {
        const float density = densities[prim_ind];
        const float3 color = make_float3(
            features[prim_ind * 3 + 0] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 1] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 2] * SH_C0 + 0.5f);
        atomicAdd(initial_drgb + 4 * j + 0, density);
        atomicAdd(initial_drgb + 4 * j + 1, density * color.x);
        atomicAdd(initial_drgb + 4 * j + 2, density * color.y);
        atomicAdd(initial_drgb + 4 * j + 3, density * color.z);
    }
}

void init_ray_start_samples(Params *params, OptixAabb *aabbs, int *d_touch_count, int *d_touch_inds) {
    const size_t block_size = 1024;
    const size_t ray_block_size = 64;
    const size_t second_block_size = 16;
    int num_prims = params->means.size;
    int num_rays = params->initial_drgb.size;

    dim3 grid_dim(
        (num_prims + block_size - 1) / block_size,
        (num_rays + ray_block_size - 1) / ray_block_size
    );
    dim3 block_dim(block_size, ray_block_size);

    bool initialize_tensors = d_touch_count == NULL;
    if (initialize_tensors) {
        cudaMalloc((void**)&d_touch_inds, num_prims * sizeof(int));
        cudaMalloc((void**)&d_touch_count, sizeof(int));
    }
    cudaMemset(d_touch_count, 0, sizeof(int));

    find_enclosing_primitives_kernel<<<grid_dim.x, block_dim.x>>>(
        aabbs,
        num_prims,
        params->tmin,
        params->ray_origins.data,
        d_touch_inds,
        d_touch_count);

    int touch_count;
    cudaMemcpy(&touch_count, d_touch_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (touch_count > 0) {
        dim3 init_grid_dim(
            (num_rays + ray_block_size - 1) / ray_block_size,
            (touch_count + second_block_size - 1) / second_block_size,
            1
        );
        dim3 init_block_dim(ray_block_size, second_block_size, 1);

        accumulate_initial_samples_kernel<<<init_grid_dim, init_block_dim>>>(
            params->means.data,
            params->scales.data,
            params->quats.data,
            params->densities.data,
            params->features.data,
            num_prims,
            num_rays,
            params->tmin,
            params->ray_origins.data,
            params->ray_directions.data,
            reinterpret_cast<float*>(params->initial_drgb.data),
            d_touch_inds,
            d_touch_count);

        CUDA_SYNC_CHECK();
    }

    if (initialize_tensors) {
        cudaFree(d_touch_inds);
        cudaFree(d_touch_count);
    }
}

__global__ void accumulate_initial_samples_single_kernel(
    const float3 *means,
    const float3 *scales,
    const float4 *quats,
    const float *densities,
    const float *features,
    const size_t num_prims,
    const float3 *rayo,
    float *initial_drgb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= num_prims)
        return;

    const float4 quat = normalize(quats[i]);
    const float density = densities[i];
    const float3 center = means[i];
    const float3 size = scales[i];

    const float3 Trayo = transform_to_ellipsoid_space(center, quat, size, rayo[0]);
    float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;

    if (dist <= 1) {
        float3 color = make_float3(
            features[i * 3 + 0] * SH_C0 + 0.5f,
            features[i * 3 + 1] * SH_C0 + 0.5f,
            features[i * 3 + 2] * SH_C0 + 0.5f);
        atomicAdd(initial_drgb + 0, density);
        atomicAdd(initial_drgb + 1, density * color.x);
        atomicAdd(initial_drgb + 2, density * color.y);
        atomicAdd(initial_drgb + 3, density * color.z);
    }
}

void init_ray_start_samples_single(Params *params) {
    const size_t block_size = 1024;
    int num_prims = params->means.size;

    accumulate_initial_samples_single_kernel<<<(num_prims + block_size - 1) / block_size, block_size>>>(
        params->means.data,
        params->scales.data,
        params->quats.data,
        params->densities.data,
        params->features.data,
        num_prims,
        params->ray_origins.data,
        reinterpret_cast<float*>(params->initial_drgb.data));

    CUDA_SYNC_CHECK();
}

void init_ray_start_samples_zero(Params *params) {
    // No-op: used when no initial samples needed
}
