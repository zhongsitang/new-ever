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
#include <iostream>
#include <sstream>
#include <stdexcept>

// =============================================================================
// Math Utilities (replacing glm)
// =============================================================================

// float3 operations
__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __host__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float length3(const float3& v) {
    return sqrtf(dot3(v, v));
}

__device__ __host__ inline float3 normalize3(const float3& v) {
    float inv_len = rsqrtf(dot3(v, v));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

// float4 operations
__device__ __host__ inline float dot4(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __host__ inline float4 normalize4(const float4& v) {
    float inv_len = rsqrtf(dot4(v, v));
    return make_float4(v.x * inv_len, v.y * inv_len, v.z * inv_len, v.w * inv_len);
}

// 3x3 Matrix stored in column-major order (like glm)
// mat3[col][row] -> m[col * 3 + row]
struct Mat3 {
    float m[9];

    __device__ __host__ Mat3() {}

    // Diagonal matrix constructor
    __device__ __host__ Mat3(float diagonal) {
        m[0] = diagonal; m[1] = 0.0f;     m[2] = 0.0f;      // col 0
        m[3] = 0.0f;     m[4] = diagonal; m[5] = 0.0f;      // col 1
        m[6] = 0.0f;     m[7] = 0.0f;     m[8] = diagonal;  // col 2
    }

    // Construct from 9 values in column-major order (same as glm brace initialization)
    __device__ __host__ Mat3(
        float m00, float m10, float m20,  // column 0
        float m01, float m11, float m21,  // column 1
        float m02, float m12, float m22   // column 2
    ) {
        m[0] = m00; m[1] = m10; m[2] = m20;  // col 0
        m[3] = m01; m[4] = m11; m[5] = m21;  // col 1
        m[6] = m02; m[7] = m12; m[8] = m22;  // col 2
    }

    // Access element at [col][row]
    __device__ __host__ float& at(int col, int row) {
        return m[col * 3 + row];
    }

    __device__ __host__ float at(int col, int row) const {
        return m[col * 3 + row];
    }
};

// Matrix-vector multiplication: M * v
__device__ __host__ inline float3 mat3_mul_vec3(const Mat3& M, const float3& v) {
    return make_float3(
        M.at(0, 0) * v.x + M.at(1, 0) * v.y + M.at(2, 0) * v.z,
        M.at(0, 1) * v.x + M.at(1, 1) * v.y + M.at(2, 1) * v.z,
        M.at(0, 2) * v.x + M.at(1, 2) * v.y + M.at(2, 2) * v.z
    );
}

// Matrix multiplication: A * B
__device__ __host__ inline Mat3 mat3_mul_mat3(const Mat3& A, const Mat3& B) {
    Mat3 result;
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            result.at(col, row) =
                A.at(0, row) * B.at(col, 0) +
                A.at(1, row) * B.at(col, 1) +
                A.at(2, row) * B.at(col, 2);
        }
    }
    return result;
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

    const float4 quat = normalize4(quats[i]);
    const float3 center = means[i];
    const float3 size = scales[i];

    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    // Quaternion to rotation matrix transpose (Rt)
    // Stored in column-major order like glm
    const Mat3 Rt(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y),  // col 0
        2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + r * x),  // col 1
        2.0f * (x * z + r * y), 2.0f * (y * z - r * x), 1.0f - 2.0f * (x * x + y * y)   // col 2
    );

    // Scale matrix
    Mat3 S(1.0f);
    S.at(0, 0) = size.x;
    S.at(1, 1) = size.y;
    S.at(2, 2) = size.z;

    // M = S * Rt (3x3 part of the transformation)
    Mat3 M = mat3_mul_mat3(S, Rt);

    // Compute column norms of M for AABB extents
    // In glm with column-major, M[col][row], so M[0][0], M[0][1], M[0][2] is column 0
    float col0_norm = sqrtf(M.at(0, 0) * M.at(0, 0) + M.at(0, 1) * M.at(0, 1) + M.at(0, 2) * M.at(0, 2));
    float col1_norm = sqrtf(M.at(1, 0) * M.at(1, 0) + M.at(1, 1) * M.at(1, 1) + M.at(1, 2) * M.at(1, 2));
    float col2_norm = sqrtf(M.at(2, 0) * M.at(2, 0) + M.at(2, 1) * M.at(2, 1) + M.at(2, 2) * M.at(2, 2));

    OptixAabb aabb;
    aabb.minX = center.x - col0_norm;
    aabb.minY = center.y - col1_norm;
    aabb.minZ = center.z - col2_norm;
    aabb.maxX = center.x + col0_norm;
    aabb.maxY = center.y + col1_norm;
    aabb.maxZ = center.z + col2_norm;
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

#define SQR(x) (x)*(x)

__device__ static const float SH_C0 = 0.28209479177387814f;

__device__ float3 transform_to_ellipsoid_space(
    const float3 center,
    const float4 quat,
    const float3 size,
    const float3 rayo)
{
    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    // Quaternion to rotation matrix transpose (Rt)
    const Mat3 Rt(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y),  // col 0
        2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + r * x),  // col 1
        2.0f * (x * z + r * y), 2.0f * (y * z - r * x), 1.0f - 2.0f * (x * x + y * y)   // col 2
    );

    const float3 Trayo = mat3_mul_vec3(Rt, rayo - center) / size;
    return Trayo;
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

    // Jim Arvo, Graphics Gems - point to AABB squared distance
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

    float3 rayo = rayos[j] + tmin * normalize3(rayds[j]);

    const int prim_ind = touch_indices[i];
    const float4 quat = normalize4(quats[prim_ind]);
    const float3 center = means[prim_ind];
    const float3 size = scales[prim_ind];

    const float3 Trayo = transform_to_ellipsoid_space(center, quat, size, rayo);
    const float dist = Trayo.x * Trayo.x + Trayo.y * Trayo.y + Trayo.z * Trayo.z;

    if (dist <= 1) {
        const float density = densities[prim_ind];
        const float3 color = make_float3(
            features[prim_ind * 3 + 0] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 1] * SH_C0 + 0.5f,
            features[prim_ind * 3 + 2] * SH_C0 + 0.5f
        );
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
        (float3 *)(params->ray_origins.data),
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
            (float3 *)(params->means.data),
            (float3 *)(params->scales.data),
            (float4 *)(params->quats.data),
            (float *)(params->densities.data),
            (float *)(params->features.data),
            num_prims,
            num_rays,
            params->tmin,
            (float3 *)(params->ray_origins.data),
            (float3 *)(params->ray_directions.data),
            (float *)(params->initial_drgb.data),
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

    const float4 quat = normalize4(quats[i]);
    const float density = densities[i];
    const float3 center = means[i];
    const float3 size = scales[i];

    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;

    // Quaternion to rotation matrix transpose (Rt)
    const Mat3 Rt(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y),  // col 0
        2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + r * x),  // col 1
        2.0f * (x * z + r * y), 2.0f * (y * z - r * x), 1.0f - 2.0f * (x * x + y * y)   // col 2
    );

    const float3 Trayo = mat3_mul_vec3(Rt, rayo[0] - center) / size;
    float dist = Trayo.x * Trayo.x + Trayo.y * Trayo.y + Trayo.z * Trayo.z;

    if (dist <= 1) {
        float3 color = make_float3(
            features[i * 3 + 0] * SH_C0 + 0.5f,
            features[i * 3 + 1] * SH_C0 + 0.5f,
            features[i * 3 + 2] * SH_C0 + 0.5f
        );
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
        (float3 *)(params->means.data),
        (float3 *)(params->scales.data),
        (float4 *)(params->quats.data),
        (float *)(params->densities.data),
        (float *)(params->features.data),
        num_prims,
        (float3 *)(params->ray_origins.data),
        (float *)(params->initial_drgb.data));

    CUDA_SYNC_CHECK();
}

void init_ray_start_samples_zero(Params *params) {
    // No-op: used when no initial samples needed
}
