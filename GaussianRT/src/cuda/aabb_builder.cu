#include "../types.h"
#include "../acceleration_structure.h"
#include <cuda_runtime.h>

namespace gaussian_rt {

// Compute AABB for a single ellipsoid
// The AABB is computed by finding the extremes of the rotated ellipsoid
__device__ AABB compute_ellipsoid_aabb(
    float3 position,
    float3 scale,
    float4 rotation
) {
    // For an ellipsoid with semi-axes (a, b, c) rotated by quaternion q,
    // we need to find the axis-aligned bounding box.
    //
    // The ellipsoid surface is: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 in local coords
    // After rotation, we need to find max extent in each world axis direction.
    //
    // For each world axis e_i, the maximum extent is:
    //   max_extent_i = sqrt((R_i1 * a)^2 + (R_i2 * b)^2 + (R_i3 * c)^2)
    // where R_ij are elements of the rotation matrix.

    // Convert quaternion to rotation matrix columns
    // q = (w, x, y, z)
    float qw = rotation.x;
    float qx = rotation.y;
    float qy = rotation.z;
    float qz = rotation.w;

    // Rotation matrix (column-major thinking, but we need rows for extent calculation)
    // First row: transforms local X to world
    float3 row0 = make_float3(
        1.0f - 2.0f * (qy * qy + qz * qz),
        2.0f * (qx * qy - qw * qz),
        2.0f * (qx * qz + qw * qy)
    );

    // Second row: transforms local Y to world
    float3 row1 = make_float3(
        2.0f * (qx * qy + qw * qz),
        1.0f - 2.0f * (qx * qx + qz * qz),
        2.0f * (qy * qz - qw * qx)
    );

    // Third row: transforms local Z to world
    float3 row2 = make_float3(
        2.0f * (qx * qz - qw * qy),
        2.0f * (qy * qz + qw * qx),
        1.0f - 2.0f * (qx * qx + qy * qy)
    );

    // Compute extent in each world axis direction
    // extent_i = sqrt(sum_j (R_ij * scale_j)^2)
    float extent_x = sqrtf(
        (row0.x * scale.x) * (row0.x * scale.x) +
        (row0.y * scale.y) * (row0.y * scale.y) +
        (row0.z * scale.z) * (row0.z * scale.z)
    );

    float extent_y = sqrtf(
        (row1.x * scale.x) * (row1.x * scale.x) +
        (row1.y * scale.y) * (row1.y * scale.y) +
        (row1.z * scale.z) * (row1.z * scale.z)
    );

    float extent_z = sqrtf(
        (row2.x * scale.x) * (row2.x * scale.x) +
        (row2.y * scale.y) * (row2.y * scale.y) +
        (row2.z * scale.z) * (row2.z * scale.z)
    );

    AABB result;
    result.min_bound = make_float3(
        position.x - extent_x,
        position.y - extent_y,
        position.z - extent_z
    );
    result.max_bound = make_float3(
        position.x + extent_x,
        position.y + extent_y,
        position.z + extent_z
    );

    return result;
}

__global__ void kernel_compute_aabbs(
    const float3* __restrict__ positions,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    AABB* __restrict__ aabbs,
    uint32_t num_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    aabbs[idx] = compute_ellipsoid_aabb(
        positions[idx],
        scales[idx],
        rotations[idx]
    );
}

void launch_compute_aabbs(
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    AABB* aabbs,
    uint32_t num_elements,
    cudaStream_t stream
) {
    constexpr uint32_t BLOCK_SIZE = 256;
    uint32_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_compute_aabbs<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        positions,
        scales,
        rotations,
        aabbs,
        num_elements
    );
}

} // namespace gaussian_rt
