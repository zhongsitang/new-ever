// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#include "ray_tracer.h"
#include <cuda_runtime.h>
#include "cuda_math.h"  // dot, length, normalize, etc.

namespace {
    constexpr int   BLOCK_SIZE = 256;

    // Quaternion normalization guard: if |q|^2 is tiny, treat as identity.
    constexpr float QUAT_EPS2 = 1e-12f;

    // Small padding to avoid overly tight bounds (FP error / later eps in intersection).
    constexpr float AABB_PAD  = 1e-6f;
}

// =============================================================================
// AABB Computation (optimized for millions of prims)
// =============================================================================

__global__ void compute_primitive_bounds_kernel_opt(
    const float* __restrict__ means,   // [N*3]
    const float* __restrict__ scales,  // [N*3]  (local radii)
    const float* __restrict__ quats,   // [N*4]  (w,x,y,z), local -> world
    int N,
    OptixAabb* __restrict__ aabbs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // -----------------------------
    // Coalesced-ish loads (SoA)
    // -----------------------------
    int m3 = i * 3;
    float cx = means[m3 + 0];
    float cy = means[m3 + 1];
    float cz = means[m3 + 2];

    float sx = fabsf(scales[m3 + 0]);
    float sy = fabsf(scales[m3 + 1]);
    float sz = fabsf(scales[m3 + 2]);

    int q4 = i * 4;
    float w0 = quats[q4 + 0];
    float x0 = quats[q4 + 1];
    float y0 = quats[q4 + 2];
    float z0 = quats[q4 + 3];

    // -----------------------------
    // Safe quaternion normalize (mostly branchless)
    // -----------------------------
    float len2 = fmaf(w0, w0, fmaf(x0, x0, fmaf(y0, y0, z0 * z0)));

    // valid = 1 if len2 >= QUAT_EPS2 else 0
    float valid = (len2 >= QUAT_EPS2) ? 1.0f : 0.0f;

    // Avoid rsqrt(0); when invalid, inv_len -> 0, and we blend to identity.
    float inv_len = rsqrtf(fmaxf(len2, QUAT_EPS2));

    float w = valid * (w0 * inv_len) + (1.0f - valid) * 1.0f;
    float x = valid * (x0 * inv_len);
    float y = valid * (y0 * inv_len);
    float z = valid * (z0 * inv_len);

    // -----------------------------
    // Compute |R| entries directly (R from unit quat w,x,y,z)
    // We only need abs(R_ij) to compute extents e = |R| * s
    // -----------------------------
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    // R00 R01 R02
    float r00 = 1.0f - 2.0f * (yy + zz);
    float r01 = 2.0f * (xy - wz);
    float r02 = 2.0f * (xz + wy);

    // R10 R11 R12
    float r10 = 2.0f * (xy + wz);
    float r11 = 1.0f - 2.0f * (xx + zz);
    float r12 = 2.0f * (yz - wx);

    // R20 R21 R22
    float r20 = 2.0f * (xz - wy);
    float r21 = 2.0f * (yz + wx);
    float r22 = 1.0f - 2.0f * (xx + yy);

    // -----------------------------
    // AABB half-extents for rotated ellipsoid:
    // ex = |R00|*sx + |R01|*sy + |R02|*sz, etc.
    // -----------------------------
    float ex = fabsf(r00) * sx + fabsf(r01) * sy + fabsf(r02) * sz + AABB_PAD;
    float ey = fabsf(r10) * sx + fabsf(r11) * sy + fabsf(r12) * sz + AABB_PAD;
    float ez = fabsf(r20) * sx + fabsf(r21) * sy + fabsf(r22) * sz + AABB_PAD;

    // Store
    OptixAabb out;
    out.minX = cx - ex; out.minY = cy - ey; out.minZ = cz - ez;
    out.maxX = cx + ex; out.maxY = cy + ey; out.maxZ = cz + ez;
    aabbs[i] = out;
}

void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs)
{
    int grid = (prims.num_prims + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_primitive_bounds_kernel_opt<<<grid, BLOCK_SIZE>>>(
        prims.means,
        prims.scales,
        prims.quats,
        prims.num_prims,
        aabbs);
    CUDA_SYNC_CHECK();
}
