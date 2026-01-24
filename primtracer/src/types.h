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

#pragma once

#include <cstdint>
#include <optix.h>

// =============================================================================
// Basic Types (ABI-stable: fixed-width integers, explicit padding)
// =============================================================================

/// Slang StructuredBuffer layout: {T* data, uint64_t count}
/// Uses uint64_t instead of size_t for ABI stability across platforms.
template <typename T>
struct StructuredBuffer {
    T* data;
    uint64_t count;
};
static_assert(sizeof(StructuredBuffer<float>) == 16);

/// GPU self-check sentinel values for early ABI mismatch detection.
/// These magic values are written by host and verified by GPU on first thread.
constexpr float GPU_CHECK_SENTINEL_0 = 1.2345678f;
constexpr float GPU_CHECK_SENTINEL_1 = 8.7654321f;
constexpr uint32_t GPU_CHECK_PASS = 0;
constexpr uint32_t GPU_CHECK_FAIL = 0xDEADBEEF;

// =============================================================================
// Primitive Data
// =============================================================================

/// Ellipsoid primitive geometry (GPU pointers)
/// Note: This is a host-side helper struct, not shared with GPU.
struct Primitives {
    float3* means;
    float3* scales;
    float4* quats;        // quaternion (w,x,y,z)
    float* densities;
    uint64_t num_prims;
    float* features;      // SH coefficients
    uint64_t feature_size;
};

// =============================================================================
// Volume Rendering State
// =============================================================================

/// Per-ray volume integrator state (48 bytes, 16-byte aligned)
struct IntegratorState {
    float4 accumulated_contrib;  // (density, r*d, g*d, b*d)
    float3 C;                    // accumulated color RGB
    float logT;                  // log transmittance
    float depth_accum;           // accumulated depth
    float t;                     // current ray parameter
    float _pad[2];
};

static_assert(sizeof(IntegratorState) == 48);
static_assert(alignof(IntegratorState) == 16);

/// State saved for backward gradient computation
struct SavedState {
    IntegratorState* states;     // (M,) per-ray integrator state
    float4* delta_contribs;      // (M,) last delta contribution
    uint32_t* iters;             // (M,) iteration count per ray
    uint32_t* prim_hits;         // (N,) hit count per primitive
    int32_t* hit_collection;     // (M * max_iters,) hit primitive indices
    float4* initial_contrib;     // (M,) contribution for rays starting inside
    int32_t* initial_prim_indices; // (N,) primitives containing ray origins
    int32_t* initial_prim_count; // (1,) count of initial_prim_indices
};

// =============================================================================
// OptiX Launch Parameters (must match slang layout)
// =============================================================================

/// Camera parameters (64 bytes, 16-byte aligned)
/// Uses float4 instead of float3 for ABI-stable 16-byte stride.
/// The .w component is unused (set to 0).
struct Camera {
    float fx, fy;
    int32_t height, width;
    float4 U, V, W;   // float4 for stable 16-byte stride (.w unused)
    float4 eye;       // float4 for stable 16-byte stride (.w unused)
};
static_assert(sizeof(Camera) == 80);
static_assert(alignof(Camera) == 16);

/// OptiX launch parameters (must match Slang Params struct exactly).
/// Layout rules:
/// - All pointers/handles are 8-byte aligned
/// - Use uint32_t for small integers (not size_t)
/// - Explicit padding before 8-byte aligned fields after 4-byte fields
/// - float3 buffers use StructuredBuffer<float> in Slang (read 3 floats manually)
///   to avoid stride mismatch (CUDA float3 = 12 bytes, Slang float3 may be 16)
struct Params {
    // ===== Output buffers =====
    StructuredBuffer<float4> image;
    StructuredBuffer<float> depth_out;
    StructuredBuffer<uint32_t> iters;
    StructuredBuffer<uint32_t> last_prim;
    StructuredBuffer<uint32_t> prim_hits;
    StructuredBuffer<float4> last_delta_contrib;
    StructuredBuffer<IntegratorState> last_state;
    StructuredBuffer<int32_t> hit_collection;

    // ===== Ray data =====
    // float3 data stored as StructuredBuffer<float>, read 3 floats per element.
    // This avoids stride mismatch: CUDA float3 is 12 bytes packed.
    StructuredBuffer<float> ray_origins;     // [N*3] floats
    StructuredBuffer<float> ray_directions;  // [N*3] floats
    Camera camera;

    // ===== Primitive data =====
    StructuredBuffer<float> means;    // [N*3] floats
    StructuredBuffer<float> scales;   // [N*3] floats
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // ===== Render settings =====
    uint32_t sh_degree;
    uint32_t max_iters;
    float tmin;
    float _pad0;  // explicit padding before 8-byte aligned tmax
    StructuredBuffer<float> tmax;
    StructuredBuffer<float4> initial_contrib;
    float max_prim_size;
    float _pad1;  // explicit padding before 8-byte aligned handle
    uint64_t handle;  // OptixTraversableHandle (use uint64_t for Slang compat)

    // ===== GPU self-check =====
    // First thread validates these sentinels and writes result to debug_flag.
    float check_sentinel0;  // should be GPU_CHECK_SENTINEL_0
    float check_sentinel1;  // should be GPU_CHECK_SENTINEL_1
    uint32_t* debug_flag;   // GPU writes GPU_CHECK_FAIL if mismatch
    uint32_t _pad2;         // padding to maintain 8-byte alignment at end
};
static_assert(sizeof(Params) % 8 == 0, "Params must be 8-byte aligned");
static_assert(offsetof(Params, handle) % 8 == 0, "handle must be 8-byte aligned");
static_assert(offsetof(Params, debug_flag) % 8 == 0, "debug_flag must be 8-byte aligned");
