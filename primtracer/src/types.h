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
// Basic Types
// =============================================================================

/// Slang StructuredBuffer layout: {T* data, uint64_t size}
/// Uses uint64_t to match pointer alignment and avoid padding/conversion overhead
template <typename T>
struct StructuredBuffer {
    T* data;
    uint64_t size;
};

// =============================================================================
// Primitive Data
// =============================================================================

/// Ellipsoid primitive geometry (GPU pointers)
struct Primitives {
    float3* means;
    float3* scales;
    float4* quats;        // quaternion (w,x,y,z)
    float* densities;
    size_t num_prims;
    float* features;      // SH coefficients
    size_t feature_size;
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

/// Camera parameters (must match Slang Camera in optix_shaders.slang)
/// Note: float3 is forbidden in shared structs (12-byte vs 16-byte alignment risk)
struct Camera {
    float fx, fy;
    int height, width;
    float4 U;    // Camera basis U (w unused)
    float4 V;    // Camera basis V (w unused)
    float4 W;    // Camera basis W (w unused)
    float4 eye;  // Camera origin (w unused)
};

/// OptiX launch parameters (binary-compatible with Slang SLANG_globalParams)
/// Layout rules:
/// - Use uint64_t for sizes (matches pointer alignment, no conversion needed)
/// - No float3 in struct fields (use float4)
/// - Explicit padding before 64-bit types after float
struct Params {
    // Output buffers
    StructuredBuffer<float4> image;
    StructuredBuffer<float> depth_out;
    StructuredBuffer<uint32_t> iters;
    StructuredBuffer<uint32_t> last_prim;
    StructuredBuffer<uint32_t> prim_hits;
    StructuredBuffer<float4> last_delta_contrib;
    StructuredBuffer<IntegratorState> last_state;
    StructuredBuffer<int32_t> hit_collection;

    // Ray data
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Camera camera;

    // Primitive data
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Render settings (use uint64_t to match pointer alignment)
    uint64_t sh_degree;
    uint64_t max_iters;
    float tmin;
    uint32_t _pad0;  // Padding for 8-byte alignment before tmax
    StructuredBuffer<float> tmax;
    StructuredBuffer<float4> initial_contrib;

    // Debug self-check buffer (always enabled, ~zero cost)
    StructuredBuffer<uint32_t> debug_flag;

    // Layout validation sentinel (host writes known pattern, shader verifies)
    uint64_t layout_sentinel;  // Must be 0xDEADBEEFCAFEBABE if layout matches

    float max_prim_size;
    uint32_t _pad1;  // Explicit padding before 64-bit handle
    OptixTraversableHandle handle;  // 64-bit, must be 8-byte aligned
};

// =============================================================================
// Layout Safety Checks (compile-time)
// =============================================================================

static_assert(sizeof(StructuredBuffer<float>) == 16,
    "StructuredBuffer must be 16 bytes (ptr + uint64_t size)");

static_assert(sizeof(Camera) == 80,
    "Camera layout mismatch with Slang");

static_assert(sizeof(Params) % 16 == 0,
    "Params must be 16-byte aligned for OptiX");

static_assert(offsetof(Params, handle) % 8 == 0,
    "OptixTraversableHandle must be 8-byte aligned");
