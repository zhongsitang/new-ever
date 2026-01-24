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
#include <cstddef>
#include <optix.h>

// =============================================================================
// ABI Stability Constants
// =============================================================================

/// Magic number for binary layout validation (ASCII "PRIM")
constexpr uint32_t PARAMS_MAGIC = 0x5052494D;

// =============================================================================
// Basic Types
// =============================================================================

/// Slang StructuredBuffer layout: {T* data, uint64 count}
/// Note: Uses uint64_t instead of size_t for ABI stability across platforms.
template <typename T>
struct StructuredBuffer {
    T* data;
    uint64_t count;  // Element count (not byte size)
};

static_assert(sizeof(StructuredBuffer<float>) == 16);
static_assert(alignof(StructuredBuffer<float>) == 8);

// =============================================================================
// Primitive Data
// =============================================================================

/// Ellipsoid primitive geometry (GPU pointers)
/// Note: Uses uint64_t instead of size_t for ABI stability.
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
/// Note: Uses int32_t instead of int for ABI stability.
struct Camera {
    float fx, fy;
    int32_t height, width;
    float3 U, V, W;
    float3 eye;
};

static_assert(sizeof(Camera) == 64);
static_assert(alignof(Camera) == 16);

/// OptiX Launch Parameters
/// Layout must match Slang global variables exactly (same order, same types).
/// Note: Uses uint64_t instead of size_t for ABI stability.
struct Params {
    // ABI validation buffer: [0]=magic (host writes), [1]=error_flag (shader writes)
    // Shader checks magic on first thread; if mismatch, writes error_flag=1
    StructuredBuffer<uint32_t> abi_check;

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

    // Render settings
    uint64_t sh_degree;
    uint64_t max_iters;
    float tmin;
    float _pad0;               // Explicit padding before 8-byte aligned pointer
    StructuredBuffer<float> tmax;
    StructuredBuffer<float4> initial_contrib;
    float max_prim_size;
    uint32_t _pad1;            // Explicit padding before 8-byte aligned handle
    OptixTraversableHandle handle;
};

// Verify critical field offsets for ABI stability
static_assert(offsetof(Params, abi_check) == 0);
static_assert(offsetof(Params, image) == 16);
static_assert(offsetof(Params, handle) == sizeof(Params) - sizeof(OptixTraversableHandle));
