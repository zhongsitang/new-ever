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

/// Slang StructuredBuffer layout: {T* data, uint64 size}
/// Using uint64_t instead of size_t for cross-platform ABI consistency
template <typename T>
struct StructuredBuffer {
    T* data;
    uint64_t size;
};

static_assert(sizeof(StructuredBuffer<int>) == 16);

// Verify float3/float4 element sizes for StructuredBuffer stride consistency
// StructuredBuffer elements are tightly packed (no padding between elements)
static_assert(sizeof(float3) == 12, "float3 must be 12 bytes for correct buffer stride");
static_assert(sizeof(float4) == 16, "float4 must be 16 bytes for correct buffer stride");

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
/// Memory layout must match Slang IntegratorState in volume_integrator.slang
///
/// Layout (using float4 instead of float3 for cross-platform compatibility):
///   offset  0: float4 accumulated_contrib (16 bytes)
///   offset 16: float4 C                   (16 bytes, w unused)
///   offset 32: float  logT                (4 bytes)
///   offset 36: float  depth_accum         (4 bytes)
///   offset 40: float  t                   (4 bytes)
///   offset 44: float  _pad                (4 bytes)
///   Total: 48 bytes
struct IntegratorState {
    float4 accumulated_contrib;  // (density, r*d, g*d, b*d)
    float4 C;                    // accumulated color RGB (w unused)
    float logT;                  // log transmittance
    float depth_accum;           // accumulated depth
    float t;                     // current ray parameter
    float _pad;
};

static_assert(sizeof(IntegratorState) == 48 && alignof(IntegratorState) == 16);

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

/// Camera parameters for ray generation
/// Memory layout must match Slang Camera in optix_shaders.slang
/// Using float4 instead of float3 for cross-platform compatibility (w unused)
struct Camera {
    float fx, fy;
    int height, width;
    float4 U, V, W;  // camera basis vectors (w unused)
    float4 eye;      // camera position (w unused)
};

static_assert(sizeof(Camera) == 80);

/// OptiX launch parameters
/// Note: StructuredBuffer<float3> elements use 12-byte stride (tightly packed),
/// which matches both CUDA float3 and Slang StructuredBuffer<float3> behavior.
struct alignas(16) Params {
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
    uint32_t _pad_tmin;          // align tmax to 8 bytes
    StructuredBuffer<float> tmax;
    StructuredBuffer<float4> initial_contrib;
    float max_prim_size;
    uint32_t _pad_handle;        // align handle to 8 bytes
    OptixTraversableHandle handle;
};

static_assert(alignof(Params) >= 16);
static_assert(sizeof(Params) % 16 == 0);
static_assert(offsetof(Params, tmax) % 8 == 0);
static_assert(offsetof(Params, handle) % 8 == 0);
