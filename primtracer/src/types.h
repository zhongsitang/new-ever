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

/// Slang StructuredBuffer layout: {T* data, int32_t size}
/// Using int32_t for stable cross-platform ABI with Slang
template <typename T>
struct StructuredBuffer {
    T* data;
    int32_t size;
};

// =============================================================================
// Primitive Data
// =============================================================================

/// Ellipsoid primitive geometry (GPU pointers)
/// All vector data stored as scalar float arrays for safe torch tensor interop
struct Primitives {
    float* means;         // (N, 3) flattened
    float* scales;        // (N, 3) flattened
    float* quats;         // (N, 4) flattened, quaternion (w,x,y,z)
    float* densities;
    float* features;      // SH coefficients
    int32_t num_prims;
    int32_t feature_size;
};

// =============================================================================
// Volume Rendering State
// =============================================================================

/// Per-ray volume integrator state (48 bytes, 16-byte aligned)
/// Using float4 for C to avoid float3 alignment issues with Slang
struct IntegratorState {
    float4 accumulated_contrib;  // (density, r*d, g*d, b*d)
    float4 C;                    // accumulated color RGB (w unused, for alignment)
    float logT;                  // log transmittance
    float depth_accum;           // accumulated depth
    float t;                     // current ray parameter
    float _pad;                  // padding to 48 bytes
};

static_assert(sizeof(IntegratorState) == 48);
static_assert(alignof(IntegratorState) == 16);

/// State saved for backward gradient computation
/// All vector data stored as scalar float arrays for safe torch tensor interop
struct SavedState {
    IntegratorState* states;     // (M,) per-ray integrator state
    float* delta_contribs;       // (M, 4) flattened, last delta contribution
    int32_t* iters;              // (M,) iteration count per ray
    int32_t* prim_hits;          // (N,) hit count per primitive
    int32_t* hit_collection;     // (M * max_iters,) hit primitive indices
    float* initial_contrib;      // (M, 4) flattened, contribution for rays starting inside
    int32_t* initial_prim_indices; // (N,) primitives containing ray origins
    int32_t* initial_prim_count; // (1,) count of initial_prim_indices
};

// =============================================================================
// OptiX Launch Parameters (must match slang layout exactly)
// =============================================================================
// Memory layout organized for optimal alignment:
// - All StructuredBuffers grouped together (16-byte aligned each)
// - Scalar parameters grouped together (4-byte aligned each)
// - OptixTraversableHandle at end (8-byte aligned)

struct Params {
    // Output buffers (16-byte aligned StructuredBuffers)
    StructuredBuffer<float> image;
    StructuredBuffer<float> depth_out;
    StructuredBuffer<int32_t> iters;
    StructuredBuffer<int32_t> last_prim;
    StructuredBuffer<int32_t> prim_hits;
    StructuredBuffer<float> last_delta_contrib;
    StructuredBuffer<IntegratorState> last_state;
    StructuredBuffer<int32_t> hit_collection;

    // Input buffers
    StructuredBuffer<float> ray_origins;
    StructuredBuffer<float> ray_directions;

    // Primitive data
    StructuredBuffer<float> means;
    StructuredBuffer<float> scales;
    StructuredBuffer<float> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Per-ray parameters (StructuredBuffers grouped for alignment)
    StructuredBuffer<float> tmax;
    StructuredBuffer<float> initial_contrib;

    // Scalar parameters (4-byte aligned, grouped for 16-byte alignment)
    int32_t sh_degree;
    int32_t max_iters;
    float tmin;
    float max_prim_size;

    // Acceleration structure handle (8-byte aligned)
    OptixTraversableHandle handle;
};
