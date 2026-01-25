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

/// Slang StructuredBuffer layout: {T* data, size_t size}
template <typename T>
struct StructuredBuffer {
    T* data;
    size_t size;
};

// =============================================================================
// Primitive Data
// =============================================================================

/// Ellipsoid primitive geometry (GPU pointers)
/// All vector data uses scalar float pointers for safe torch tensor interop.
/// - means: (N, 3) flattened - access as means[i*3 + component]
/// - scales: (N, 3) flattened - access as scales[i*3 + component]
/// - quats: (N, 4) flattened - access as quats[i*4 + component], (w,x,y,z)
struct Primitives {
    float* means;         // (N, 3) centers
    float* scales;        // (N, 3) radii
    float* quats;         // (N, 4) quaternion (w,x,y,z)
    float* densities;     // (N,)
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
/// Vector data uses scalar float pointers for safe torch tensor interop.
struct SavedState {
    IntegratorState* states;     // (M,) per-ray integrator state
    float* delta_contribs;       // (M, 4) last delta contribution
    uint32_t* iters;             // (M,) iteration count per ray
    uint32_t* prim_hits;         // (N,) hit count per primitive
    int32_t* hit_collection;     // (M * max_iters,) hit primitive indices
    float* initial_contrib;      // (M, 4) contribution for rays starting inside
    int32_t* initial_prim_indices; // (N,) primitives containing ray origins
    int32_t* initial_prim_count; // (1,) count of initial_prim_indices
};

// =============================================================================
// OptiX Launch Parameters (must match slang layout)
// =============================================================================

struct Camera {
    float fx, fy;
    int height, width;
    float3 U, V, W;
    float3 eye;
};

/// OptiX launch parameters.
/// All vector data (float3/float4) uses scalar float buffers for safe torch interop.
/// Slang shaders use helper functions from tensor_utils.slang to reconstruct vectors.
struct Params {
    // Output buffers (scalar float for safe interop)
    StructuredBuffer<float> image;           // (M, 4) RGBA
    StructuredBuffer<float> depth_out;       // (M,)
    StructuredBuffer<uint32_t> iters;        // (M,)
    StructuredBuffer<uint32_t> last_prim;    // (M,)
    StructuredBuffer<uint32_t> prim_hits;    // (N,)
    StructuredBuffer<float> last_delta_contrib; // (M, 4)
    StructuredBuffer<IntegratorState> last_state; // (M,)
    StructuredBuffer<int32_t> hit_collection; // (M * max_iters,)

    // Ray data (scalar float for safe interop)
    StructuredBuffer<float> ray_origins;     // (M, 3)
    StructuredBuffer<float> ray_directions;  // (M, 3)
    Camera camera;

    // Primitive data (scalar float for safe interop)
    StructuredBuffer<float> means;           // (N, 3)
    StructuredBuffer<float> scales;          // (N, 3)
    StructuredBuffer<float> quats;           // (N, 4)
    StructuredBuffer<float> densities;       // (N,)
    StructuredBuffer<float> features;        // (N, feature_size, 3)

    // Render settings
    size_t sh_degree;
    size_t max_iters;
    float tmin;
    StructuredBuffer<float> tmax;            // (M,)
    StructuredBuffer<float> initial_contrib; // (M, 4)
    float max_prim_size;
    OptixTraversableHandle handle;
};
