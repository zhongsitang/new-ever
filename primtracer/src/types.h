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
///
/// NOTE: We use explicit float scalars instead of float3 for C to ensure
/// consistent memory layout across C++ and Slang. Slang may add padding
/// after float3 types, causing layout mismatch with C++.
struct IntegratorState {
    float4 accumulated_contrib;  // (density, r*d, g*d, b*d) - offset 0, 16 bytes
    float C_r, C_g, C_b;         // accumulated color RGB    - offset 16, 12 bytes
    float logT;                  // log transmittance        - offset 28, 4 bytes
    float depth_accum;           // accumulated depth        - offset 32, 4 bytes
    float t;                     // current ray parameter    - offset 36, 4 bytes
    float _pad[2];               // padding to 48 bytes      - offset 40, 8 bytes
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

/// Camera parameters for ray generation.
///
/// NOTE: We use explicit float scalars instead of float3 to ensure
/// consistent memory layout across C++ and Slang compilers.
struct Camera {
    float fx, fy;                        // focal lengths        - offset 0, 8 bytes
    int height, width;                   // image dimensions     - offset 8, 8 bytes
    float U_x, U_y, U_z;                 // camera right vector  - offset 16, 12 bytes
    float V_x, V_y, V_z;                 // camera up vector     - offset 28, 12 bytes
    float W_x, W_y, W_z;                 // camera forward vector- offset 40, 12 bytes
    float eye_x, eye_y, eye_z;           // camera position      - offset 52, 12 bytes
};                                       // total: 64 bytes

static_assert(sizeof(Camera) == 64, "Camera size must be 64 bytes for GPU compatibility");

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

    // Render settings
    size_t sh_degree;
    size_t max_iters;
    float tmin;
    StructuredBuffer<float> tmax;
    StructuredBuffer<float4> initial_contrib;
    float max_prim_size;
    OptixTraversableHandle handle;
};
