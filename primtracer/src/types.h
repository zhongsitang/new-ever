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
#include "glm/glm.hpp"
#include <optix.h>
#include <optix_types.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <typename T> struct StructuredBuffer {
  T *data;
  size_t size;
};


struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float density;
};

/// Volume rendering state per ray
///
/// Memory layout (12 floats = 48 bytes, 16-byte aligned):
///   accumulated_contrib: density-weighted contributions
///   C: accumulated color RGB
///   logT: log transmittance (for numerical stability)
///   depth_accum: accumulated depth
///   t: current ray parameter
struct IntegratorState
{
  float4 accumulated_contrib;  // density, r*d, g*d, b*d
  float3 C;                    // accumulated color
  float logT;                  // log transmittance
  float depth_accum;           // accumulated depth
  float t;                     // current t
  float _pad[2];               // padding to 48 bytes
};

static_assert(sizeof(IntegratorState) == 48, "IntegratorState must be 48 bytes");
static_assert(alignof(IntegratorState) == 16, "IntegratorState must be 16-byte aligned");

/// Primitive geometry data (GPU pointers)
struct Primitives {
  float3 *means;
  float3 *scales;
  float4 *quats;
  float *densities;
  size_t num_prims;
  float *features;
  size_t feature_size;
};

struct Cam {
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};

using uint = uint32_t;

// =============================================================================
// Launch Parameters for OptiX ray tracing
// =============================================================================

/// Launch parameters - must match slang layout exactly.
/// Note: StructuredBuffer<T> = {T* data, size_t size} (16 bytes with padding)
struct Params {
    StructuredBuffer<float4> image;                    // Rendered RGBA output
    StructuredBuffer<float> depth_out;                 // Rendered depth output
    StructuredBuffer<uint> iters;                      // Iteration count per ray
    StructuredBuffer<uint> last_prim;                  // Last primitive hit
    StructuredBuffer<uint> prim_hits;                  // Hit count per primitive
    StructuredBuffer<float4> last_delta_contrib;       // Last sample delta contribution
    StructuredBuffer<IntegratorState> last_state;      // Final volume state per ray
    StructuredBuffer<int> hit_collection;              // Collected hit IDs for backward pass
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<float3> means;                    // Ellipsoid centers
    StructuredBuffer<float3> scales;                   // Ellipsoid radii
    StructuredBuffer<float4> quats;                    // Ellipsoid rotations (quaternion wxyz)
    StructuredBuffer<float> densities;                 // Ellipsoid densities
    StructuredBuffer<float> features;                  // SH coefficients for color

    size_t sh_degree;                                  // Spherical harmonics degree
    size_t max_iters;                                  // Maximum iterations per ray
    float tmin;                                        // Minimum ray t
    StructuredBuffer<float> tmax;                      // Maximum ray t (per-ray)
    StructuredBuffer<float4> initial_contrib;          // Initial accumulated contribution
    float max_prim_size;                               // Maximum primitive size
    OptixTraversableHandle handle;                     // BVH acceleration structure
};

/// State saved during forward pass for backward gradient computation.
struct SavedState {
    IntegratorState* states;        // (M, 12) volume integrator state per ray
    float4* delta_contribs;         // (M, 4) last delta contribution
    uint* iters;                    // (M,) iteration count per ray
    uint* prim_hits;                // (N,) hit count per primitive
    int* hit_collection;            // (M * max_iters,) hit primitive indices
    float4* initial_contrib;        // (M, 4) contribution for rays starting inside
    int* initial_prim_indices;      // (N,) primitives containing ray origins
    int* initial_prim_count;        // (1,) count of initial_prim_indices
};
