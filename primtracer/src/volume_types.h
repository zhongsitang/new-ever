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

// Always on GPU
struct Primitives {
  float3 *means;
  float3 *scales;
  float4 *quats;
  float *densities;
  size_t num_prims;
  float *features;
  size_t feature_size;

  OptixAabb *aabbs;  // Set by RayPipeline, managed by DeviceResources

  /// Compute AABBs for all primitives (aabbs buffer must be pre-allocated)
  void compute_aabbs() const;
};

struct Cam {
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};
