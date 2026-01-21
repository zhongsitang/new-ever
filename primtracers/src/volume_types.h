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

/// Volume rendering state - must match slang IntegratorState layout exactly
struct IntegratorState
{
  // Distortion loss components (for regularization)
  float2 distortion_parts;
  float2 cum_sum;

  // Depth accumulator ([0] used, [1-2] reserved for memory layout compatibility)
  float3 depth_accum;

  // Current ray parameter t
  float t;

  // Accumulated density-weighted contributions: (density, r*density, g*density, b*density)
  float4 accumulated_contrib;

  // Volume rendering state
  float logT;   // Log of accumulated optical depth (negative log transmittance)
  float3 C;     // Accumulated color
};

// Always on GPU
struct Primitives {
  float3 *means;
  float3 *scales;
  float4 *quats;
  float *densities;
  size_t num_prims;
  float *features;
  size_t feature_size;

  OptixAabb *aabbs;
  size_t prev_alloc_size;
};

struct Cam {
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};
