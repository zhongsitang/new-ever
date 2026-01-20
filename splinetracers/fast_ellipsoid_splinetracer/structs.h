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
#include <cuda_fp16.h>
#include <optix.h>
#include <optix_types.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// StructuredBuffer<T> must match slang's (RW)StructuredBuffer layout
// Slang generates: { T* data, size_t count } = 16 bytes on 64-bit
template <typename T> struct StructuredBuffer {
  T *data;
  size_t size;
};
static_assert(sizeof(StructuredBuffer<float>) == 16, "StructuredBuffer size must be 16 bytes");


struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

// SplineState must match slang spline-machine.slang layout EXACTLY
// Layout verified:
//   offset 0:  float2 distortion_parts (8 bytes)
//   offset 8:  float2 cum_sum (8 bytes)
//   offset 16: float3 padding (12 bytes)
//   offset 28: float t (4 bytes)
//   offset 32: float4 drgb (16 bytes) - naturally 16-byte aligned
//   offset 48: float logT (4 bytes)
//   offset 52: float3 C (12 bytes)
//   Total: 64 bytes
struct SplineState {
  float2 distortion_parts;
  float2 cum_sum;
  float3 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};
static_assert(sizeof(SplineState) == 64, "SplineState size must be 64 bytes to match slang");
static_assert(offsetof(SplineState, drgb) == 32, "SplineState::drgb must be at offset 32");

// Always on GPU
struct Primitives {
  __half *half_attribs;
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
