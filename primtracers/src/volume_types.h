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
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_types.h>

// =============================================================================
// OptiX SBT Record Template
// =============================================================================

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// =============================================================================
// Structured Buffer (matches Slang StructuredBuffer layout)
// =============================================================================

template <typename T>
struct StructuredBuffer {
    T *data;
    size_t size;
};

// =============================================================================
// Hit Data: primitive parameters at intersection
// =============================================================================

struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float density;
};

// =============================================================================
// Volume Rendering State (must match Slang VolumeState layout exactly)
// =============================================================================

struct VolumeState
{
    // Distortion loss computation
    float2 distortion_parts;
    float2 cum_sum;

    // Depth accumulation (stored in padding[0])
    float3 padding;

    // Current ray parameter
    float t;

    // Accumulated density and density-weighted color: (sigma, sigma*r, sigma*g, sigma*b)
    float4 accumulated_drgb;

    // Volume rendering state
    float log_transmittance;  // log(T) for numerical stability
    float3 color;             // Accumulated color C = sum(w_i * c_i)
};

// =============================================================================
// Primitive Collection (GPU-resident)
// =============================================================================

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

// =============================================================================
// Camera Parameters (must match Slang Camera layout exactly)
// =============================================================================

struct Cam {
    float fx, fy;           // Focal lengths
    int image_height;       // Image dimensions
    int image_width;
    float3 U, V, W;         // Camera basis vectors
    float3 eye;             // Camera position
};
