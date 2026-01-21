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

// =============================================================================
// OptiX Shader Binding Table record template
// =============================================================================
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// =============================================================================
// GPU buffer descriptor for Slang interop
// =============================================================================
template <typename T>
struct StructuredBuffer {
    T* data;
    size_t size;
};

// =============================================================================
// Ray-primitive intersection data
// =============================================================================
struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

// =============================================================================
// Volume integration state
// Stores accumulated values during ray marching through volume primitives
// =============================================================================
struct IntegrationState {
    // Distortion loss components (for regularization)
    float2 distortion_parts;
    float2 cum_sum;

    // Depth accumulator (stored in first component)
    float3 depth_accum;

    // Current ray parameter
    float t;

    // Accumulated density-weighted RGB sample: (density, density*R, density*G, density*B)
    float4 sample;

    // Log transmittance: log(T) where T = exp(-integral of density)
    float log_transmittance;

    // Accumulated color from volume rendering equation
    float3 color;
};

// Legacy alias for backward compatibility
using SplineState = IntegrationState;

// =============================================================================
// GPU primitive collection
// Stores all primitive data on GPU for ray tracing
// =============================================================================
struct Primitives {
    float3* means;           // Primitive centers
    float3* scales;          // Primitive axis scales
    float4* quats;           // Primitive rotations (quaternions)
    float* densities;        // Primitive peak densities
    size_t num_prims;        // Number of primitives

    float* features;         // SH coefficients for view-dependent color
    size_t feature_size;     // Number of SH coefficients per primitive

    OptixAabb* aabbs;        // Axis-aligned bounding boxes for acceleration
    size_t prev_alloc_size;  // Previous allocation size (for reuse)
};

// =============================================================================
// Camera parameters
// =============================================================================
struct Cam {
    float fx, fy;            // Focal lengths
    int height;              // Image height
    int width;               // Image width
    float3 U, V, W;          // Camera basis vectors
    float3 eye;              // Camera position
};
