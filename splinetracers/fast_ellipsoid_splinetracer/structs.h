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

#include <cuda_fp16.h>
#include <optix.h>
#include <optix_types.h>

#include "glm/glm.hpp"

// ============================================================================
// Spline State Structure
// ============================================================================
//
// This structure matches the layout expected by Slang shaders.
// Used for storing intermediate rendering state during ray marching.
// ============================================================================

struct __align__(16) SplineState {
    float2 distortion_parts;
    float2 cum_sum;
    float3 padding;
    float  t;
    float4 drgb;
    float  logT;
    float3 C;
};

// ============================================================================
// Hit Data Structure
// ============================================================================

struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float  height;
};

// ============================================================================
// Primitives Structure
// ============================================================================
//
// Contains all GPU-resident data for the scene primitives (ellipsoids).
// All pointers are device pointers.
// ============================================================================

struct Primitives {
    // Geometry attributes
    __half* half_attribs;     // Half-precision attributes (legacy)
    float3* means;            // Ellipsoid centers
    float3* scales;           // Ellipsoid scales (radii)
    float4* quats;            // Ellipsoid orientations (quaternions)
    float*  densities;        // Opacity/density values

    // Counts
    size_t  num_prims;        // Number of primitives

    // Feature data (for spherical harmonics)
    float*  features;         // SH coefficients
    size_t  feature_size;     // Features per primitive

    // Acceleration structure
    OptixAabb* aabbs;         // Axis-aligned bounding boxes
    size_t prev_alloc_size;   // Previous allocation size (for reallocation)
};

// ============================================================================
// Camera Structure
// ============================================================================

struct Cam {
    float  fx, fy;            // Focal lengths
    int    width, height;     // Image dimensions
    float3 U, V, W;           // Camera basis vectors
    float3 eye;               // Camera position
};
