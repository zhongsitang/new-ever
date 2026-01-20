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
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>

// ============================================================================
// Memory Alignment Helpers
// ============================================================================

#ifndef __CUDACC__
#define __align__(n) alignas(n)
#endif

// ============================================================================
// Slang-Compatible Buffer Types
// ============================================================================

// Slang's RWStructuredBuffer/StructuredBuffer maps to this layout
template <typename T>
struct __align__(16) DeviceBuffer {
    T*       data;
    uint64_t size;  // Use fixed-size type for consistent alignment
};

// ============================================================================
// Spline State (matches Slang shader definition)
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
// Camera Parameters
// ============================================================================

struct __align__(16) CameraParams {
    // 4-byte aligned members
    float  fx;
    float  fy;
    int    width;
    int    height;
    // float3 is 12 bytes, but usually aligned to 16 bytes in CUDA
    float3 U;
    float  _pad0;
    float3 V;
    float  _pad1;
    float3 W;
    float  _pad2;
    float3 eye;
    float  _pad3;
};

// ============================================================================
// Model Data Pointers (8-byte aligned section)
// ============================================================================

struct __align__(16) ModelData {
    float3* means;
    float3* scales;
    float4* quats;
    float*  densities;
    float*  features;
    __half* halfAttribs;
    uint64_t numPrims;
    uint64_t featureSize;
};

// ============================================================================
// Ray Tracing Buffers (8-byte aligned section)
// ============================================================================

struct __align__(16) RayBuffers {
    // Input
    float3* origins;
    float3* directions;
    uint64_t numRays;
    uint64_t _pad;
};

// ============================================================================
// Output Buffers (8-byte aligned section)
// ============================================================================

struct __align__(16) OutputBuffers {
    float4*      image;
    SplineState* lastState;
    float4*      lastDirac;
    int*         triCollection;
    uint32_t*    iters;
    uint32_t*    lastFace;
    uint32_t*    touchCount;
    float4*      initialDrgb;
};

// ============================================================================
// Render Parameters (4-byte aligned section)
// ============================================================================

struct __align__(16) RenderParams {
    float    tmin;
    float    tmax;
    float    maxPrimSize;
    float    sceneEpsilon;
    uint32_t shDegree;
    uint32_t maxIters;
    uint32_t _pad[2];  // Padding to 16-byte boundary
};

// ============================================================================
// Launch Parameters - Main Structure
// ============================================================================
//
// This structure is passed to OptiX shaders via pipelineLaunchParamsVariableName.
// For Slang compatibility, it must match the layout expected by the compiled shader.
//
// Memory Layout Guidelines:
// 1. 8-byte aligned members first (pointers, OptixTraversableHandle)
// 2. 4-byte aligned members second (int, uint, float)
// 3. Explicit padding for alignment
// ============================================================================

struct __align__(128) LaunchParams {
    // ========== 8-byte aligned section (pointers and handles) ==========

    // Acceleration structure handle (8 bytes)
    OptixTraversableHandle handle;

    // Output buffers (Slang RWStructuredBuffer)
    DeviceBuffer<float4>      image;
    DeviceBuffer<uint32_t>    iters;
    DeviceBuffer<uint32_t>    last_face;
    DeviceBuffer<uint32_t>    touch_count;
    DeviceBuffer<float4>      last_dirac;
    DeviceBuffer<SplineState> last_state;
    DeviceBuffer<int>         tri_collection;

    // Input ray buffers (Slang StructuredBuffer)
    DeviceBuffer<float3> ray_origins;
    DeviceBuffer<float3> ray_directions;

    // Camera parameters
    CameraParams camera;

    // Model attribute buffer (unused in modern shaders, kept for compatibility)
    DeviceBuffer<__half> half_attribs;

    // Model data buffers (Slang RWStructuredBuffer)
    DeviceBuffer<float3> means;
    DeviceBuffer<float3> scales;
    DeviceBuffer<float4> quats;
    DeviceBuffer<float>  densities;
    DeviceBuffer<float>  features;

    // ========== 4-byte aligned section (scalars) ==========

    uint32_t sh_degree;
    uint32_t max_iters;
    float    tmin;
    float    tmax;

    // Initial DRGB buffer
    DeviceBuffer<float4> initial_drgb;

    float    max_prim_size;
    float    _pad[3];  // Padding to maintain alignment
};

// Verify alignment at compile time
static_assert(sizeof(LaunchParams) % 128 == 0, "LaunchParams must be 128-byte aligned");
static_assert(alignof(LaunchParams) >= 128, "LaunchParams alignment must be at least 128 bytes");

// ============================================================================
// Utility Functions
// ============================================================================

namespace launch_params {

// Initialize a DeviceBuffer from raw pointer and size
template <typename T>
inline void setBuffer(DeviceBuffer<T>& buf, T* data, size_t size) {
    buf.data = data;
    buf.size = static_cast<uint64_t>(size);
}

// Clear a DeviceBuffer
template <typename T>
inline void clearBuffer(DeviceBuffer<T>& buf) {
    buf.data = nullptr;
    buf.size = 0;
}

}  // namespace launch_params
