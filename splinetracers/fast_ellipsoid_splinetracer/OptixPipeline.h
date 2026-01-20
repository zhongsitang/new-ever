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

// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cstdint>
#include <string>
#include "structs.h"
#include "exception.h"

// Forward declarations for embedded OptiX-IR data
extern "C" {
extern const unsigned char shaders_optixir[];
extern const size_t shaders_optixir_size;
extern const unsigned char fast_shaders_optixir[];
extern const size_t fast_shaders_optixir_size;
}

// =============================================================================
// LaunchParams - Must match global variables in slang shaders EXACTLY
//
// CRITICAL: Memory alignment between C++ and Slang
// - Slang generates SLANG_globalParams struct from global shader variables
// - The order, types, and alignment must match the slang global declarations
// - StructuredBuffer<T> in both C++ and Slang is {T* data, size_t count} = 16 bytes
// - Use explicit padding to ensure alignment matches across compilers
//
// Slang global declarations (shaders.slang lines 36-61):
//   RWStructuredBuffer<float4>      image;
//   RWStructuredBuffer<uint>        iters;
//   RWStructuredBuffer<uint>        last_face;
//   RWStructuredBuffer<uint>        touch_count;
//   RWStructuredBuffer<float4>      last_dirac;
//   RWStructuredBuffer<SplineState> last_state;
//   RWStructuredBuffer<int>         tri_collection;
//   StructuredBuffer<float3>        ray_origins;
//   StructuredBuffer<float3>        ray_directions;
//   Camera camera;
//   RWStructuredBuffer<float>       half_attribs;  // NOTE: float in slang!
//   RWStructuredBuffer<float3>      means;
//   RWStructuredBuffer<float3>      scales;
//   RWStructuredBuffer<float4>      quats;
//   RWStructuredBuffer<float>       densities;
//   RWStructuredBuffer<float>       features;
//   size_t sh_degree;
//   size_t max_iters;
//   float tmin;
//   float tmax;
//   RWStructuredBuffer<float4>      initial_drgb;
//   float max_prim_size;
//   RaytracingAccelerationStructure traversable;
// =============================================================================

// Camera struct - must match slang Camera exactly
// Layout: fx(4) + fy(4) + height(4) + width(4) + U(12) + V(12) + W(12) + eye(12) = 64 bytes
struct alignas(8) Camera {
    float fx, fy;           // offset 0-7
    int32_t height;         // offset 8-11
    int32_t width;          // offset 12-15
    float3 U, V, W;         // offset 16-51 (3 x 12 bytes)
    float3 eye;             // offset 52-63
};
static_assert(sizeof(Camera) == 64, "Camera struct size mismatch");

struct alignas(8) LaunchParams {
    // Output buffers (RWStructuredBuffer in slang) - each 16 bytes
    StructuredBuffer<float4> image;              // offset 0
    StructuredBuffer<uint32_t> iters;            // offset 16
    StructuredBuffer<uint32_t> last_face;        // offset 32
    StructuredBuffer<uint32_t> touch_count;      // offset 48
    StructuredBuffer<float4> last_dirac;         // offset 64
    StructuredBuffer<SplineState> last_state;    // offset 80
    StructuredBuffer<int32_t> tri_collection;    // offset 96

    // Input buffers (StructuredBuffer in slang)
    StructuredBuffer<float3> ray_origins;        // offset 112
    StructuredBuffer<float3> ray_directions;     // offset 128

    // Camera struct - 64 bytes
    Camera camera;                               // offset 144

    // Primitive attributes (RWStructuredBuffer)
    // NOTE: slang declares half_attribs as RWStructuredBuffer<float>, not half!
    // The buffer contains half data but slang interface uses float for simplicity
    // StructuredBuffer<T> layout is {T*, size_t}, pointer type doesn't affect layout
    StructuredBuffer<float> half_attribs;        // offset 208 (changed from __half)
    StructuredBuffer<float3> means;              // offset 224
    StructuredBuffer<float3> scales;             // offset 240
    StructuredBuffer<float4> quats;              // offset 256
    StructuredBuffer<float> densities;           // offset 272
    StructuredBuffer<float> features;            // offset 288

    // Scalar parameters - alignment analysis:
    // offset 304: sh_degree (8 bytes) - 304 % 8 == 0, OK
    // offset 312: max_iters (8 bytes) - 312 % 8 == 0, OK
    // offset 320: tmin (4 bytes)
    // offset 324: tmax (4 bytes)
    // offset 328: initial_drgb needs 8-byte alignment, 328 % 8 == 0, OK - NO PADDING NEEDED
    // offset 344: max_prim_size (4 bytes)
    // offset 348: need 4 bytes padding for handle (8-byte alignment)
    // offset 352: handle (8 bytes) - 352 % 8 == 0, OK
    uint64_t sh_degree;                          // offset 304
    uint64_t max_iters;                          // offset 312
    float tmin;                                  // offset 320
    float tmax;                                  // offset 324
    StructuredBuffer<float4> initial_drgb;       // offset 328 (8-byte aligned: 328/8=41)
    float max_prim_size;                         // offset 344
    uint32_t _pad0;                              // offset 348 - padding for 8-byte alignment of handle

    // Acceleration structure handle (OptixTraversableHandle = uint64_t)
    OptixTraversableHandle handle;               // offset 352
};

// Verify critical layout assumptions at compile time
static_assert(sizeof(StructuredBuffer<float4>) == 16, "StructuredBuffer size mismatch");
static_assert(alignof(StructuredBuffer<float4>) == 8, "StructuredBuffer alignment mismatch");
static_assert(offsetof(LaunchParams, camera) == 144, "Camera offset mismatch");
static_assert(offsetof(LaunchParams, half_attribs) == 208, "half_attribs offset mismatch");
static_assert(offsetof(LaunchParams, sh_degree) == 304, "sh_degree offset mismatch");
static_assert(offsetof(LaunchParams, initial_drgb) == 328, "initial_drgb offset mismatch");
static_assert(offsetof(LaunchParams, max_prim_size) == 344, "max_prim_size offset mismatch");
static_assert(offsetof(LaunchParams, handle) == 352, "handle offset mismatch");
static_assert(sizeof(LaunchParams) == 360, "LaunchParams total size mismatch");

// =============================================================================
// SBT Record Types (simplified)
// =============================================================================
struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenRecord = SbtRecord<RayGenData>;
using MissRecord = SbtRecord<MissData>;
using HitGroupRecord = SbtRecord<HitGroupData>;

// =============================================================================
// RTPipeline - Modern, simplified OptiX pipeline management
//
// NOTE: Named RTPipeline (not OptixPipeline) to avoid conflict with
// OptiX SDK's OptixPipeline typedef in optix_types.h
//
// Key improvements over the old Forward class:
// 1. Clear separation of initialization phases (module, program groups, pipeline, SBT)
// 2. Proper use of OptiX 7.7+ API (optixModuleCreate instead of optixModuleCreateFromPTX)
// 3. Simplified error handling
// 4. Clean resource management with RAII
// =============================================================================
class RTPipeline {
public:
    RTPipeline() = default;

    // Initialize with context and device
    // backward_mode: true uses shaders.slang (with gradient tracking)
    //                false uses fast_shaders.slang (optimized forward pass)
    void init(OptixDeviceContext ctx, int device_id, bool backward_mode = false);

    // Clean up resources
    void destroy();

    ~RTPipeline();

    // Non-copyable
    RTPipeline(const RTPipeline&) = delete;
    RTPipeline& operator=(const RTPipeline&) = delete;

    // Move semantics
    RTPipeline(RTPipeline&& other) noexcept;
    RTPipeline& operator=(RTPipeline&& other) noexcept;

    // Launch ray tracing
    void launch(const LaunchParams& params, uint32_t width, uint32_t height, CUstream stream = nullptr);

    bool isValid() const { return m_pipeline != nullptr; }

private:
    void createModule(const unsigned char* ir_data, size_t ir_size);
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // OptiX handles
    OptixDeviceContext m_context = nullptr;
    OptixModule m_module = nullptr;
    OptixPipeline_t m_pipeline = nullptr;

    // Program groups
    OptixProgramGroup m_raygen_pg = nullptr;
    OptixProgramGroup m_miss_pg = nullptr;
    OptixProgramGroup m_hitgroup_pg = nullptr;

    // Shader Binding Table
    OptixShaderBindingTable m_sbt = {};
    CUdeviceptr m_d_raygen_record = 0;
    CUdeviceptr m_d_miss_record = 0;
    CUdeviceptr m_d_hitgroup_record = 0;

    // Launch params buffer (persistent GPU allocation)
    CUdeviceptr m_d_params = 0;

    // Device info
    int m_device_id = -1;
    bool m_backward_mode = false;

    // Compile options (stored for pipeline creation)
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
};
