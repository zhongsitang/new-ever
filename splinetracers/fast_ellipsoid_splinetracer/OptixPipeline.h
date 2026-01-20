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
// LaunchParams - Must match global variables in slang shaders exactly
// Slang generates SLANG_globalParams struct from global shader variables
// The order and types must match the slang global declarations
// =============================================================================
struct LaunchParams {
    // Output buffers (RWStructuredBuffer in slang)
    StructuredBuffer<float4> image;
    StructuredBuffer<uint32_t> iters;
    StructuredBuffer<uint32_t> last_face;
    StructuredBuffer<uint32_t> touch_count;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int32_t> tri_collection;

    // Input buffers (StructuredBuffer in slang)
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;

    // Camera struct
    Cam camera;

    // Primitive attributes (RWStructuredBuffer)
    StructuredBuffer<__half> half_attribs;
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Scalar parameters
    size_t sh_degree;
    size_t max_iters;
    float tmin;
    float tmax;
    StructuredBuffer<float4> initial_drgb;
    float max_prim_size;

    // Acceleration structure handle
    OptixTraversableHandle handle;
};

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
// OptixPipeline - Modern, simplified OptiX pipeline management
//
// Key improvements over the old Forward class:
// 1. Clear separation of initialization phases (module, program groups, pipeline, SBT)
// 2. Proper use of OptiX 7.7+ API (optixModuleCreate instead of optixModuleCreateFromPTX)
// 3. Simplified error handling
// 4. Clean resource management with RAII
// =============================================================================
class OptixPipeline {
public:
    OptixPipeline() = default;

    // Initialize with context and device
    // backward_mode: true uses shaders.slang (with gradient tracking)
    //                false uses fast_shaders.slang (optimized forward pass)
    void init(OptixDeviceContext ctx, int device_id, bool backward_mode = false);

    // Clean up resources
    void destroy();

    ~OptixPipeline();

    // Non-copyable
    OptixPipeline(const OptixPipeline&) = delete;
    OptixPipeline& operator=(const OptixPipeline&) = delete;

    // Move semantics
    OptixPipeline(OptixPipeline&& other) noexcept;
    OptixPipeline& operator=(OptixPipeline&& other) noexcept;

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
