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

// =============================================================================
// RayTracer - OptiX-based volume ray tracer for ellipsoid primitives
// =============================================================================

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <stdexcept>
#include "structs.h"

using uint = uint32_t;

// Embedded PTX code (generated at build time from shaders.slang)
#include "shaders_ptx.h"

// =============================================================================
// SBT (Shader Binding Table) record types
// =============================================================================

struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// =============================================================================
// Launch parameters - must match Slang shader layout exactly
// Note: StructuredBuffer<T> in Slang = {T* data, size_t size} (16 bytes with padding)
// =============================================================================

struct LaunchParams {
    // Output buffers
    StructuredBuffer<float4> image;              // Rendered RGBA image
    StructuredBuffer<uint> iterations;           // Iteration count per ray
    StructuredBuffer<uint> last_hit_face;        // Last hit face index
    StructuredBuffer<uint> touch_count;          // Hit count per primitive
    StructuredBuffer<float4> last_sample;        // Last sample value
    StructuredBuffer<IntegrationState> last_state;  // Final integration state
    StructuredBuffer<int> hit_sequence;          // Sequence of all hits per ray

    // Input buffers
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    // Primitive data
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Rendering parameters
    size_t sh_degree;
    size_t max_iterations;
    float t_near;
    float t_far;
    StructuredBuffer<float4> initial_sample;
    float max_prim_size;
    OptixTraversableHandle accel_handle;
};

// =============================================================================
// RayTracer class - manages OptiX pipeline for volume rendering
// =============================================================================

class RayTracer {
public:
    RayTracer() = default;
    RayTracer(OptixDeviceContext context, int8_t device, const Primitives& primitives);
    ~RayTracer() noexcept(false);

    // Non-copyable
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    // Main ray tracing function
    void trace_rays(
        OptixTraversableHandle accel_handle,
        size_t num_rays,
        float3* ray_origins,
        float3* ray_directions,
        void* image_out,
        uint sh_degree,
        float t_near,
        float t_far,
        float4* initial_sample,
        Cam* camera = nullptr,
        size_t max_iterations = 10000,
        float max_prim_size = 3.0f,
        uint* iterations = nullptr,
        uint* last_hit_face = nullptr,
        uint* touch_count = nullptr,
        float4* last_sample = nullptr,
        IntegrationState* last_state = nullptr,
        int* hit_sequence = nullptr,
        int* initial_touch_count = nullptr,
        int* initial_touch_indices = nullptr
    );

    void update_features(const Primitives& primitives);

    size_t num_primitives() const { return num_prims_; }

private:
    void create_module();
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    OptixDeviceContext context_ = nullptr;
    int8_t device_ = -1;
    const Primitives* primitives_ = nullptr;
    size_t num_prims_ = 0;

    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    OptixShaderBindingTable sbt_ = {};
    CUdeviceptr d_params_ = 0;
    CUstream stream_ = nullptr;

    LaunchParams params_ = {};
    OptixPipelineCompileOptions pipeline_options_ = {};

    // Logging buffer for OptiX
    static constexpr size_t LOG_SIZE = 2048;
    char log_[LOG_SIZE];
    size_t log_size_ = LOG_SIZE;
};

// Legacy alias for backward compatibility
using Forward = RayTracer;
