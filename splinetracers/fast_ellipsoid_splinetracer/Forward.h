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

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "structs.h"

// =============================================================================
// Modern C++20 OptiX Pipeline for Spline-based Volume Rendering
// =============================================================================

namespace optix_pipeline {

// -----------------------------------------------------------------------------
// SBT Record Types (OPTIX_SBT_RECORD_ALIGNMENT compliant)
// -----------------------------------------------------------------------------

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecordHeader {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// -----------------------------------------------------------------------------
// Launch Parameters (matches Slang SLANG_globalParams)
// -----------------------------------------------------------------------------

struct Params {
    // Output buffers
    StructuredBuffer<float4> image;
    StructuredBuffer<uint32_t> iters;
    StructuredBuffer<uint32_t> last_face;
    StructuredBuffer<uint32_t> touch_count;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int> tri_collection;

    // Ray data
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    // Primitive attributes
    StructuredBuffer<__half> half_attribs;
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Rendering parameters
    size_t sh_degree;
    size_t max_iters;
    float tmin;
    float tmax;
    StructuredBuffer<float4> initial_drgb;
    float max_prim_size;
    OptixTraversableHandle handle;
};

// -----------------------------------------------------------------------------
// Pipeline Configuration
// -----------------------------------------------------------------------------

struct PipelineConfig {
    uint32_t num_payload_values = 32;
    uint32_t num_attribute_values = 1;
    uint32_t max_trace_depth = 1;
    OptixCompileOptimizationLevel opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    OptixCompileDebugLevel debug_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    std::string_view launch_params_name = "SLANG_globalParams";
};

// -----------------------------------------------------------------------------
// Ray Trace Launch Configuration
// -----------------------------------------------------------------------------

struct LaunchConfig {
    OptixTraversableHandle handle;
    size_t num_rays;
    float3* ray_origins;
    float3* ray_directions;
    void* image_out;
    uint32_t sh_degree;
    float tmin;
    float tmax;
    float4* initial_drgb;
    Cam* camera = nullptr;
    size_t max_iters = 10000;
    float max_prim_size = 3.0f;

    // Optional debug outputs
    uint32_t* iters = nullptr;
    uint32_t* last_face = nullptr;
    uint32_t* touch_count = nullptr;
    float4* last_dirac = nullptr;
    SplineState* last_state = nullptr;
    int* tri_collection = nullptr;
    int* d_touch_count = nullptr;
    int* d_touch_inds = nullptr;
};

// -----------------------------------------------------------------------------
// Forward Pipeline Class
// -----------------------------------------------------------------------------

class Forward {
public:
    // Construction / Destruction
    Forward() = default;
    Forward(OptixDeviceContext context, int8_t device,
            const Primitives& model, bool enable_backward);
    ~Forward() noexcept(false);

    // Non-copyable, movable
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;
    Forward(Forward&& other) noexcept;
    Forward& operator=(Forward&& other) noexcept;

    // Main interface
    void trace_rays(const LaunchConfig& config);
    void reset_features(const Primitives& model);

    // Accessors
    [[nodiscard]] bool is_backward_enabled() const noexcept { return enable_backward_; }
    [[nodiscard]] size_t num_primitives() const noexcept { return num_prims_; }

private:
    // Internal initialization
    void init_module(const PipelineConfig& config);
    void init_program_groups();
    void init_pipeline(const PipelineConfig& config);
    void init_sbt();
    void init_params(const Primitives& model);
    void cleanup() noexcept;

    // OptiX handles
    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_ = {};

    // Program groups
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    // CUDA resources
    CUstream stream_ = nullptr;
    CUdeviceptr d_params_ = 0;

    // State
    int8_t device_ = -1;
    bool enable_backward_ = false;
    size_t num_prims_ = 0;
    const Primitives* model_ = nullptr;
    Params params_ = {};

    // Pipeline compile options (needed for module creation)
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
};

} // namespace optix_pipeline

// Backward compatibility alias
using Forward = optix_pipeline::Forward;
