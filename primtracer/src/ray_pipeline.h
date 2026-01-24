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

// Prevent Windows min/max macro pollution
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "accel_structure.h"

// Embedded PTX code (generated header)
#include "shaders_ptx.h"

// SBT record types
struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// =============================================================================
// RayPipeline - OptiX ray tracing pipeline for volume rendering
// =============================================================================

/// Complete ray tracing pipeline for ellipsoid volume rendering.
///
/// This class manages:
/// - OptiX pipeline (module, program groups, SBT)
/// - Ray tracing execution
class RayPipeline {
public:
    /// Construct a ray pipeline with primitive data.
    RayPipeline(const Primitives& prims, int device_index);

    ~RayPipeline();

    // Non-copyable
    RayPipeline(const RayPipeline&) = delete;
    RayPipeline& operator=(const RayPipeline&) = delete;

    /// Trace rays through the scene.
    void trace_rays(
        size_t num_rays,
        float3* ray_origins,
        float3* ray_directions,
        float4* color_out,
        float* depth_out,
        uint sh_degree,
        float tmin,
        float* tmax,
        size_t max_iters,
        SavedState* saved
    );

    size_t num_prims() const { return model_.num_prims; }

private:
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    DeviceContext& ctx_;
    std::unique_ptr<AccelStructure> accel_;
    Primitives model_ = {};

    // OptiX pipeline objects
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    OptixShaderBindingTable sbt_ = {};
    CUdeviceptr d_param_ = 0;
    CUstream stream_ = nullptr;

    Params params_ = {};
    OptixPipelineCompileOptions pipeline_options_ = {};

    // Log buffer for OptiX
    static constexpr size_t LOG_SIZE = 2048;
    char log_[LOG_SIZE];
    size_t log_size_ = LOG_SIZE;
};
