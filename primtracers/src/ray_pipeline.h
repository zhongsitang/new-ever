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
#include <stdexcept>
#include "volume_types.h"

using uint = uint32_t;

// Embedded PTX code (generated header)
#include "shaders_ptx.h"

// SBT record types
struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// Launch parameters - must match slang layout exactly
// Note: StructuredBuffer<T> = {T* data, size_t size} (16 bytes with padding)
struct Params {
    StructuredBuffer<float4> image;                    // Rendered RGBA output
    StructuredBuffer<uint> iters;                      // Iteration count per ray
    StructuredBuffer<uint> last_prim;                  // Last primitive hit
    StructuredBuffer<uint> primitive_hit_count;        // Hit count per primitive
    StructuredBuffer<float4> last_delta_contrib;       // Last sample delta contribution
    StructuredBuffer<VolumeState> last_state;          // Final volume state per ray
    StructuredBuffer<int> hit_collection;              // Collected hit IDs for backward pass
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<float3> means;                    // Ellipsoid centers
    StructuredBuffer<float3> scales;                   // Ellipsoid radii
    StructuredBuffer<float4> quats;                    // Ellipsoid rotations (quaternion wxyz)
    StructuredBuffer<float> densities;                 // Ellipsoid densities
    StructuredBuffer<float> features;                  // SH coefficients for color

    size_t sh_degree;                                  // Spherical harmonics degree
    size_t max_hits;                                   // Maximum hits per ray
    float tmin;                                        // Minimum ray t
    float tmax;                                        // Maximum ray t
    StructuredBuffer<float4> initial_contrib;          // Initial accumulated contribution
    float max_prim_size;                               // Maximum primitive size
    OptixTraversableHandle handle;                     // BVH acceleration structure
};

class RayPipeline {
public:
    RayPipeline() = default;
    RayPipeline(OptixDeviceContext context, int8_t device, const Primitives& model, bool enable_backward);
    ~RayPipeline() noexcept(false);

    // Non-copyable
    RayPipeline(const RayPipeline&) = delete;
    RayPipeline& operator=(const RayPipeline&) = delete;

    void trace_rays(
        OptixTraversableHandle handle,
        size_t num_rays,
        float3* ray_origins,
        float3* ray_directions,
        float4* image_out,
        uint sh_degree,
        float tmin, float tmax,
        float4* initial_contrib,
        Cam* camera = nullptr,
        size_t max_hits = 10000,
        float max_prim_size = 3.0f,
        uint* iters = nullptr,
        uint* last_prim = nullptr,
        uint* primitive_hit_count = nullptr,
        float4* last_delta_contrib = nullptr,
        VolumeState* last_state = nullptr,
        int* hit_collection = nullptr,
        int* d_hit_count = nullptr,
        int* d_hit_inds = nullptr
    );

    void reset_features(const Primitives& model);

    bool enable_backward = false;
    size_t num_prims = 0;

private:
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    OptixDeviceContext context_ = nullptr;
    int8_t device_ = -1;
    const Primitives* model_ = nullptr;

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
};
