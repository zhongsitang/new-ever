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
#include <cstdint>
#include <stdexcept>
#include "structs.h"

using uint = uint32_t;

// Forward declarations for embedded PTX code (null-terminated strings)
extern "C" {
extern const char shaders_ptx[];
extern const char fast_shaders_ptx[];
}

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
    StructuredBuffer<float4> image;
    StructuredBuffer<uint> iters;
    StructuredBuffer<uint> last_face;
    StructuredBuffer<uint> touch_count;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int> tri_collection;
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    size_t sh_degree;
    size_t max_iters;
    float tmin;
    float tmax;
    StructuredBuffer<float4> initial_drgb;
    float max_prim_size;
    OptixTraversableHandle handle;
};

class Forward {
public:
    Forward() = default;
    Forward(OptixDeviceContext context, int8_t device, const Primitives& model, bool enable_backward);
    ~Forward() noexcept(false);

    // Non-copyable
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    void trace_rays(
        OptixTraversableHandle handle,
        size_t num_rays,
        float3* ray_origins,
        float3* ray_directions,
        void* image_out,
        uint sh_degree,
        float tmin, float tmax,
        float4* initial_drgb,
        Cam* camera = nullptr,
        size_t max_iters = 10000,
        float max_prim_size = 3.0f,
        uint* iters = nullptr,
        uint* last_face = nullptr,
        uint* touch_count = nullptr,
        float4* last_dirac = nullptr,
        SplineState* last_state = nullptr,
        int* tri_collection = nullptr,
        int* d_touch_count = nullptr,
        int* d_touch_inds = nullptr
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
