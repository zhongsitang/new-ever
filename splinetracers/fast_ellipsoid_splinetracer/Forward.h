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
#include <string>
#include <vector>
#include "structs.h"

using uint = uint32_t;

// Forward declaration for SlangCompiler
class SlangCompiler;

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

/**
 * Compile Slang shader to PTX at runtime using the Slang C++ API
 *
 * @param slangFilePath Path to the .slang shader file
 * @param searchPaths   Additional paths to search for imported modules
 * @return Compiled PTX code as a string
 */
std::string compileSlangToPTX(
    const std::string& slangFilePath,
    const std::vector<std::string>& searchPaths = {}
);

class Forward {
public:
    Forward() = default;

    /**
     * Constructor with pre-compiled PTX
     * @param ptx Pre-compiled PTX code (can be from compileSlangToPTX)
     */
    Forward(OptixDeviceContext context, int8_t device, const Primitives& model,
            const std::string& ptx);

    /**
     * Constructor that compiles Slang at runtime
     * @param slangFilePath Path to the .slang shader file
     * @param searchPaths   Additional paths for import resolution
     */
    Forward(OptixDeviceContext context, int8_t device, const Primitives& model,
            const std::string& slangFilePath,
            const std::vector<std::string>& searchPaths);

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
    void initialize(const std::string& ptx);
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    OptixDeviceContext context_ = nullptr;
    int8_t device_ = -1;
    const Primitives* model_ = nullptr;

    // Store PTX to keep it alive during Forward's lifetime
    std::string ptx_;

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
