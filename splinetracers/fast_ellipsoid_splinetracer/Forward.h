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
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "structs.h"

using uint = uint32_t;

// Forward declarations for embedded OptiX-IR data (per-entry modules)
// These are defined in the generated .c files from bin2c
extern "C" {
extern const unsigned char shader_raygen_optixir[];
extern const size_t shader_raygen_optixir_size;
extern const unsigned char shader_miss_optixir[];
extern const size_t shader_miss_optixir_size;
extern const unsigned char shader_intersection_optixir[];
extern const size_t shader_intersection_optixir_size;
extern const unsigned char shader_anyhit_optixir[];
extern const size_t shader_anyhit_optixir_size;
}

struct RayGenData
{
    // No data needed
};
struct MissData
{
    float3 bg_color;
};
typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;

struct HitGroupData {
};
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct Params
{
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

    StructuredBuffer<__half> half_attribs;

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
    Forward(
        const OptixDeviceContext &context,
        int8_t device,
        const Primitives &model,
        const bool enable_backward);
    ~Forward() noexcept(false);
    void trace_rays(const OptixTraversableHandle &handle,
                    const size_t num_rays,
                    float3 *ray_origins,
                    float3 *ray_directions,
                    void *image_out,
                    uint sh_degree,
                    float tmin,
                    float tmax,
                    float4 *initial_drgb,
                    Cam *camera=NULL,
                    const size_t max_iters=10000,
                    const float max_prim_size=3,
                    uint *iters=NULL,
                    uint *last_face=NULL,
                    uint *touch_count=NULL,
                    float4 *last_dirac=NULL,
                    SplineState *last_state=NULL,
                    int *tri_collection=NULL,
                    int *d_touch_count=NULL,
                    int *d_touch_inds=NULL);
   void reset_features(const Primitives &model);
   bool enable_backward = false;
   size_t num_prims = 0;
   private:
    Params params;
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    const Primitives *model;
    // Local fields used for this pipeline - multiple modules for per-entry compilation
    OptixModule module_raygen = nullptr;
    OptixModule module_miss = nullptr;
    OptixModule module_intersection = nullptr;
    OptixModule module_anyhit = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    float eps = 1e-6;
};