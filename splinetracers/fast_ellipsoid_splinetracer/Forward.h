// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include "structs.h"

// =============================================================================
// Launch Parameters - must match Slang global variable order exactly
// =============================================================================

struct Params {
    StructuredBuffer<float4> image;
    StructuredBuffer<uint32_t> iters;
    StructuredBuffer<uint32_t> last_face;
    StructuredBuffer<uint32_t> touch_count;
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
    uint32_t _pad0;
    OptixTraversableHandle handle;
};

// =============================================================================
// Forward Pipeline
// =============================================================================

class Forward {
public:
    Forward() = default;
    Forward(OptixDeviceContext context, int8_t device,
            const Primitives& model, bool enable_backward);
    ~Forward();

    // Non-copyable
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    void trace_rays(OptixTraversableHandle handle, size_t num_rays,
                    float3* ray_origins, float3* ray_directions,
                    void* image_out, uint32_t sh_degree,
                    float tmin, float tmax, float4* initial_drgb,
                    Cam* camera, size_t max_iters, float max_prim_size,
                    uint32_t* iters, uint32_t* last_face, uint32_t* touch_count,
                    float4* last_dirac, SplineState* last_state,
                    int* tri_collection, int* d_touch_count, int* d_touch_inds);

    void reset_features(const Primitives& model);

private:
    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_ = {};
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;
    OptixPipelineCompileOptions pipeline_options_ = {};

    CUstream stream_ = nullptr;
    CUdeviceptr d_params_ = 0;

    int8_t device_ = -1;
    bool enable_backward_ = false;
    const Primitives* model_ = nullptr;
    Params params_ = {};
};
