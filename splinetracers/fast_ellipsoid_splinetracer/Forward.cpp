// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#include "Forward.h"
#include "exception.h"
#include "initialize_density.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cstring>
#include <array>

#include "shaders_ptx.ptx.h"
#include "fast_shaders_ptx.ptx.h"

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// =============================================================================

Forward::Forward(OptixDeviceContext context, int8_t device,
                 const Primitives& model, bool enable_backward)
    : context_(context)
    , device_(device)
    , enable_backward_(enable_backward)
    , model_(&model)
{
    CUDA_CHECK(cudaSetDevice(device_));

    // -------------------------------------------------------------------------
    // Pipeline compile options
    // -------------------------------------------------------------------------
    pipeline_options_ = {
        .usesMotionBlur = false,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        .numPayloadValues = 32,
        .numAttributeValues = 1,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "SLANG_globalParams",
        .usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM
    };

    // -------------------------------------------------------------------------
    // Module (PTX -> OptiX module)
    // -------------------------------------------------------------------------
    OptixModuleCompileOptions module_opts = {
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE
    };

    const char* ptx = enable_backward_ ? ptx::shaders_ptx : ptx::fast_shaders_ptx;

    OPTIX_CHECK_LOG(optixModuleCreate(
        context_, &module_opts, &pipeline_options_,
        ptx, strlen(ptx),
        _log, &_log_size, &module_));

    // -------------------------------------------------------------------------
    // Program groups
    // -------------------------------------------------------------------------
    OptixProgramGroupOptions pg_opts = {};

    OptixProgramGroupDesc raygen_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = {
            .module = module_,
            .entryFunctionName = "__raygen__rg_float"
        }
    };
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &raygen_desc, 1, &pg_opts,
        _log, &_log_size, &raygen_pg_));

    OptixProgramGroupDesc miss_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = {
            .module = module_,
            .entryFunctionName = "__miss__ms"
        }
    };
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &miss_desc, 1, &pg_opts,
        _log, &_log_size, &miss_pg_));

    OptixProgramGroupDesc hit_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = {
            .moduleAH = module_,
            .entryFunctionNameAH = "__anyhit__ah",
            .moduleIS = module_,
            .entryFunctionNameIS = "__intersection__ellipsoid"
        }
    };
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &hit_desc, 1, &pg_opts,
        _log, &_log_size, &hitgroup_pg_));

    // -------------------------------------------------------------------------
    // Pipeline
    // -------------------------------------------------------------------------
    std::array<OptixProgramGroup, 3> pgs = { raygen_pg_, miss_pg_, hitgroup_pg_ };

    OptixPipelineLinkOptions link_opts = { .maxTraceDepth = 1 };

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_, &pipeline_options_, &link_opts,
        pgs.data(), pgs.size(),
        _log, &_log_size, &pipeline_));

    // Stack sizes
    OptixStackSizes stack = {};
    for (auto pg : pgs) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack, pipeline_));
    }

    uint32_t dc_trav, dc_state, cont;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack, 1, 0, 0, &dc_trav, &dc_state, &cont));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, dc_trav, dc_state, cont, 1));

    // -------------------------------------------------------------------------
    // Shader Binding Table
    // -------------------------------------------------------------------------
    SbtRecord rec = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &rec));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.raygenRecord), sizeof(SbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_.raygenRecord), &rec, sizeof(rec), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &rec));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.missRecordBase), sizeof(SbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_.missRecordBase), &rec, sizeof(rec), cudaMemcpyHostToDevice));
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.missRecordCount = 1;

    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &rec));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.hitgroupRecordBase), sizeof(SbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_.hitgroupRecordBase), &rec, sizeof(rec), cudaMemcpyHostToDevice));
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.hitgroupRecordCount = 1;

    // -------------------------------------------------------------------------
    // Launch params
    // -------------------------------------------------------------------------
    params_.means = { reinterpret_cast<float3*>(model.means), model.num_prims };
    params_.scales = { reinterpret_cast<float3*>(model.scales), model.num_prims };
    params_.quats = { reinterpret_cast<float4*>(model.quats), model.num_prims };
    params_.densities = { model.densities, model.num_prims };
    params_.features = { model.features, model.num_prims * model.feature_size };
    params_._pad0 = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_), sizeof(Params)));
}

// =============================================================================

Forward::~Forward()
{
    if (d_params_) cudaFree(reinterpret_cast<void*>(d_params_));
    if (sbt_.raygenRecord) cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord));
    if (sbt_.missRecordBase) cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase));
    if (sbt_.hitgroupRecordBase) cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase));

    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (raygen_pg_) optixProgramGroupDestroy(raygen_pg_);
    if (miss_pg_) optixProgramGroupDestroy(miss_pg_);
    if (hitgroup_pg_) optixProgramGroupDestroy(hitgroup_pg_);
    if (module_) optixModuleDestroy(module_);
}

// =============================================================================

void Forward::reset_features(const Primitives& model)
{
    params_.features = { model.features, model.num_prims * model.feature_size };
}

// =============================================================================

void Forward::trace_rays(
    OptixTraversableHandle handle, size_t num_rays,
    float3* ray_origins, float3* ray_directions,
    void* image_out, uint32_t sh_degree,
    float tmin, float tmax, float4* initial_drgb,
    Cam* camera, size_t max_iters, float max_prim_size,
    uint32_t* iters, uint32_t* last_face, uint32_t* touch_count,
    float4* last_dirac, SplineState* last_state,
    int* tri_collection, int* d_touch_count, int* d_touch_inds)
{
    CUDA_CHECK(cudaSetDevice(device_));

    // Output buffers
    params_.image = { reinterpret_cast<float4*>(image_out), num_rays };
    params_.iters = { iters, num_rays };
    params_.last_face = { last_face, num_rays };
    params_.touch_count = { touch_count, num_rays };
    params_.last_dirac = { last_dirac, num_rays };
    params_.last_state = { last_state, num_rays };
    params_.tri_collection = { tri_collection, num_rays * max_iters };

    // Ray data
    params_.ray_origins = { ray_origins, num_rays };
    params_.ray_directions = { ray_directions, num_rays };

    // Rendering params
    params_.sh_degree = sh_degree;
    params_.max_iters = max_iters;
    params_.tmin = tmin;
    params_.tmax = tmax;
    params_.max_prim_size = max_prim_size;
    params_.handle = handle;

    if (camera) {
        params_.camera = *camera;
    }

    // Initial density
    CUDA_CHECK(cudaMemset(initial_drgb, 0, num_rays * sizeof(float4)));
    params_.initial_drgb = { initial_drgb, num_rays };
    initialize_density(&params_, model_->aabbs, d_touch_count, d_touch_inds);

    // Copy params to device and launch
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_params_), &params_,
        sizeof(Params), cudaMemcpyHostToDevice));

    uint32_t w = camera ? camera->width : num_rays;
    uint32_t h = camera ? camera->height : 1;

    OPTIX_CHECK(optixLaunch(pipeline_, stream_, d_params_, sizeof(Params), &sbt_, w, h, 1));
    CUDA_SYNC_CHECK();
}
