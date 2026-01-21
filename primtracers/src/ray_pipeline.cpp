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

#include "ray_pipeline.h"
#include "optix_error.h"
#include "cuda_buffer.h"
#include "cuda_kernels.h"

#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cstring>

// -----------------------------------------------------------------------------
// RayPipeline implementation
// -----------------------------------------------------------------------------

RayPipeline::RayPipeline(OptixDeviceContext context, int8_t device, const Primitives& model, bool enable_backward)
    : enable_backward(enable_backward)
    , context_(context)
    , device_(device)
    , model_(&model)
{
    CUDA_CHECK(cudaSetDevice(device));

    // Setup pipeline compile options
    pipeline_options_.usesMotionBlur = false;
    pipeline_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options_.numPayloadValues = 32;
    pipeline_options_.numAttributeValues = 1;
    pipeline_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options_.pipelineLaunchParamsVariableName = "SLANG_globalParams";
    pipeline_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    create_module(shaders_ptx);
    create_program_groups();
    create_pipeline();
    create_sbt();

    // Initialize params with model data
    params_.means = {reinterpret_cast<float3*>(model.means), model.num_prims};
    params_.scales = {reinterpret_cast<float3*>(model.scales), model.num_prims};
    params_.quats = {reinterpret_cast<float4*>(model.quats), model.num_prims};
    params_.densities = {model.densities, model.num_prims};
    params_.features = {model.features, model.num_prims * model.feature_size};

    num_prims = model.num_prims;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(Params)));
}

void RayPipeline::create_module(const char* ptx) {
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OPTIX_CHECK_LOG(optixModuleCreate(
        context_,
        &module_options,
        &pipeline_options_,
        ptx,
        strlen(ptx),
        log_, &log_size_,
        &module_
    ));
}

void RayPipeline::create_program_groups() {
    OptixProgramGroupOptions pg_options = {};

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__render_volume";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &raygen_desc, 1, &pg_options, log_, &log_size_, &raygen_pg_
    ));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__miss";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &miss_desc, 1, &pg_options, log_, &log_size_, &miss_pg_
    ));

    // Hitgroup
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleAH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__collect_hits";
    hitgroup_desc.hitgroup.moduleIS = module_;
    hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__intersect_ellipsoid";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &hitgroup_desc, 1, &pg_options, log_, &log_size_, &hitgroup_pg_
    ));
}

void RayPipeline::create_pipeline() {
    constexpr uint32_t max_trace_depth = 1;

    OptixProgramGroup program_groups[] = {raygen_pg_, miss_pg_, hitgroup_pg_};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = max_trace_depth;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_,
        &pipeline_options_,
        &link_options,
        program_groups,
        std::size(program_groups),
        log_, &log_size_,
        &pipeline_
    ));

    // Compute and set stack sizes
    OptixStackSizes stack_sizes = {};
    for (auto pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline_));
    }

    uint32_t dc_from_traversal, dc_from_state, continuation;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth,
        0, 0, // CC/DC depth
        &dc_from_traversal, &dc_from_state, &continuation
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_, dc_from_traversal, dc_from_state, continuation, 1
    ));
}

void RayPipeline::create_sbt() {
    // Raygen record
    CUdeviceptr raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayGenSbtRecord)));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    // Miss record
    CUdeviceptr miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f}; // Background color
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt,
                          sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

    // Hitgroup record
    CUdeviceptr hitgroup_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), sizeof(HitGroupSbtRecord)));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt,
                          sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

    sbt_.raygenRecord = raygen_record;
    sbt_.missRecordBase = miss_record;
    sbt_.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = hitgroup_record;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt_.hitgroupRecordCount = 1;
}

void RayPipeline::trace_rays(
    OptixTraversableHandle handle,
    size_t num_rays,
    float3* ray_origins,
    float3* ray_directions,
    float4* image_out,
    uint sh_degree,
    float tmin, float tmax,
    float4* initial_contrib,
    Cam* camera,
    size_t max_hits,
    float max_prim_size,
    uint* iters,
    uint* last_prim,
    uint* primitive_hit_count,
    float4* last_delta_contrib,
    VolumeState* last_state,
    int* hit_collection,
    int* d_hit_count,
    int* d_hit_inds)
{
    CUDA_CHECK(cudaSetDevice(device_));

    // Setup params
    params_.image = {image_out, num_rays};
    params_.last_state = {last_state, num_rays};
    params_.last_delta_contrib = {last_delta_contrib, num_rays};
    params_.hit_collection = {hit_collection, num_rays * max_hits};
    params_.iters = {iters, num_rays};
    params_.last_prim = {last_prim, num_rays};
    params_.primitive_hit_count = {primitive_hit_count, num_prims};
    params_.sh_degree = sh_degree;
    params_.max_prim_size = max_prim_size;
    params_.max_hits = max_hits;
    params_.ray_origins = {ray_origins, num_rays};
    params_.ray_directions = {ray_directions, num_rays};
    params_.tmin = tmin;
    params_.tmax = tmax;

    if (camera) {
        params_.camera = *camera;
    }

    CUDA_CHECK(cudaMemset(initial_contrib, 0, num_rays * sizeof(float4)));
    params_.initial_contrib = {initial_contrib, num_rays};

    init_ray_start_samples(&params_, model_->aabbs, d_hit_count, d_hit_inds);

    params_.handle = handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_,
                          sizeof(Params), cudaMemcpyHostToDevice));

    if (camera) {
        OPTIX_CHECK(optixLaunch(pipeline_, stream_, d_param_, sizeof(Params), &sbt_,
                                camera->width, camera->height, 1));
    } else {
        OPTIX_CHECK(optixLaunch(pipeline_, stream_, d_param_, sizeof(Params), &sbt_,
                                num_rays, 1, 1));
    }

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void RayPipeline::reset_features(const Primitives& model) {
    params_.features = {model.features, model.num_prims * model.feature_size};
}

RayPipeline::~RayPipeline() noexcept(false) {
    if (d_param_)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(d_param_, 0))));
    if (sbt_.raygenRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.raygenRecord, 0))));
    if (sbt_.missRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.missRecordBase, 0))));
    if (sbt_.hitgroupRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.hitgroupRecordBase, 0))));
    if (sbt_.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.callablesRecordBase, 0))));
    if (sbt_.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.exceptionRecord, 0))));
    sbt_ = {};

    if (stream_)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream_, nullptr)));
    if (pipeline_)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline_, nullptr)));
    if (raygen_pg_)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_pg_, nullptr)));
    if (miss_pg_)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_pg_, nullptr)));
    if (hitgroup_pg_)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_pg_, nullptr)));
    if (module_)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module_, nullptr)));
}
