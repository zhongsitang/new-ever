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

#include "ray_tracer.h"

#include <optix_stack_size.h>

// Embedded PTX code (generated header)
#include "shaders_ptx.h"

// =============================================================================
// RayTracer Implementation
// =============================================================================

RayTracer::RayTracer(int device_index)
    : ctx_(DeviceContext::get(device_index))
{
    CUDA_CHECK(cudaSetDevice(ctx_.device()));

    // Create acceleration structure (empty, will be populated by update_primitives)
    accel_ = std::make_unique<AccelStructure>(ctx_);

    // Setup pipeline compile options
    pipeline_options_.usesMotionBlur = false;
    pipeline_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options_.numPayloadValues = 32;
    pipeline_options_.numAttributeValues = 1;
    pipeline_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options_.pipelineLaunchParamsVariableName = "SLANG_globalParams";
    pipeline_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    // Compile pipeline once
    create_module(shaders_ptx);
    create_program_groups();
    create_pipeline();
    create_sbt();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(Params)));

    // Allocate debug_flag buffer (always enabled, ~zero cost self-check)
    CUDA_CHECK(cudaMalloc(&d_debug_flag_, sizeof(uint32_t)));
}

void RayTracer::create_module(const char* ptx) {
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OPTIX_CHECK_LOG(optixModuleCreate(
        ctx_.context(),
        &module_options,
        &pipeline_options_,
        ptx,
        strlen(ptx),
        log_, &log_sz_,
        &module_
    ));
}

void RayTracer::create_program_groups() {
    OptixProgramGroupOptions pg_options = {};

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__render_volume";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx_.context(), &raygen_desc, 1, &pg_options, log_, &log_sz_, &raygen_pg_
    ));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__miss";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx_.context(), &miss_desc, 1, &pg_options, log_, &log_sz_, &miss_pg_
    ));

    // Hitgroup
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleAH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__collect_hits";
    hitgroup_desc.hitgroup.moduleIS = module_;
    hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__intersect_ellipsoid";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx_.context(), &hitgroup_desc, 1, &pg_options, log_, &log_sz_, &hitgroup_pg_
    ));
}

void RayTracer::create_pipeline() {
    constexpr uint32_t max_trace_depth = 1;

    OptixProgramGroup program_groups[] = {raygen_pg_, miss_pg_, hitgroup_pg_};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = max_trace_depth;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        ctx_.context(),
        &pipeline_options_,
        &link_options,
        program_groups,
        std::size(program_groups),
        log_, &log_sz_,
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

void RayTracer::create_sbt() {
    auto create_record = [this](OptixProgramGroup pg) {
        CUdeviceptr record;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&record), sizeof(SbtRecord)));
        SbtRecord sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(pg, &sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(record), &sbt,
                              sizeof(SbtRecord), cudaMemcpyHostToDevice));
        return record;
    };

    sbt_.raygenRecord = create_record(raygen_pg_);
    sbt_.missRecordBase = create_record(miss_pg_);
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = create_record(hitgroup_pg_);
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.hitgroupRecordCount = 1;
}

void RayTracer::update_primitives(const Primitives& prims) {
    CUDA_CHECK(cudaSetDevice(ctx_.device()));

    prims_ = prims;
    accel_->rebuild(prims);

    // Update params with model data
    params_.means = {prims_.means, prims_.num_prims};
    params_.scales = {prims_.scales, prims_.num_prims};
    params_.quats = {prims_.quats, prims_.num_prims};
    params_.densities = {prims_.densities, prims_.num_prims};
    params_.features = {prims_.features, prims_.num_prims * prims_.feature_size};
}

void RayTracer::trace_rays(
    size_t num_rays,
    float3* ray_origins,
    float3* ray_directions,
    float4* color_out,
    float* depth_out,
    uint32_t sh_degree,
    float tmin,
    float* tmax,
    size_t max_iters,
    SavedState* saved)
{
    if (!has_primitives()) {
        throw Exception("Must call update_primitives() before trace_rays()");
    }

    CUDA_CHECK(cudaSetDevice(ctx_.device()));

    // Allocate temporary buffer for last_prim
    uint32_t* last_prim = nullptr;
    CUDA_CHECK(cudaMalloc(&last_prim, num_rays * sizeof(uint32_t)));

    // Setup params
    params_.image = {color_out, num_rays};
    params_.depth_out = {depth_out, num_rays};
    params_.sh_degree = sh_degree;
    params_.max_prim_size = 3.0f;
    params_.max_iters = max_iters;
    params_.ray_origins = {ray_origins, num_rays};
    params_.ray_directions = {ray_directions, num_rays};
    params_.tmin = tmin;
    params_.tmax = {tmax, num_rays};
    params_.last_prim = {last_prim, num_rays};

    if (saved) {
        params_.last_state = {saved->states, num_rays};
        params_.last_delta_contrib = {saved->delta_contribs, num_rays};
        params_.hit_collection = {saved->hit_collection, num_rays * max_iters};
        params_.iters = {saved->iters, num_rays};
        params_.prim_hits = {saved->prim_hits, prims_.num_prims};

        CUDA_CHECK(cudaMemset(saved->initial_contrib, 0, num_rays * sizeof(float4)));
        params_.initial_contrib = {saved->initial_contrib, num_rays};

        init_ray_start_samples(&params_, accel_->aabbs(),
                               saved->initial_prim_count,
                               saved->initial_prim_indices);
    } else {
        params_.last_state = {nullptr, 0};
        params_.last_delta_contrib = {nullptr, 0};
        params_.hit_collection = {nullptr, 0};
        params_.iters = {nullptr, 0};
        params_.prim_hits = {nullptr, 0};
        params_.initial_contrib = {nullptr, 0};
    }

    params_.handle = accel_->handle();

    // Setup debug_flag for zero-cost layout self-check (always enabled)
    CUDA_CHECK(cudaMemset(d_debug_flag_, 0, sizeof(uint32_t)));
    params_.debug_flag = {d_debug_flag_, 1};

    // Set layout sentinel (shader verifies this value to detect struct mismatch)
    params_.layout_sentinel = 0xDEADBEEFCAFEBABEull;

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_,
                          sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline_, 0, d_param_, sizeof(Params), &sbt_,
                            num_rays, 1, 1));

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Check debug_flag for layout/stride mismatch (fail fast on first frame)
    uint32_t debug_result = 0;
    CUDA_CHECK(cudaMemcpy(&debug_result, d_debug_flag_, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    if (debug_result != 0) {
        throw Exception("Layout/stride mismatch between host and shader! "
                        "Check StructuredBuffer<float3> stride or Params layout. "
                        "Error code: " + std::to_string(debug_result));
    }

    // Free temporary buffer
    CUDA_CHECK(cudaFree(last_prim));
}

RayTracer::~RayTracer() {
    // Note: Don't use CUDA_CHECK/OPTIX_CHECK in destructor - they may throw,
    // but destructors are implicitly noexcept.

    if (d_debug_flag_)
        cudaFree(std::exchange(d_debug_flag_, nullptr));
    if (d_param_)
        cudaFree(reinterpret_cast<void*>(std::exchange(d_param_, 0)));
    if (sbt_.raygenRecord)
        cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.raygenRecord, 0)));
    if (sbt_.missRecordBase)
        cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.missRecordBase, 0)));
    if (sbt_.hitgroupRecordBase)
        cudaFree(reinterpret_cast<void*>(std::exchange(sbt_.hitgroupRecordBase, 0)));
    sbt_ = {};
    if (pipeline_)
        optixPipelineDestroy(std::exchange(pipeline_, nullptr));
    if (raygen_pg_)
        optixProgramGroupDestroy(std::exchange(raygen_pg_, nullptr));
    if (miss_pg_)
        optixProgramGroupDestroy(std::exchange(miss_pg_, nullptr));
    if (hitgroup_pg_)
        optixProgramGroupDestroy(std::exchange(hitgroup_pg_, nullptr));
    if (module_)
        optixModuleDestroy(std::exchange(module_, nullptr));

    // Note: DeviceContext is globally cached, not destroyed here
    // AccelStructure is destroyed automatically via unique_ptr
}
