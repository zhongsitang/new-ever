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

#include "OptixPipeline.h"
#include <cstring>
#include <algorithm>

// =============================================================================
// Configuration constants
// =============================================================================
namespace {
    constexpr int MAX_TRACE_DEPTH = 1;
    constexpr int NUM_PAYLOAD_VALUES = 32;  // For sorting buffer in anyhit
    constexpr int NUM_ATTRIBUTE_VALUES = 1; // Custom primitive attributes

    // Entry point names (must match slang shader function names)
    // Slang automatically adds __raygen__, __miss__, etc. prefixes
    constexpr const char* RAYGEN_ENTRY = "__raygen__rg_float";
    constexpr const char* MISS_ENTRY = "__miss__ms";
    constexpr const char* ANYHIT_ENTRY = "__anyhit__ah";
    constexpr const char* INTERSECTION_ENTRY = "__intersection__ellipsoid";

    // Launch params variable name (slang global params)
    // This is how slang exposes global variables to OptiX
    constexpr const char* PARAMS_VAR_NAME = "SLANG_globalParams";
}

// =============================================================================
// Lifecycle
// =============================================================================

void RTPipeline::init(OptixDeviceContext ctx, int device_id, bool backward_mode) {
    m_context = ctx;
    m_device_id = device_id;
    m_backward_mode = backward_mode;

    CUDA_CHECK(cudaSetDevice(device_id));

    // Select shader variant
    const unsigned char* ir_data = backward_mode ? shaders_optixir : fast_shaders_optixir;
    size_t ir_size = backward_mode ? shaders_optixir_size : fast_shaders_optixir_size;

    createModule(ir_data, ir_size);
    createProgramGroups();
    createPipeline();
    createSBT();

    // Allocate params buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_params), sizeof(LaunchParams)));
}

void RTPipeline::destroy() {
    CUDA_CHECK(cudaSetDevice(m_device_id));

    // Free SBT records
    if (m_d_raygen_record) {
        cudaFree(reinterpret_cast<void*>(m_d_raygen_record));
        m_d_raygen_record = 0;
    }
    if (m_d_miss_record) {
        cudaFree(reinterpret_cast<void*>(m_d_miss_record));
        m_d_miss_record = 0;
    }
    if (m_d_hitgroup_record) {
        cudaFree(reinterpret_cast<void*>(m_d_hitgroup_record));
        m_d_hitgroup_record = 0;
    }

    // Free params buffer
    if (m_d_params) {
        cudaFree(reinterpret_cast<void*>(m_d_params));
        m_d_params = 0;
    }

    // Destroy OptiX objects
    if (m_pipeline) {
        optixPipelineDestroy(m_pipeline);
        m_pipeline = nullptr;
    }
    if (m_raygen_pg) {
        optixProgramGroupDestroy(m_raygen_pg);
        m_raygen_pg = nullptr;
    }
    if (m_miss_pg) {
        optixProgramGroupDestroy(m_miss_pg);
        m_miss_pg = nullptr;
    }
    if (m_hitgroup_pg) {
        optixProgramGroupDestroy(m_hitgroup_pg);
        m_hitgroup_pg = nullptr;
    }
    if (m_module) {
        optixModuleDestroy(m_module);
        m_module = nullptr;
    }

    m_sbt = {};
}

RTPipeline::~OptixPipeline() {
    if (m_device_id >= 0) {
        destroy();
    }
}

RTPipeline::OptixPipeline(OptixPipeline&& other) noexcept
    : m_context(std::exchange(other.m_context, nullptr))
    , m_module(std::exchange(other.m_module, nullptr))
    , m_pipeline(std::exchange(other.m_pipeline, nullptr))
    , m_raygen_pg(std::exchange(other.m_raygen_pg, nullptr))
    , m_miss_pg(std::exchange(other.m_miss_pg, nullptr))
    , m_hitgroup_pg(std::exchange(other.m_hitgroup_pg, nullptr))
    , m_sbt(std::exchange(other.m_sbt, {}))
    , m_d_raygen_record(std::exchange(other.m_d_raygen_record, 0))
    , m_d_miss_record(std::exchange(other.m_d_miss_record, 0))
    , m_d_hitgroup_record(std::exchange(other.m_d_hitgroup_record, 0))
    , m_d_params(std::exchange(other.m_d_params, 0))
    , m_device_id(std::exchange(other.m_device_id, -1))
    , m_backward_mode(other.m_backward_mode)
    , m_pipeline_compile_options(other.m_pipeline_compile_options)
{}

OptixPipeline& RTPipeline::operator=(OptixPipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        m_context = std::exchange(other.m_context, nullptr);
        m_module = std::exchange(other.m_module, nullptr);
        m_pipeline = std::exchange(other.m_pipeline, nullptr);
        m_raygen_pg = std::exchange(other.m_raygen_pg, nullptr);
        m_miss_pg = std::exchange(other.m_miss_pg, nullptr);
        m_hitgroup_pg = std::exchange(other.m_hitgroup_pg, nullptr);
        m_sbt = std::exchange(other.m_sbt, {});
        m_d_raygen_record = std::exchange(other.m_d_raygen_record, 0);
        m_d_miss_record = std::exchange(other.m_d_miss_record, 0);
        m_d_hitgroup_record = std::exchange(other.m_d_hitgroup_record, 0);
        m_d_params = std::exchange(other.m_d_params, 0);
        m_device_id = std::exchange(other.m_device_id, -1);
        m_backward_mode = other.m_backward_mode;
        m_pipeline_compile_options = other.m_pipeline_compile_options;
    }
    return *this;
}

// =============================================================================
// Module Creation - OptiX 7.7+ API
// =============================================================================

void RTPipeline::createModule(const unsigned char* ir_data, size_t ir_size) {
    // Module compile options
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // Pipeline compile options
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipeline_compile_options.numPayloadValues = NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues = NUM_ATTRIBUTE_VALUES;
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = PARAMS_VAR_NAME;
    m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    // Create module from OptiX-IR (OptiX 7.7+ unified API)
    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &module_options,
        &m_pipeline_compile_options,
        reinterpret_cast<const char*>(ir_data),
        ir_size,
        log,
        &log_size,
        &m_module
    ));
}

// =============================================================================
// Program Groups
// =============================================================================

void RTPipeline::createProgramGroups() {
    char log[4096];
    size_t log_size;

    OptixProgramGroupOptions pg_options = {};

    // Ray generation
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = m_module;
        desc.raygen.entryFunctionName = RAYGEN_ENTRY;

        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &pg_options, log, &log_size, &m_raygen_pg
        ));
    }

    // Miss
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = m_module;
        desc.miss.entryFunctionName = MISS_ENTRY;

        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &pg_options, log, &log_size, &m_miss_pg
        ));
    }

    // Hit group (intersection + anyhit for custom primitives)
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = m_module;
        desc.hitgroup.entryFunctionNameIS = INTERSECTION_ENTRY;
        desc.hitgroup.moduleAH = m_module;
        desc.hitgroup.entryFunctionNameAH = ANYHIT_ENTRY;
        // No closest hit needed for volume rendering

        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &pg_options, log, &log_size, &m_hitgroup_pg
        ));
    }
}

// =============================================================================
// Pipeline
// =============================================================================

void RTPipeline::createPipeline() {
    OptixProgramGroup program_groups[] = { m_raygen_pg, m_miss_pg, m_hitgroup_pg };
    const int num_program_groups = sizeof(program_groups) / sizeof(program_groups[0]);

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = MAX_TRACE_DEPTH;

    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &m_pipeline_compile_options,
        &link_options,
        program_groups,
        num_program_groups,
        log,
        &log_size,
        &m_pipeline
    ));

    // Compute and set stack sizes
    OptixStackSizes stack_sizes = {};
    for (auto pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, m_pipeline));
    }

    uint32_t dc_from_traversal, dc_from_state, continuation;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        MAX_TRACE_DEPTH,
        0,  // maxCCDepth
        0,  // maxDCDepth
        &dc_from_traversal,
        &dc_from_state,
        &continuation
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        dc_from_traversal,
        dc_from_state,
        continuation,
        1  // maxTraversableGraphDepth
    ));
}

// =============================================================================
// Shader Binding Table
// =============================================================================

void RTPipeline::createSBT() {
    // Ray generation record
    {
        RayGenRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_pg, &record));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_raygen_record), sizeof(RayGenRecord)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_d_raygen_record),
            &record,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
        ));
    }

    // Miss record
    {
        MissRecord record = {};
        record.data.bg_color = make_float3(0.0f, 0.0f, 0.0f);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_pg, &record));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_miss_record), sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_d_miss_record),
            &record,
            sizeof(MissRecord),
            cudaMemcpyHostToDevice
        ));
    }

    // Hit group record
    {
        HitGroupRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_pg, &record));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_hitgroup_record), sizeof(HitGroupRecord)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_d_hitgroup_record),
            &record,
            sizeof(HitGroupRecord),
            cudaMemcpyHostToDevice
        ));
    }

    // Setup SBT
    m_sbt.raygenRecord = m_d_raygen_record;
    m_sbt.missRecordBase = m_d_miss_record;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = m_d_hitgroup_record;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount = 1;
}

// =============================================================================
// Launch
// =============================================================================

void RTPipeline::launch(const LaunchParams& params, uint32_t width, uint32_t height, CUstream stream) {
    CUDA_CHECK(cudaSetDevice(m_device_id));

    // Upload params to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_d_params),
        &params,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice
    ));

    // Launch
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        stream,
        m_d_params,
        sizeof(LaunchParams),
        &m_sbt,
        width,
        height,
        1  // depth
    ));

    CUDA_SYNC_CHECK();
}
