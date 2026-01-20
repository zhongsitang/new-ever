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

#include "Forward.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <utility>

#include "CUDABuffer.h"
#include "exception.h"
#include "initialize_density.h"

// ============================================================================
// Helper Macros
// ============================================================================

#define SET_BUFFER(buf, ptr, count) \
    do { (buf).data = (ptr); (buf).size = (count); } while(0)

#define CLEAR_BUFFER(buf) \
    do { (buf).data = nullptr; (buf).size = 0; } while(0)

// ============================================================================
// Constructor
// ============================================================================

Forward::Forward(
    const OptixDeviceContext& context,
    int8_t device,
    const Primitives& model,
    const bool enableBackward
)
    : enable_backward(enableBackward)
    , m_context(context)
    , m_device(device)
    , m_model(&model)
{
    CUDA_CHECK(cudaSetDevice(device));

    char log[16384];
    size_t logSize;

    // ========== Module Creation ==========
    {
        logSize = sizeof(log);

        OptixModuleCompileOptions moduleOptions = {};
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        m_pipelineOptions = {};
        m_pipelineOptions.usesMotionBlur        = false;
        m_pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        m_pipelineOptions.numPayloadValues      = 32;
        m_pipelineOptions.numAttributeValues    = 2;
        m_pipelineOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
        m_pipelineOptions.pipelineLaunchParamsVariableName = "SLANG_globalParams";
        m_pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        const unsigned char* shaderData = enable_backward ? shaders_optixir : fast_shaders_optixir;
        size_t shaderSize = enable_backward ? shaders_optixir_size : fast_shaders_optixir_size;

        OPTIX_CHECK_LOG(optixModuleCreate(
            m_context, &moduleOptions, &m_pipelineOptions,
            reinterpret_cast<const char*>(shaderData), shaderSize,
            log, &logSize, &m_module
        ));
    }

    // ========== Program Group Creation ==========
    OptixProgramGroupOptions groupOptions = {};

    // Ray Generation
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = m_module;
        desc.raygen.entryFunctionName = "__raygen__rg_float";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &groupOptions, log, &logSize, &m_raygenGroup
        ));
    }

    // Miss
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module            = m_module;
        desc.miss.entryFunctionName = "__miss__ms";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &groupOptions, log, &logSize, &m_missGroup
        ));
    }

    // Hit Group
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH            = m_module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        desc.hitgroup.moduleAH            = m_module;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        desc.hitgroup.moduleIS            = m_module;
        desc.hitgroup.entryFunctionNameIS = "__intersection__ellipsoid";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &desc, 1, &groupOptions, log, &logSize, &m_hitGroup
        ));
    }

    // ========== Pipeline Creation ==========
    {
        logSize = sizeof(log);

        OptixProgramGroup programGroups[] = { m_raygenGroup, m_missGroup, m_hitGroup };

        OptixPipelineLinkOptions linkOptions = {};
        linkOptions.maxTraceDepth = 1;

        OPTIX_CHECK_LOG(optixPipelineCreate(
            m_context, &m_pipelineOptions, &linkOptions,
            programGroups, 3, log, &logSize, &m_pipeline
        ));

        // Calculate stack sizes
        OptixStackSizes stackSizes = {};
        for (auto& group : programGroups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, m_pipeline));
        }

        uint32_t directCallableFromTraversal, directCallableFromState, continuationStackSize;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stackSizes, 1, 0, 0,
            &directCallableFromTraversal, &directCallableFromState, &continuationStackSize
        ));

        OPTIX_CHECK(optixPipelineSetStackSize(
            m_pipeline, directCallableFromTraversal, directCallableFromState, continuationStackSize, 1
        ));
    }

    // ========== Shader Binding Table ==========

    // Ray Generation Record
    {
        RayGenSbtRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenGroup, &record));

        CUdeviceptr devicePtr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(record)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(devicePtr), &record, sizeof(record), cudaMemcpyHostToDevice));
        m_sbt.raygenRecord = devicePtr;
    }

    // Miss Record
    {
        MissSbtRecord record = {};
        record.data.backgroundColor = make_float3(0.3f, 0.1f, 0.2f);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_missGroup, &record));

        CUdeviceptr devicePtr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(record)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(devicePtr), &record, sizeof(record), cudaMemcpyHostToDevice));
        m_sbt.missRecordBase          = devicePtr;
        m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        m_sbt.missRecordCount         = 1;
    }

    // Hit Group Record
    {
        HitGroupSbtRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitGroup, &record));

        CUdeviceptr devicePtr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(record)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(devicePtr), &record, sizeof(record), cudaMemcpyHostToDevice));
        m_sbt.hitgroupRecordBase          = devicePtr;
        m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        m_sbt.hitgroupRecordCount         = 1;
    }

    m_sbt.callablesRecordBase = 0;
    m_sbt.callablesRecordStrideInBytes = 0;
    m_sbt.callablesRecordCount = 0;
    m_sbt.exceptionRecord = 0;

    // ========== Initialize Launch Params ==========
    std::memset(&m_params, 0, sizeof(m_params));
    SET_BUFFER(m_params.half_attribs, model.half_attribs, model.num_prims);
    SET_BUFFER(m_params.means,        reinterpret_cast<float3*>(model.means), model.num_prims);
    SET_BUFFER(m_params.scales,       reinterpret_cast<float3*>(model.scales), model.num_prims);
    SET_BUFFER(m_params.quats,        reinterpret_cast<float4*>(model.quats), model.num_prims);
    SET_BUFFER(m_params.densities,    model.densities, model.num_prims);
    SET_BUFFER(m_params.features,     model.features, model.num_prims * model.feature_size);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_dParams), sizeof(Params)));
    num_prims = model.num_prims;
}

// ============================================================================
// Trace Rays
// ============================================================================

void Forward::trace_rays(
    const OptixTraversableHandle& handle,
    size_t numRays,
    float3* rayOrigins,
    float3* rayDirections,
    void* imageOut,
    uint shDegree,
    float tmin,
    float tmax,
    float4* initialDrgb,
    Cam* camera,
    size_t maxIters,
    float maxPrimSize,
    uint* iters,
    uint* lastFace,
    uint* touchCount,
    float4* lastDirac,
    SplineState* lastState,
    int* triCollection,
    int* dTouchCount,
    int* dTouchInds
) {
    CUDA_CHECK(cudaSetDevice(m_device));

    // Update launch parameters
    SET_BUFFER(m_params.image,          reinterpret_cast<float4*>(imageOut), numRays);
    SET_BUFFER(m_params.last_state,     lastState, numRays);
    SET_BUFFER(m_params.last_dirac,     lastDirac, numRays);
    SET_BUFFER(m_params.tri_collection, triCollection, numRays * maxIters);
    SET_BUFFER(m_params.iters,          iters, numRays);
    SET_BUFFER(m_params.last_face,      lastFace, numRays);
    SET_BUFFER(m_params.touch_count,    touchCount, m_model->num_prims);
    SET_BUFFER(m_params.ray_origins,    rayOrigins, numRays);
    SET_BUFFER(m_params.ray_directions, rayDirections, numRays);

    if (camera != nullptr) {
        m_params.camera = *camera;
    }

    m_params.sh_degree     = shDegree;
    m_params.max_iters     = static_cast<uint>(maxIters);
    m_params.tmin          = tmin;
    m_params.tmax          = tmax;
    m_params.max_prim_size = maxPrimSize;
    m_params.handle        = handle;

    // Initialize DRGB if provided
    if (initialDrgb != nullptr) {
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(initialDrgb), 0, numRays * sizeof(float4)));
        SET_BUFFER(m_params.initial_drgb, initialDrgb, numRays);

        if (dTouchCount != nullptr && dTouchInds != nullptr) {
            initialize_density(&m_params, m_model->aabbs, dTouchCount, dTouchInds);
        }
    }

    // Upload params to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_dParams), &m_params, sizeof(Params), cudaMemcpyHostToDevice
    ));

    // Determine dispatch dimensions
    uint32_t width  = (camera != nullptr) ? static_cast<uint32_t>(camera->width) : static_cast<uint32_t>(numRays);
    uint32_t height = (camera != nullptr) ? static_cast<uint32_t>(camera->height) : 1;

    // Launch
    OPTIX_CHECK(optixLaunch(m_pipeline, nullptr, m_dParams, sizeof(Params), &m_sbt, width, height, 1));
    CUDA_SYNC_CHECK();
}

// ============================================================================
// Reset Features
// ============================================================================

void Forward::reset_features(const Primitives& model) {
    SET_BUFFER(m_params.features, model.features, model.num_prims * model.feature_size);
}

// ============================================================================
// Move Operations
// ============================================================================

Forward::Forward(Forward&& other) noexcept
    : enable_backward(other.enable_backward)
    , num_prims(other.num_prims)
    , m_context(std::exchange(other.m_context, nullptr))
    , m_device(other.m_device)
    , m_model(std::exchange(other.m_model, nullptr))
    , m_module(std::exchange(other.m_module, nullptr))
    , m_pipeline(std::exchange(other.m_pipeline, nullptr))
    , m_raygenGroup(std::exchange(other.m_raygenGroup, nullptr))
    , m_missGroup(std::exchange(other.m_missGroup, nullptr))
    , m_hitGroup(std::exchange(other.m_hitGroup, nullptr))
    , m_sbt(std::exchange(other.m_sbt, OptixShaderBindingTable{}))
    , m_dParams(std::exchange(other.m_dParams, 0))
    , m_params(other.m_params)
    , m_pipelineOptions(other.m_pipelineOptions)
{
}

Forward& Forward::operator=(Forward&& other) noexcept {
    if (this != &other) {
        cleanup();
        enable_backward   = other.enable_backward;
        num_prims         = other.num_prims;
        m_context         = std::exchange(other.m_context, nullptr);
        m_device          = other.m_device;
        m_model           = std::exchange(other.m_model, nullptr);
        m_module          = std::exchange(other.m_module, nullptr);
        m_pipeline        = std::exchange(other.m_pipeline, nullptr);
        m_raygenGroup     = std::exchange(other.m_raygenGroup, nullptr);
        m_missGroup       = std::exchange(other.m_missGroup, nullptr);
        m_hitGroup        = std::exchange(other.m_hitGroup, nullptr);
        m_sbt             = std::exchange(other.m_sbt, OptixShaderBindingTable{});
        m_dParams         = std::exchange(other.m_dParams, 0);
        m_params          = other.m_params;
        m_pipelineOptions = other.m_pipelineOptions;
    }
    return *this;
}

// ============================================================================
// Cleanup
// ============================================================================

void Forward::cleanup() {
    if (m_dParams != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_dParams)));
        m_dParams = 0;
    }

    if (m_sbt.raygenRecord != 0)       CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    if (m_sbt.missRecordBase != 0)     CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    if (m_sbt.hitgroupRecordBase != 0) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
    m_sbt = {};

    if (m_pipeline != nullptr)    { OPTIX_CHECK(optixPipelineDestroy(m_pipeline));       m_pipeline = nullptr; }
    if (m_raygenGroup != nullptr) { OPTIX_CHECK(optixProgramGroupDestroy(m_raygenGroup)); m_raygenGroup = nullptr; }
    if (m_missGroup != nullptr)   { OPTIX_CHECK(optixProgramGroupDestroy(m_missGroup));   m_missGroup = nullptr; }
    if (m_hitGroup != nullptr)    { OPTIX_CHECK(optixProgramGroupDestroy(m_hitGroup));    m_hitGroup = nullptr; }
    if (m_module != nullptr)      { OPTIX_CHECK(optixModuleDestroy(m_module));           m_module = nullptr; }
}

// ============================================================================
// Destructor
// ============================================================================

Forward::~Forward() noexcept(false) {
    cleanup();
}
