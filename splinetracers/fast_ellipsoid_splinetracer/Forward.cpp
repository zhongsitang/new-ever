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
// Helper Macros for Buffer Operations
// ============================================================================

#define SET_BUFFER(buf, ptr, count) \
    do { \
        (buf).data = (ptr); \
        (buf).size = (count); \
    } while(0)

#define CLEAR_BUFFER(buf) \
    do { \
        (buf).data = nullptr; \
        (buf).size = 0; \
    } while(0)

// ============================================================================
// Constructor
// ============================================================================

Forward::Forward(
    const OptixDeviceContext& context,
    int8_t device,
    const Primitives& model,
    bool enableBackward,
    const PipelineConfig& config
)
    : enable_backward(enableBackward)
    , m_config(config)
    , m_context(context)
    , m_device(device)
    , m_model(&model)
{
    CUDA_CHECK(cudaSetDevice(device));

    // Initialize pipeline
    initModule(config);
    initProgramGroups();
    initPipeline(config);
    initShaderBindingTable();
    initLaunchParams(model);

    // Allocate device memory for launch params
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_dParams), sizeof(Params)));

    // Store primitive count
    num_prims = model.num_prims;
}

// ============================================================================
// Module Creation
// ============================================================================

void Forward::initModule(const PipelineConfig& config) {
    char log[16384];
    size_t logSize = sizeof(log);

    // Configure module compile options
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel   = config.optimizationLevel;
    moduleOptions.debugLevel = config.debugLevel;

    // Configure pipeline compile options
    m_pipelineOptions = {};
    m_pipelineOptions.usesMotionBlur        = false;
    m_pipelineOptions.traversableGraphFlags = config.traversableGraphFlags;
    m_pipelineOptions.numPayloadValues      = config.numPayloadValues;
    m_pipelineOptions.numAttributeValues    = config.numAttributeValues;
    m_pipelineOptions.exceptionFlags        = config.exceptionFlags;
    m_pipelineOptions.pipelineLaunchParamsVariableName = config.launchParamsName;
    m_pipelineOptions.usesPrimitiveTypeFlags = config.primitiveTypeFlags;

    // Select shader module based on backward pass requirement
    const unsigned char* shaderData;
    size_t shaderSize;

    if (enable_backward) {
        shaderData = shaders_optixir;
        shaderSize = shaders_optixir_size;
    } else {
        shaderData = fast_shaders_optixir;
        shaderSize = fast_shaders_optixir_size;
    }

    // Create module from OptiX-IR
    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &moduleOptions,
        &m_pipelineOptions,
        reinterpret_cast<const char*>(shaderData),
        shaderSize,
        log,
        &logSize,
        &m_module
    ));
}

// ============================================================================
// Program Group Creation
// ============================================================================

void Forward::initProgramGroups() {
    char log[16384];
    size_t logSize;

    OptixProgramGroupOptions groupOptions = {};

    // Ray Generation Program
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = m_module;
        desc.raygen.entryFunctionName = "__raygen__rg_float";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &desc,
            1,
            &groupOptions,
            log,
            &logSize,
            &m_raygenGroup
        ));
    }

    // Miss Program
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module            = m_module;
        desc.miss.entryFunctionName = "__miss__ms";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &desc,
            1,
            &groupOptions,
            log,
            &logSize,
            &m_missGroup
        ));
    }

    // Hit Group Program (intersection + any-hit + closest-hit)
    {
        logSize = sizeof(log);
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // Closest-hit shader
        desc.hitgroup.moduleCH            = m_module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        // Any-hit shader
        desc.hitgroup.moduleAH            = m_module;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";

        // Intersection shader (for procedural geometry)
        desc.hitgroup.moduleIS            = m_module;
        desc.hitgroup.entryFunctionNameIS = "__intersection__ellipsoid";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &desc,
            1,
            &groupOptions,
            log,
            &logSize,
            &m_hitGroup
        ));
    }
}

// ============================================================================
// Pipeline Creation
// ============================================================================

void Forward::initPipeline(const PipelineConfig& config) {
    char log[16384];
    size_t logSize = sizeof(log);

    // Collect program groups
    OptixProgramGroup programGroups[] = {
        m_raygenGroup,
        m_missGroup,
        m_hitGroup
    };
    const uint32_t numGroups = sizeof(programGroups) / sizeof(programGroups[0]);

    // Pipeline link options
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = config.maxTraceDepth;

    // Create pipeline
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &m_pipelineOptions,
        &linkOptions,
        programGroups,
        numGroups,
        log,
        &logSize,
        &m_pipeline
    ));

    // Calculate and set stack sizes
    OptixStackSizes stackSizes = {};
    for (auto& group : programGroups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, m_pipeline));
    }

    uint32_t directCallableFromTraversal;
    uint32_t directCallableFromState;
    uint32_t continuationStackSize;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        config.maxTraceDepth,
        0,  // maxCCDepth
        0,  // maxDCDepth
        &directCallableFromTraversal,
        &directCallableFromState,
        &continuationStackSize
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        directCallableFromTraversal,
        directCallableFromState,
        continuationStackSize,
        config.maxTraversableDepth
    ));
}

// ============================================================================
// Shader Binding Table Setup
// ============================================================================

void Forward::initShaderBindingTable() {
    // Ray Generation Record
    {
        RayGenSbtRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenGroup, &record));

        CUdeviceptr devicePtr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(record)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(devicePtr),
            &record,
            sizeof(record),
            cudaMemcpyHostToDevice
        ));

        m_sbt.raygenRecord = devicePtr;
    }

    // Miss Record
    {
        MissSbtRecord record = {};
        record.data.backgroundColor = make_float3(0.3f, 0.1f, 0.2f);  // Default background
        OPTIX_CHECK(optixSbtRecordPackHeader(m_missGroup, &record));

        CUdeviceptr devicePtr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(record)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(devicePtr),
            &record,
            sizeof(record),
            cudaMemcpyHostToDevice
        ));

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
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(devicePtr),
            &record,
            sizeof(record),
            cudaMemcpyHostToDevice
        ));

        m_sbt.hitgroupRecordBase          = devicePtr;
        m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        m_sbt.hitgroupRecordCount         = 1;
    }

    // No callable or exception records used
    m_sbt.callablesRecordBase = 0;
    m_sbt.callablesRecordStrideInBytes = 0;
    m_sbt.callablesRecordCount = 0;
    m_sbt.exceptionRecord = 0;
}

// ============================================================================
// Launch Parameters Initialization
// ============================================================================

void Forward::initLaunchParams(const Primitives& model) {
    // Zero-initialize params
    std::memset(&m_params, 0, sizeof(m_params));

    // Set model data buffers
    SET_BUFFER(m_params.half_attribs, model.half_attribs, model.num_prims);
    SET_BUFFER(m_params.means,        reinterpret_cast<float3*>(model.means), model.num_prims);
    SET_BUFFER(m_params.scales,       reinterpret_cast<float3*>(model.scales), model.num_prims);
    SET_BUFFER(m_params.quats,        reinterpret_cast<float4*>(model.quats), model.num_prims);
    SET_BUFFER(m_params.densities,    model.densities, model.num_prims);
    SET_BUFFER(m_params.features,     model.features, model.num_prims * model.feature_size);
}

// ============================================================================
// Trace Rays - Modern Interface
// ============================================================================

void Forward::traceRays(const TraceParams& params) {
    CUDA_CHECK(cudaSetDevice(m_device));

    updateParams(params);
    uploadParams();

    // Determine dispatch dimensions
    uint32_t width, height, depth;
    if (params.camera != nullptr) {
        width  = static_cast<uint32_t>(params.camera->width);
        height = static_cast<uint32_t>(params.camera->height);
        depth  = 1;
    } else {
        width  = static_cast<uint32_t>(params.numRays);
        height = 1;
        depth  = 1;
    }

    // Launch OptiX
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        m_stream,
        m_dParams,
        sizeof(Params),
        &m_sbt,
        width,
        height,
        depth
    ));

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

// ============================================================================
// Trace Rays - Legacy Interface
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
    TraceParams params;
    params.handle        = handle;
    params.numRays       = numRays;
    params.rayOrigins    = rayOrigins;
    params.rayDirections = rayDirections;
    params.imageOut      = imageOut;
    params.shDegree      = shDegree;
    params.tmin          = tmin;
    params.tmax          = tmax;
    params.maxPrimSize   = maxPrimSize;
    params.maxIters      = maxIters;
    params.camera        = camera;
    params.initialDrgb   = initialDrgb;
    params.iters         = iters;
    params.lastFace      = lastFace;
    params.touchCount    = touchCount;
    params.lastDirac     = lastDirac;
    params.lastState     = lastState;
    params.triCollection = triCollection;
    params.dTouchCount   = dTouchCount;
    params.dTouchInds    = dTouchInds;

    traceRays(params);
}

// ============================================================================
// Update Launch Parameters
// ============================================================================

void Forward::updateParams(const TraceParams& params) {
    // Output buffers
    SET_BUFFER(m_params.image,          reinterpret_cast<float4*>(params.imageOut), params.numRays);
    SET_BUFFER(m_params.last_state,     params.lastState, params.numRays);
    SET_BUFFER(m_params.last_dirac,     params.lastDirac, params.numRays);
    SET_BUFFER(m_params.tri_collection, params.triCollection, params.numRays * params.maxIters);
    SET_BUFFER(m_params.iters,          params.iters, params.numRays);
    SET_BUFFER(m_params.last_face,      params.lastFace, params.numRays);
    SET_BUFFER(m_params.touch_count,    params.touchCount, m_model->num_prims);

    // Input buffers
    SET_BUFFER(m_params.ray_origins,    params.rayOrigins, params.numRays);
    SET_BUFFER(m_params.ray_directions, params.rayDirections, params.numRays);

    // Camera
    if (params.camera != nullptr) {
        m_params.camera = *params.camera;
    }

    // Render parameters
    m_params.sh_degree    = params.shDegree;
    m_params.max_iters    = static_cast<uint>(params.maxIters);
    m_params.tmin         = params.tmin;
    m_params.tmax         = params.tmax;
    m_params.max_prim_size = params.maxPrimSize;

    // Initial DRGB buffer
    if (params.initialDrgb != nullptr) {
        CUDA_CHECK(cudaMemset(
            reinterpret_cast<void*>(params.initialDrgb),
            0,
            params.numRays * sizeof(float4)
        ));
        SET_BUFFER(m_params.initial_drgb, params.initialDrgb, params.numRays);

        // Initialize density if helpers provided
        if (params.dTouchCount != nullptr && params.dTouchInds != nullptr) {
            initialize_density(&m_params, m_model->aabbs, params.dTouchCount, params.dTouchInds);
        }
    }

    // Acceleration structure handle
    m_params.handle = params.handle;
}

// ============================================================================
// Upload Launch Parameters
// ============================================================================

void Forward::uploadParams() {
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_dParams),
        &m_params,
        sizeof(Params),
        cudaMemcpyHostToDevice
    ));
}

// ============================================================================
// Reset Features
// ============================================================================

void Forward::resetFeatures(const Primitives& model) {
    SET_BUFFER(m_params.features, model.features, model.num_prims * model.feature_size);
}

// ============================================================================
// Move Operations
// ============================================================================

Forward::Forward(Forward&& other) noexcept
    : enable_backward(other.enable_backward)
    , num_prims(other.num_prims)
    , m_config(other.m_config)
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
    , m_stream(std::exchange(other.m_stream, nullptr))
    , m_params(other.m_params)
    , m_pipelineOptions(other.m_pipelineOptions)
{
}

Forward& Forward::operator=(Forward&& other) noexcept {
    if (this != &other) {
        cleanup();

        enable_backward = other.enable_backward;
        num_prims       = other.num_prims;
        m_config        = other.m_config;
        m_context       = std::exchange(other.m_context, nullptr);
        m_device        = other.m_device;
        m_model         = std::exchange(other.m_model, nullptr);
        m_module        = std::exchange(other.m_module, nullptr);
        m_pipeline      = std::exchange(other.m_pipeline, nullptr);
        m_raygenGroup   = std::exchange(other.m_raygenGroup, nullptr);
        m_missGroup     = std::exchange(other.m_missGroup, nullptr);
        m_hitGroup      = std::exchange(other.m_hitGroup, nullptr);
        m_sbt           = std::exchange(other.m_sbt, OptixShaderBindingTable{});
        m_dParams       = std::exchange(other.m_dParams, 0);
        m_stream        = std::exchange(other.m_stream, nullptr);
        m_params        = other.m_params;
        m_pipelineOptions = other.m_pipelineOptions;
    }
    return *this;
}

// ============================================================================
// Cleanup
// ============================================================================

void Forward::cleanup() {
    // Free device memory
    if (m_dParams != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_dParams)));
        m_dParams = 0;
    }

    // Free SBT records
    if (m_sbt.raygenRecord != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    }
    if (m_sbt.missRecordBase != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    }
    if (m_sbt.hitgroupRecordBase != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
    }
    if (m_sbt.callablesRecordBase != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.callablesRecordBase)));
    }
    if (m_sbt.exceptionRecord != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.exceptionRecord)));
    }
    m_sbt = {};

    // Destroy CUDA stream
    if (m_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(m_stream));
        m_stream = nullptr;
    }

    // Destroy OptiX objects
    if (m_pipeline != nullptr) {
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
        m_pipeline = nullptr;
    }
    if (m_raygenGroup != nullptr) {
        OPTIX_CHECK(optixProgramGroupDestroy(m_raygenGroup));
        m_raygenGroup = nullptr;
    }
    if (m_missGroup != nullptr) {
        OPTIX_CHECK(optixProgramGroupDestroy(m_missGroup));
        m_missGroup = nullptr;
    }
    if (m_hitGroup != nullptr) {
        OPTIX_CHECK(optixProgramGroupDestroy(m_hitGroup));
        m_hitGroup = nullptr;
    }
    if (m_module != nullptr) {
        OPTIX_CHECK(optixModuleDestroy(m_module));
        m_module = nullptr;
    }
}

// ============================================================================
// Destructor
// ============================================================================

Forward::~Forward() noexcept(false) {
    cleanup();
}
