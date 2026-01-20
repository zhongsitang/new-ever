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
#include "exception.h"
#include "initialize_density.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <array>
#include <utility>

// =============================================================================
// Embedded PTX Shaders (generated at build time)
// =============================================================================

#include "shaders_ptx.ptx.h"
#include "fast_shaders_ptx.ptx.h"

namespace optix_pipeline {

// =============================================================================
// Helper Utilities
// =============================================================================

namespace {

// RAII wrapper for CUDA device context switching
struct ScopedDevice {
    explicit ScopedDevice(int8_t device) { CUDA_CHECK(cudaSetDevice(device)); }
};

// Program entry point names (Slang generates these prefixes)
constexpr std::string_view RAYGEN_ENTRY    = "__raygen__rg_float";
constexpr std::string_view MISS_ENTRY      = "__miss__ms";
constexpr std::string_view ANYHIT_ENTRY    = "__anyhit__ah";
constexpr std::string_view INTERSECT_ENTRY = "__intersection__ellipsoid";

} // anonymous namespace

// =============================================================================
// Forward Implementation
// =============================================================================

Forward::Forward(OptixDeviceContext context, int8_t device,
                 const Primitives& model, bool enable_backward)
    : context_(context)
    , device_(device)
    , enable_backward_(enable_backward)
    , model_(&model)
{
    ScopedDevice scoped(device_);

    // Configure pipeline
    PipelineConfig config{
        .num_payload_values = 32,
        .num_attribute_values = 1,
        .max_trace_depth = 1,
        .opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
        .debug_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        .launch_params_name = "SLANG_globalParams"
    };

    // Initialize pipeline components
    init_module(config);
    init_program_groups();
    init_pipeline(config);
    init_sbt();
    init_params(model);
}

Forward::~Forward() noexcept(false) {
    cleanup();
}

Forward::Forward(Forward&& other) noexcept
    : context_(std::exchange(other.context_, nullptr))
    , module_(std::exchange(other.module_, nullptr))
    , pipeline_(std::exchange(other.pipeline_, nullptr))
    , sbt_(std::exchange(other.sbt_, {}))
    , raygen_pg_(std::exchange(other.raygen_pg_, nullptr))
    , miss_pg_(std::exchange(other.miss_pg_, nullptr))
    , hitgroup_pg_(std::exchange(other.hitgroup_pg_, nullptr))
    , stream_(std::exchange(other.stream_, nullptr))
    , d_params_(std::exchange(other.d_params_, 0))
    , device_(other.device_)
    , enable_backward_(other.enable_backward_)
    , num_prims_(other.num_prims_)
    , model_(other.model_)
    , params_(other.params_)
    , pipeline_compile_options_(other.pipeline_compile_options_)
{
}

Forward& Forward::operator=(Forward&& other) noexcept {
    if (this != &other) {
        cleanup();

        context_ = std::exchange(other.context_, nullptr);
        module_ = std::exchange(other.module_, nullptr);
        pipeline_ = std::exchange(other.pipeline_, nullptr);
        sbt_ = std::exchange(other.sbt_, {});
        raygen_pg_ = std::exchange(other.raygen_pg_, nullptr);
        miss_pg_ = std::exchange(other.miss_pg_, nullptr);
        hitgroup_pg_ = std::exchange(other.hitgroup_pg_, nullptr);
        stream_ = std::exchange(other.stream_, nullptr);
        d_params_ = std::exchange(other.d_params_, 0);
        device_ = other.device_;
        enable_backward_ = other.enable_backward_;
        num_prims_ = other.num_prims_;
        model_ = other.model_;
        params_ = other.params_;
        pipeline_compile_options_ = other.pipeline_compile_options_;
    }
    return *this;
}

void Forward::cleanup() noexcept {
    // Free CUDA resources
    auto safe_cuda_free = [](CUdeviceptr& ptr) {
        if (ptr != 0) {
            cudaFree(reinterpret_cast<void*>(std::exchange(ptr, 0)));
        }
    };

    safe_cuda_free(d_params_);
    safe_cuda_free(sbt_.raygenRecord);
    safe_cuda_free(sbt_.missRecordBase);
    safe_cuda_free(sbt_.hitgroupRecordBase);
    safe_cuda_free(sbt_.callablesRecordBase);
    safe_cuda_free(sbt_.exceptionRecord);
    sbt_ = {};

    if (stream_) {
        cudaStreamDestroy(std::exchange(stream_, nullptr));
    }

    // Destroy OptiX objects (order matters: pipeline -> program groups -> module)
    if (pipeline_) {
        optixPipelineDestroy(std::exchange(pipeline_, nullptr));
    }
    if (raygen_pg_) {
        optixProgramGroupDestroy(std::exchange(raygen_pg_, nullptr));
    }
    if (miss_pg_) {
        optixProgramGroupDestroy(std::exchange(miss_pg_, nullptr));
    }
    if (hitgroup_pg_) {
        optixProgramGroupDestroy(std::exchange(hitgroup_pg_, nullptr));
    }
    if (module_) {
        optixModuleDestroy(std::exchange(module_, nullptr));
    }
}

// =============================================================================
// Module Creation (PTX -> OptiX Module)
// =============================================================================

void Forward::init_module(const PipelineConfig& config) {
    // Configure pipeline compile options
    pipeline_compile_options_ = {
        .usesMotionBlur = false,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        .numPayloadValues = static_cast<int>(config.num_payload_values),
        .numAttributeValues = static_cast<int>(config.num_attribute_values),
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = config.launch_params_name.data(),
        .usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM
    };

    // Module compile options
    OptixModuleCompileOptions module_options = {
        .optLevel = config.opt_level,
        .debugLevel = config.debug_level
    };

    // Select PTX based on backward mode
    const char* ptx_source = enable_backward_
        ? ptx::shaders_ptx
        : ptx::fast_shaders_ptx;

    // Create module from PTX string
    // OptiX 7.7+ uses optixModuleCreate for both PTX and OptiX-IR
    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreate(
        context_,
        &module_options,
        &pipeline_compile_options_,
        ptx_source,
        strlen(ptx_source),
        log, &log_size,
        &module_
    ));
}

// =============================================================================
// Program Group Creation
// =============================================================================

void Forward::init_program_groups() {
    OptixProgramGroupOptions pg_options = {};
    char log[4096];
    size_t log_size;

    // Ray generation program
    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = module_,
                .entryFunctionName = RAYGEN_ENTRY.data()
            }
        };
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &raygen_pg_
        ));
    }

    // Miss program
    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = module_,
                .entryFunctionName = MISS_ENTRY.data()
            }
        };
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &miss_pg_
        ));
    }

    // Hit group (intersection + any-hit)
    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = nullptr,
                .entryFunctionNameCH = nullptr,
                .moduleAH = module_,
                .entryFunctionNameAH = ANYHIT_ENTRY.data(),
                .moduleIS = module_,
                .entryFunctionNameIS = INTERSECT_ENTRY.data()
            }
        };
        log_size = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_pg_
        ));
    }
}

// =============================================================================
// Pipeline Creation and Stack Size Configuration
// =============================================================================

void Forward::init_pipeline(const PipelineConfig& config) {
    std::array<OptixProgramGroup, 3> program_groups = {
        raygen_pg_, miss_pg_, hitgroup_pg_
    };

    OptixPipelineLinkOptions link_options = {
        .maxTraceDepth = config.max_trace_depth
    };

    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_,
        &pipeline_compile_options_,
        &link_options,
        program_groups.data(),
        static_cast<unsigned int>(program_groups.size()),
        log, &log_size,
        &pipeline_
    ));

    // Compute and set stack sizes
    OptixStackSizes stack_sizes = {};
    for (auto pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline_));
    }

    uint32_t dc_stack_from_traversal = 0;
    uint32_t dc_stack_from_state = 0;
    uint32_t continuation_stack = 0;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        config.max_trace_depth,
        0,  // maxCCDepth
        0,  // maxDCDepth
        &dc_stack_from_traversal,
        &dc_stack_from_state,
        &continuation_stack
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        dc_stack_from_traversal,
        dc_stack_from_state,
        continuation_stack,
        1  // maxTraversableDepth
    ));
}

// =============================================================================
// Shader Binding Table Setup
// =============================================================================

void Forward::init_sbt() {
    // Ray generation record
    {
        RayGenSbtRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &record));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&sbt_.raygenRecord),
            sizeof(RayGenSbtRecord)
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(sbt_.raygenRecord),
            &record, sizeof(record), cudaMemcpyHostToDevice
        ));
    }

    // Miss record
    {
        MissSbtRecord record = {};
        record.data.bg_color = make_float3(0.3f, 0.1f, 0.2f);
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &record));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&sbt_.missRecordBase),
            sizeof(MissSbtRecord)
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(sbt_.missRecordBase),
            &record, sizeof(record), cudaMemcpyHostToDevice
        ));
        sbt_.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt_.missRecordCount = 1;
    }

    // Hit group record
    {
        HitGroupSbtRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, &record));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&sbt_.hitgroupRecordBase),
            sizeof(HitGroupSbtRecord)
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(sbt_.hitgroupRecordBase),
            &record, sizeof(record), cudaMemcpyHostToDevice
        ));
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt_.hitgroupRecordCount = 1;
    }
}

// =============================================================================
// Launch Parameters Initialization
// =============================================================================

void Forward::init_params(const Primitives& model) {
    // Initialize primitive data pointers
    // NOTE: half_attribs is cast to float* to match Slang's RWStructuredBuffer<float>
    // The actual data is still __half, but StructuredBuffer only stores pointer+size
    params_.half_attribs = { reinterpret_cast<float*>(model.half_attribs), model.num_prims };
    params_.means = { reinterpret_cast<float3*>(model.means), model.num_prims };
    params_.scales = { reinterpret_cast<float3*>(model.scales), model.num_prims };
    params_.quats = { reinterpret_cast<float4*>(model.quats), model.num_prims };
    params_.densities = { model.densities, model.num_prims };
    params_.features = { model.features, model.num_prims * model.feature_size };

    num_prims_ = model.num_prims;

    // Initialize padding to 0
    params_._pad0 = 0;

    // Allocate device-side params
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_), sizeof(Params)));
}

// =============================================================================
// Feature Reset
// =============================================================================

void Forward::reset_features(const Primitives& model) {
    params_.features = { model.features, model.num_prims * model.feature_size };
}

// =============================================================================
// Ray Tracing Launch
// =============================================================================

void Forward::trace_rays(const LaunchConfig& config) {
    ScopedDevice scoped(device_);

    // Update params with launch configuration
    params_.image = { reinterpret_cast<float4*>(config.image_out), config.num_rays };
    params_.last_state = { config.last_state, config.num_rays };
    params_.last_dirac = { config.last_dirac, config.num_rays };
    params_.tri_collection = { config.tri_collection, config.num_rays * config.max_iters };
    params_.iters = { config.iters, config.num_rays };
    params_.last_face = { config.last_face, config.num_rays };
    params_.touch_count = { config.touch_count, config.num_rays };
    params_.sh_degree = config.sh_degree;
    params_.max_prim_size = config.max_prim_size;
    params_.max_iters = config.max_iters;
    params_.ray_origins = { config.ray_origins, config.num_rays };
    params_.ray_directions = { config.ray_directions, config.num_rays };
    params_.tmin = config.tmin;
    params_.tmax = config.tmax;
    params_.handle = config.handle;

    if (config.camera) {
        params_.camera = *config.camera;
    }

    // Initialize density (clear and set initial values)
    CUDA_CHECK(cudaMemset(config.initial_drgb, 0, config.num_rays * sizeof(float4)));
    params_.initial_drgb = { config.initial_drgb, config.num_rays };

    initialize_density(&params_, model_->aabbs, config.d_touch_count, config.d_touch_inds);

    // Copy params to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_params_),
        &params_, sizeof(Params),
        cudaMemcpyHostToDevice
    ));

    // Launch OptiX kernel
    const uint32_t width = config.camera ? config.camera->width : static_cast<uint32_t>(config.num_rays);
    const uint32_t height = config.camera ? config.camera->height : 1;

    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_params_,
        sizeof(Params),
        &sbt_,
        width,
        height,
        1  // depth
    ));

    // Synchronize
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

} // namespace optix_pipeline
