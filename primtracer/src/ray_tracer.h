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
#include <optix_stubs.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "types.h"

// =============================================================================
// Error Handling
// =============================================================================

namespace detail {

inline std::string optix_error_name(OptixResult res) {
#if !defined(__CUDACC__)
    const char* name = optixGetErrorName(res);
    return name ? name : "OPTIX_ERROR_UNKNOWN";
#else
    (void)res;
    return "OPTIX_ERROR";
#endif
}

inline const char* cuda_error_string(cudaError_t err) {
    const char* s = cudaGetErrorString(err);
    return s ? s : "cudaErrorUnknown";
}

inline std::string make_location(const char* file, int line) {
    std::ostringstream out;
    out << file << ":" << line;
    return out.str();
}

inline void context_log_cb(unsigned int /*level*/, const char* /*tag*/,
                           const char* /*message*/, void* /*cbdata*/) {
    // Silently ignore OptiX log messages
}

} // namespace detail

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
    explicit Exception(const char* msg) : std::runtime_error(msg ? msg : "Exception") {}

    Exception(OptixResult res, const std::string& msg)
        : std::runtime_error(detail::optix_error_name(res) + ": " + msg) {}
};

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult res_ = (call);                                              \
        if (res_ != OPTIX_SUCCESS) {                                            \
            std::ostringstream ss_;                                             \
            ss_ << "OptiX call '" << #call << "' failed at "                    \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(res_, ss_.str());                                   \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                   \
    do {                                                                        \
        char log_[8192];                                                        \
        size_t log_size_ = sizeof(log_);                                        \
        OptixResult res_ = (call);                                              \
        if (res_ != OPTIX_SUCCESS) {                                            \
            std::ostringstream ss_;                                             \
            ss_ << "OptiX call '" << #call << "' failed at "                    \
                << detail::make_location(__FILE__, __LINE__)                    \
                << "\nLog: " << log_;                                           \
            throw Exception(res_, ss_.str());                                   \
        }                                                                       \
    } while (0)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA call (" << #call << ") failed: '"                      \
                << detail::cuda_error_string(err_) << "' at "                   \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
    } while (0)

#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                        \
        cudaError_t sync_err_ = cudaDeviceSynchronize();                        \
        cudaError_t last_err_ = cudaGetLastError();                             \
        if (sync_err_ != cudaSuccess) {                                         \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA sync failed: '"                                        \
                << detail::cuda_error_string(sync_err_) << "' at "              \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
        if (last_err_ != cudaSuccess) {                                         \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA last error: '"                                         \
                << detail::cuda_error_string(last_err_) << "' at "              \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
    } while (0)

// =============================================================================
// CUDA Kernel Declarations
// =============================================================================

/// Compute axis-aligned bounding boxes for ellipsoid primitives.
void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs);

/// Initialize contributions for rays starting inside primitives.
void init_ray_start_samples(Params* params, OptixAabb* aabbs,
                            int* d_hit_count = nullptr,
                            int* d_hit_inds = nullptr);

// =============================================================================
// DeviceContext - Per-device OptiX context (globally cached)
// =============================================================================

/// Manages per-device OptiX context. Cached globally and shared across RayTracers.
class DeviceContext {
public:
    /// Get or create context for a specific device (cached globally)
    static DeviceContext& get(int device_index) {
        auto it = contexts().find(device_index);
        if (it != contexts().end()) {
            return *it->second;
        }

        auto ctx = std::unique_ptr<DeviceContext>(new DeviceContext(device_index));
        auto& ref = *ctx;
        contexts()[device_index] = std::move(ctx);
        return ref;
    }

    OptixDeviceContext context() const { return context_; }
    int device() const { return device_; }

    // Non-copyable
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    ~DeviceContext() {
        if (context_) optixDeviceContextDestroy(context_);
    }

private:
    explicit DeviceContext(int device_index) : device_(device_index) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaFree(0));  // Initialize CUDA context
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &detail::context_log_cb;
        options.logCallbackLevel = 4;

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
    }

    static std::unordered_map<int, std::unique_ptr<DeviceContext>>& contexts() {
        static std::unordered_map<int, std::unique_ptr<DeviceContext>> g_contexts;
        return g_contexts;
    }

    int device_ = -1;
    OptixDeviceContext context_ = nullptr;
};

// =============================================================================
// AccelStructure - AABB and GAS management with buffer reuse
// =============================================================================

/// Manages acceleration structure: AABB buffer and GAS (Geometry Acceleration Structure).
/// Supports efficient rebuilding with buffer capacity tracking to avoid repeated allocations.
class AccelStructure {
public:
    explicit AccelStructure(DeviceContext& ctx) : ctx_(ctx) {}

    ~AccelStructure() {
        if (gas_compact_) cudaFree(reinterpret_cast<void*>(gas_compact_));
        if (gas_temp_) cudaFree(reinterpret_cast<void*>(gas_temp_));
        if (gas_output_) cudaFree(reinterpret_cast<void*>(gas_output_));
        if (aabb_buffer_) cudaFree(aabb_buffer_);
    }

    // Non-copyable
    AccelStructure(const AccelStructure&) = delete;
    AccelStructure& operator=(const AccelStructure&) = delete;

    /// Rebuild acceleration structure for new primitives.
    /// Reuses existing buffers when capacity is sufficient.
    void rebuild(const Primitives& prims) {
        CUDA_CHECK(cudaSetDevice(ctx_.device()));
        num_prims_ = prims.num_prims;

        ensure_aabb_capacity(num_prims_);
        compute_primitive_aabbs(prims, aabb_buffer_);
        build_gas(num_prims_);
    }

    OptixTraversableHandle handle() const { return gas_handle_; }
    OptixAabb* aabbs() const { return aabb_buffer_; }
    size_t num_prims() const { return num_prims_; }

private:
    void ensure_aabb_capacity(size_t num_prims) {
        if (num_prims > aabb_capacity_) {
            if (aabb_buffer_) {
                CUDA_CHECK(cudaFree(aabb_buffer_));
            }
            CUDA_CHECK(cudaMalloc(&aabb_buffer_, num_prims * sizeof(OptixAabb)));
            aabb_capacity_ = num_prims;
        }
    }

    void ensure_gas_capacity(size_t output_size, size_t temp_size) {
        if (output_size > gas_output_capacity_) {
            if (gas_output_) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gas_output_)));
            }
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas_output_), output_size));
            gas_output_capacity_ = output_size;
        }

        if (temp_size > gas_temp_capacity_) {
            if (gas_temp_) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gas_temp_)));
            }
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas_temp_), temp_size));
            gas_temp_capacity_ = temp_size;
        }
    }

    void build_gas(size_t num_prims) {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
        CUdeviceptr d_aabbs = reinterpret_cast<CUdeviceptr>(aabb_buffer_);

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers = &d_aabbs;
        input.customPrimitiveArray.numPrimitives = num_prims;
        input.customPrimitiveArray.flags = &flags;
        input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_.context(), &accel_options, &input, 1, &sizes));

        ensure_gas_capacity(sizes.outputSizeInBytes, sizes.tempSizeInBytes);

        size_t* d_compacted_size;
        CUDA_CHECK(cudaMalloc(&d_compacted_size, sizeof(size_t)));

        OptixAccelEmitDesc emit = {};
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = reinterpret_cast<CUdeviceptr>(d_compacted_size);

        OPTIX_CHECK(optixAccelBuild(
            ctx_.context(), 0, &accel_options, &input, 1,
            gas_temp_, sizes.tempSizeInBytes,
            gas_output_, sizes.outputSizeInBytes,
            &gas_handle_, &emit, 1
        ));

        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, d_compacted_size, sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_compacted_size));

        if (compacted_size < sizes.outputSizeInBytes) {
            if (compacted_size > gas_compact_capacity_) {
                if (gas_compact_) {
                    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gas_compact_)));
                }
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas_compact_), compacted_size));
                gas_compact_capacity_ = compacted_size;
            }
            OPTIX_CHECK(optixAccelCompact(ctx_.context(), 0, gas_handle_, gas_compact_, compacted_size, &gas_handle_));
        }
    }

    DeviceContext& ctx_;
    OptixTraversableHandle gas_handle_ = 0;
    size_t num_prims_ = 0;

    OptixAabb* aabb_buffer_ = nullptr;
    size_t aabb_capacity_ = 0;

    CUdeviceptr gas_output_ = 0;
    size_t gas_output_capacity_ = 0;
    CUdeviceptr gas_temp_ = 0;
    size_t gas_temp_capacity_ = 0;
    CUdeviceptr gas_compact_ = 0;
    size_t gas_compact_capacity_ = 0;
};

// =============================================================================
// SBT Record Types
// =============================================================================

struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// =============================================================================
// RayTracer - Main class with reusable Pipeline and rebuildable AccelStructure
// =============================================================================

/// Complete ray tracing pipeline for ellipsoid volume rendering.
///
/// Design: Pipeline is compiled once at construction. AccelStructure can be
/// rebuilt when primitives change via update_primitives().
///
/// Usage:
///   RayTracer tracer(device_index);
///   tracer.update_primitives(prims);
///   tracer.trace_rays(...);
///   // Later, with different primitives:
///   tracer.update_primitives(new_prims);
///   tracer.trace_rays(...);
class RayTracer {
public:
    /// Construct and compile the OptiX pipeline (no primitives required).
    explicit RayTracer(int device_index);

    ~RayTracer();

    // Non-copyable
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    /// Update primitives and rebuild acceleration structure.
    /// Call this when primitive data changes (count or values).
    void update_primitives(const Primitives& prims);

    /// Trace rays through the scene.
    /// Requires update_primitives() to be called first.
    void trace_rays(
        size_t num_rays,
        float3* ray_origins,
        float3* ray_directions,
        float4* color_out,
        float* depth_out,
        uint sh_degree,
        float tmin,
        float* tmax,
        size_t max_iters,
        SavedState* saved
    );

    bool has_primitives() const { return model_.num_prims > 0; }
    size_t num_prims() const { return model_.num_prims; }
    int device_index() const { return ctx_.device(); }

private:
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    DeviceContext& ctx_;
    std::unique_ptr<AccelStructure> accel_;
    Primitives model_ = {};

    // OptiX pipeline objects (created once)
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

    // Log buffer for OptiX
    static constexpr size_t LOG_SIZE = 8192;
    char log_[LOG_SIZE];
    size_t log_size_ = LOG_SIZE;
};
