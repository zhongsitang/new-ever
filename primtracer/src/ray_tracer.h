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

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "types.h"

// =============================================================================
// Error Handling
// =============================================================================

class Exception : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline void optix_log_cb(unsigned int, const char*, const char*, void*) {}

#define OPTIX_CHECK(call)                                                       \
    if (OptixResult res_ = (call); res_ != OPTIX_SUCCESS)                       \
        throw Exception(std::string(optixGetErrorName(res_)) + " at " +         \
                        __FILE__ + ":" + std::to_string(__LINE__))

#define OPTIX_CHECK_LOG(call)                                                   \
    do {                                                                        \
        char log_[2048]; size_t log_sz_ = sizeof(log_);                         \
        if (OptixResult res_ = (call); res_ != OPTIX_SUCCESS)                   \
            throw Exception(std::string(optixGetErrorName(res_)) + " at " +     \
                            __FILE__ + ":" + std::to_string(__LINE__) +         \
                            "\n" + log_);                                       \
    } while (0)

#define CUDA_CHECK(call)                                                        \
    if (cudaError_t err_ = (call); err_ != cudaSuccess)                         \
        throw Exception(std::string("CUDA: ") + cudaGetErrorString(err_) +      \
                        " at " + __FILE__ + ":" + std::to_string(__LINE__))

#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                        \
        cudaDeviceSynchronize();                                                \
        if (cudaError_t err_ = cudaGetLastError(); err_ != cudaSuccess)         \
            throw Exception(std::string("CUDA: ") + cudaGetErrorString(err_) +  \
                            " at " + __FILE__ + ":" + std::to_string(__LINE__));\
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
        options.logCallbackFunction = &optix_log_cb;
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
    uint64_t num_prims() const { return num_prims_; }

private:
    void ensure_aabb_capacity(uint64_t num_prims) {
        if (num_prims > aabb_capacity_) {
            if (aabb_buffer_) {
                CUDA_CHECK(cudaFree(aabb_buffer_));
            }
            CUDA_CHECK(cudaMalloc(&aabb_buffer_, num_prims * sizeof(OptixAabb)));
            aabb_capacity_ = num_prims;
        }
    }

    void ensure_gas_capacity(uint64_t output_size, uint64_t temp_size) {
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

    void build_gas(uint64_t num_prims) {
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
    uint64_t num_prims_ = 0;

    OptixAabb* aabb_buffer_ = nullptr;
    uint64_t aabb_capacity_ = 0;

    CUdeviceptr gas_output_ = 0;
    uint64_t gas_output_capacity_ = 0;
    CUdeviceptr gas_temp_ = 0;
    uint64_t gas_temp_capacity_ = 0;
    CUdeviceptr gas_compact_ = 0;
    uint64_t gas_compact_capacity_ = 0;
};

// =============================================================================
// SBT Record Types
// =============================================================================

/// Minimal SBT record containing only the required header (no per-shader data).
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

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
    /// Note: ray_origins/ray_directions use float4 for ABI stability (.w unused).
    void trace_rays(
        uint64_t num_rays,
        float4* ray_origins,
        float4* ray_directions,
        float4* color_out,
        float* depth_out,
        uint32_t sh_degree,
        float tmin,
        float* tmax,
        uint32_t max_iters,
        SavedState* saved
    );

    bool has_primitives() const { return prims_.num_prims > 0; }
    uint64_t num_prims() const { return prims_.num_prims; }
    int device_index() const { return ctx_.device(); }

private:
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    DeviceContext& ctx_;
    std::unique_ptr<AccelStructure> accel_;
    Primitives prims_ = {};

    // OptiX pipeline objects (created once)
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    OptixShaderBindingTable sbt_ = {};
    CUdeviceptr d_param_ = 0;
    uint32_t* d_debug_flag_ = nullptr;  // GPU self-check flag

    Params params_ = {};
    OptixPipelineCompileOptions pipeline_options_ = {};
};
