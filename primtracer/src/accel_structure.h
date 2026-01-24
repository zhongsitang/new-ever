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

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "device_context.h"
#include "primitive_kernels.h"
#include "optix_error.h"

// =============================================================================
// AccelStructure - AABB and GAS management with buffer reuse (header-only)
// =============================================================================

/// Manages acceleration structure: AABB buffer and GAS (Geometry Acceleration Structure).
/// Supports efficient rebuilding with buffer capacity tracking to avoid repeated allocations.
class AccelStructure {
public:
    /// Create an empty acceleration structure (call rebuild() to populate)
    explicit AccelStructure(DeviceContext& ctx) : ctx_(ctx) {}

    /// Create and build acceleration structure for primitives
    AccelStructure(DeviceContext& ctx, const Primitives& prims) : ctx_(ctx) {
        rebuild(prims);
    }

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

        // Ensure AABB buffer capacity and compute AABBs
        ensure_aabb_capacity(num_prims_);
        compute_primitive_aabbs(prims, aabb_buffer_);

        // Build GAS
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

        // Ensure buffer capacity (reuses existing if sufficient)
        ensure_gas_capacity(sizes.outputSizeInBytes, sizes.tempSizeInBytes);

        // Query compacted size
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

        // Compact if beneficial
        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, d_compacted_size, sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_compacted_size));

        if (compacted_size < sizes.outputSizeInBytes) {
            // Ensure compact buffer capacity
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

    // AABB buffer with capacity tracking
    OptixAabb* aabb_buffer_ = nullptr;
    size_t aabb_capacity_ = 0;

    // GAS buffers with capacity tracking
    CUdeviceptr gas_output_ = 0;
    size_t gas_output_capacity_ = 0;
    CUdeviceptr gas_temp_ = 0;
    size_t gas_temp_capacity_ = 0;
    CUdeviceptr gas_compact_ = 0;
    size_t gas_compact_capacity_ = 0;
};
