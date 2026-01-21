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

// =============================================================================
// AccelStructure - OptiX Geometry Acceleration Structure (GAS) wrapper
// =============================================================================
//
// This header-only class manages the OptiX acceleration structure for ray
// tracing against custom AABB primitives (ellipsoids).

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdint>
#include <utility>

#include "cuda_buffer.h"
#include "structs.h"

// =============================================================================
// Global buffer cache for GAS memory (avoids repeated allocations)
// =============================================================================
namespace accel_detail {

inline CUdeviceptr& output_buffer() {
    static CUdeviceptr buf = 0;
    return buf;
}

inline size_t& output_buffer_size() {
    static size_t size = 0;
    return size;
}

inline CUdeviceptr& temp_buffer() {
    static CUdeviceptr buf = 0;
    return buf;
}

inline size_t& temp_buffer_size() {
    static size_t size = 0;
    return size;
}

inline CUdeviceptr& compact_buffer() {
    static CUdeviceptr buf = 0;
    return buf;
}

inline size_t& compact_buffer_size() {
    static size_t size = 0;
    return size;
}

}  // namespace accel_detail

// =============================================================================
// AccelStructure class
// =============================================================================

class AccelStructure {
public:
    OptixTraversableHandle handle() const { return handle_; }
    bool is_valid() const { return handle_ != 0; }

    AccelStructure() noexcept = default;

    AccelStructure(OptixDeviceContext context, uint8_t device)
        : context_(context), device_(device) {}

    AccelStructure(OptixDeviceContext context, uint8_t device, const Primitives& primitives)
        : AccelStructure(context, device)
    {
        build(primitives);
    }

    ~AccelStructure() noexcept(false) {
        release();
    }

    // Non-copyable
    AccelStructure(const AccelStructure&) = delete;
    AccelStructure& operator=(const AccelStructure&) = delete;

    // Movable
    AccelStructure(AccelStructure&& other) noexcept
        : device_(std::exchange(other.device_, -1))
        , context_(std::exchange(other.context_, nullptr))
        , handle_(std::exchange(other.handle_, 0))
    {}

    AccelStructure& operator=(AccelStructure&& other) noexcept {
        if (this != &other) {
            release();
            device_ = std::exchange(other.device_, -1);
            context_ = std::exchange(other.context_, nullptr);
            handle_ = std::exchange(other.handle_, 0);
        }
        return *this;
    }

    void build(const Primitives& primitives) {
        release();
        CUDA_CHECK(cudaSetDevice(device_));

        // Configure build options
        OptixAccelBuildOptions build_options = {};
        build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                                   OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Configure AABB input
        uint32_t input_flags = OPTIX_GEOMETRY_FLAG_NONE;
        CUdeviceptr d_aabbs = reinterpret_cast<CUdeviceptr>(primitives.aabbs);

        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabbs;
        build_input.customPrimitiveArray.numPrimitives = primitives.num_prims;
        build_input.customPrimitiveArray.flags = &input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        // Compute memory requirements
        OptixAccelBufferSizes buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context_, &build_options, &build_input, 1, &buffer_sizes
        ));

        // Allocate/resize output buffer
        ensure_buffer_size(
            accel_detail::output_buffer(),
            accel_detail::output_buffer_size(),
            buffer_sizes.outputSizeInBytes
        );

        // Allocate/resize temp buffer
        ensure_buffer_size(
            accel_detail::temp_buffer(),
            accel_detail::temp_buffer_size(),
            buffer_sizes.tempSizeInBytes
        );

        // Setup compaction query
        size_t* d_compacted_size;
        CUDA_CHECK(cudaMalloc(&d_compacted_size, sizeof(size_t)));

        OptixAccelEmitDesc emit_desc = {};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = reinterpret_cast<CUdeviceptr>(d_compacted_size);

        // Build acceleration structure
        OPTIX_CHECK(optixAccelBuild(
            context_,
            0,  // CUDA stream
            &build_options,
            &build_input,
            1,  // num build inputs
            accel_detail::temp_buffer(),
            buffer_sizes.tempSizeInBytes,
            accel_detail::output_buffer(),
            buffer_sizes.outputSizeInBytes,
            &handle_,
            &emit_desc,
            1  // num emitted properties
        ));

        // Compact if beneficial
        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(
            &compacted_size, d_compacted_size,
            sizeof(size_t), cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaFree(d_compacted_size));

        if (compacted_size < buffer_sizes.outputSizeInBytes) {
            ensure_buffer_size(
                accel_detail::compact_buffer(),
                accel_detail::compact_buffer_size(),
                compacted_size
            );

            OPTIX_CHECK(optixAccelCompact(
                context_, 0, handle_,
                accel_detail::compact_buffer(),
                accel_detail::compact_buffer_size(),
                &handle_
            ));
        }
    }

private:
    void release() {
        handle_ = 0;
    }

    static void ensure_buffer_size(CUdeviceptr& buffer, size_t& current_size, size_t required_size) {
        if (current_size < required_size) {
            if (buffer != 0) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
            }
            current_size = required_size;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), current_size));
        }
    }

    int8_t device_ = -1;
    OptixDeviceContext context_ = nullptr;
    OptixTraversableHandle handle_ = 0;
};

// Legacy alias for backward compatibility
using GAS = AccelStructure;
