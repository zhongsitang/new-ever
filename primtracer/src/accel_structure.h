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
#include <math.h>
#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_buffer.h"
#include "optix_error.h"
#include "volume_types.h"

namespace gas_internal {
inline CUdeviceptr D_GAS_OUTPUT_BUFFER = 0;
inline size_t OUTPUT_BUFFER_SIZE = 0;
inline CUdeviceptr D_TEMP_BUFFER_GAS = 0;
inline size_t TEMP_BUFFER_SIZE = 0;
inline CUdeviceptr D_COMPACT_GAS_BUFFER = 0;
inline size_t COMPACT_GAS_BUFFER_SIZE = 0;
}

class GAS {
   public:
    OptixTraversableHandle gas_handle = 0;
    OptixTraversableHandle compactedAccelHandle = 0;

    GAS() noexcept
        : device(-1),
          context(nullptr),
          gas_handle(0) {}

    GAS(const OptixDeviceContext &context, const uint8_t device, const bool enable_backwards, const bool fast_build)
        : device(device), context(context), enable_backwards(enable_backwards), fast_build(fast_build) {}

    GAS(const OptixDeviceContext &context,
        const uint8_t device,
        const Primitives &model,
        const bool enable_backwards=false,
        const bool fast_build=false)
        : GAS(context, device, enable_backwards, fast_build) {
        build(model);
    }

    ~GAS() noexcept(false) {
        if (this->device != -1) {
            release();
        }
        device = -1;
    }

    GAS(const GAS &) = delete;
    GAS &operator=(const GAS &) = delete;

    GAS(GAS &&other) noexcept
        : device(std::exchange(other.device, -1)),
          context(std::exchange(other.context, nullptr)),
          gas_handle(std::exchange(other.gas_handle, 0)) {}

    GAS &operator=(GAS &&other) {
        using std::swap;
        if (this != &other) {
            GAS tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }

    friend void swap(GAS &first, GAS &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.gas_handle, second.gas_handle);
    }

    bool defined() const {
        return gas_handle != 0;
    }

   private:
    void release() {
        gas_handle = 0;
    }

    void build(const Primitives &model) {
        using namespace gas_internal;

        release();
        CUDA_CHECK(cudaSetDevice(device));

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        uint32_t aabb_input_flags[1];
        aabb_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

        CUdeviceptr d_aabbs = (CUdeviceptr)model.aabbs;
        OptixBuildInput aabb_input = {};
        aabb_input.type                        = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabbs;
        aabb_input.customPrimitiveArray.numPrimitives = model.num_prims;
        aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
        aabb_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &aabb_input,
            1,
            &gas_buffer_sizes
        ));

        if (OUTPUT_BUFFER_SIZE <= gas_buffer_sizes.outputSizeInBytes) {
            if (D_GAS_OUTPUT_BUFFER != 0) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_GAS_OUTPUT_BUFFER)));
            }
            OUTPUT_BUFFER_SIZE = size_t(gas_buffer_sizes.outputSizeInBytes);
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&D_GAS_OUTPUT_BUFFER),
                OUTPUT_BUFFER_SIZE
            ));
        }

        if (TEMP_BUFFER_SIZE <= gas_buffer_sizes.tempSizeInBytes) {
            if (D_TEMP_BUFFER_GAS != 0) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_TEMP_BUFFER_GAS)));
            }
            TEMP_BUFFER_SIZE = size_t(gas_buffer_sizes.tempSizeInBytes);
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&D_TEMP_BUFFER_GAS),
                TEMP_BUFFER_SIZE
            ));
        }

        size_t *d_compactedSize;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_compactedSize),
            sizeof(size_t)
        ));

        OptixAccelEmitDesc property = {};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = (CUdeviceptr)d_compactedSize;

        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &accel_options,
            &aabb_input,
            1,
            D_TEMP_BUFFER_GAS,
            gas_buffer_sizes.tempSizeInBytes,
            D_GAS_OUTPUT_BUFFER,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            &property,
            1
        ));

        size_t compactedSize;
        cudaMemcpy(&compactedSize, d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost);

        if (compactedSize < gas_buffer_sizes.outputSizeInBytes) {
            if (COMPACT_GAS_BUFFER_SIZE <= compactedSize) {
                if (D_COMPACT_GAS_BUFFER != 0) {
                    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(D_COMPACT_GAS_BUFFER)));
                }
                COMPACT_GAS_BUFFER_SIZE = compactedSize;
                CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void**>(&D_COMPACT_GAS_BUFFER),
                    COMPACT_GAS_BUFFER_SIZE
                ));
                OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, D_COMPACT_GAS_BUFFER, COMPACT_GAS_BUFFER_SIZE, &gas_handle));
            }
        }

        CUDA_CHECK(cudaFree(d_compactedSize));
    }

    bool enable_backwards = false;
    bool fast_build = false;
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
};
