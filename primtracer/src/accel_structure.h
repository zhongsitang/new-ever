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
#include <cstdint>

#include "optix_error.h"
#include "volume_types.h"

/// Buffer references for GAS building (managed externally by DeviceResources)
struct GASBuffers {
    CUdeviceptr& output_buffer;
    size_t& output_size;
    CUdeviceptr& temp_buffer;
    size_t& temp_size;
    CUdeviceptr& compact_buffer;
    size_t& compact_size;
};

/// Geometry Acceleration Structure for OptiX ray tracing
class GAS {
public:
    OptixTraversableHandle gas_handle = 0;

    GAS(OptixDeviceContext context, int device, const Primitives& model, GASBuffers buffers)
        : context_(context), device_(device)
    {
        build(model, buffers);
    }

    ~GAS() = default;

    // Non-copyable, non-movable (handle lifetime tied to buffers)
    GAS(const GAS&) = delete;
    GAS& operator=(const GAS&) = delete;

private:
    void build(const Primitives& model, GASBuffers& buf) {
        CUDA_CHECK(cudaSetDevice(device_));

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
        CUdeviceptr d_aabbs = reinterpret_cast<CUdeviceptr>(model.aabbs);

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers = &d_aabbs;
        input.customPrimitiveArray.numPrimitives = model.num_prims;
        input.customPrimitiveArray.flags = &flags;
        input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accel_options, &input, 1, &sizes));

        // Ensure buffers are large enough
        if (buf.output_size < sizes.outputSizeInBytes) {
            if (buf.output_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf.output_buffer)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf.output_buffer), sizes.outputSizeInBytes));
            buf.output_size = sizes.outputSizeInBytes;
        }
        if (buf.temp_size < sizes.tempSizeInBytes) {
            if (buf.temp_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf.temp_buffer)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf.temp_buffer), sizes.tempSizeInBytes));
            buf.temp_size = sizes.tempSizeInBytes;
        }

        // Query compacted size
        size_t* d_compacted_size;
        CUDA_CHECK(cudaMalloc(&d_compacted_size, sizeof(size_t)));

        OptixAccelEmitDesc emit = {};
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = reinterpret_cast<CUdeviceptr>(d_compacted_size);

        OPTIX_CHECK(optixAccelBuild(
            context_, 0, &accel_options, &input, 1,
            buf.temp_buffer, sizes.tempSizeInBytes,
            buf.output_buffer, sizes.outputSizeInBytes,
            &gas_handle, &emit, 1
        ));

        // Compact if beneficial
        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, d_compacted_size, sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_compacted_size));

        if (compacted_size < sizes.outputSizeInBytes) {
            if (buf.compact_size < compacted_size) {
                if (buf.compact_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buf.compact_buffer)));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf.compact_buffer), compacted_size));
                buf.compact_size = compacted_size;
            }
            OPTIX_CHECK(optixAccelCompact(context_, 0, gas_handle, buf.compact_buffer, buf.compact_size, &gas_handle));
        }
    }

    OptixDeviceContext context_;
    int device_;
};
