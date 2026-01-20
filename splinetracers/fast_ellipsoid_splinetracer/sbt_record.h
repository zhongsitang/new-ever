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

#include <optix.h>
#include <cuda_runtime.h>

// ============================================================================
// Shader Binding Table (SBT) Record Types
// ============================================================================
//
// SBT records must be aligned to OPTIX_SBT_RECORD_ALIGNMENT (typically 16 bytes)
// and have a header of size OPTIX_SBT_RECORD_HEADER_SIZE (typically 32 bytes).
//
// Modern OptiX 9.1 pattern:
// - Use template for type-safe record creation
// - Separate empty records (no data) from data records
// - Ensure proper alignment
// ============================================================================

// Empty SBT record (header only, no user data)
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// Generic SBT record template with user data
template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// ============================================================================
// Ray Generation Record Data
// ============================================================================

struct RayGenData {
    // No additional data needed for ray generation
    // Camera and other parameters come from launch params
};

using RayGenSbtRecord = SbtRecord<RayGenData>;

// ============================================================================
// Miss Shader Record Data
// ============================================================================

struct MissData {
    float3 backgroundColor;
    float  padding;  // Ensure 16-byte alignment
};

using MissSbtRecord = SbtRecord<MissData>;

// ============================================================================
// Hit Group Record Data
// ============================================================================

// For procedural geometry (ellipsoids), we don't need per-primitive SBT data
// since all geometry data comes from global buffers in launch params.
// Using empty record for hit group.

struct HitGroupData {
    // Reserved for future use (e.g., material index, instance data)
    uint32_t materialIndex;
    uint32_t instanceIndex;
    uint32_t padding[2];  // Align to 16 bytes
};

using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// Alternative: Empty hit group record for simpler setups
using EmptyHitGroupSbtRecord = EmptySbtRecord;

// ============================================================================
// Exception Record Data (optional)
// ============================================================================

struct ExceptionData {
    // Exception handling parameters
    int debugMode;
    int padding[3];
};

using ExceptionSbtRecord = SbtRecord<ExceptionData>;

// ============================================================================
// Callable Record Data (optional, for future use)
// ============================================================================

struct CallableData {
    // For direct/continuation callables
    int functionIndex;
    int padding[3];
};

using CallableSbtRecord = SbtRecord<CallableData>;

// ============================================================================
// SBT Configuration Helper
// ============================================================================

struct SbtConfig {
    // Ray types
    static constexpr uint32_t RAY_TYPE_RADIANCE = 0;
    static constexpr uint32_t RAY_TYPE_SHADOW   = 1;  // Reserved for future
    static constexpr uint32_t NUM_RAY_TYPES     = 1;  // Currently only radiance

    // Geometry types
    static constexpr uint32_t GEOMETRY_TYPE_ELLIPSOID = 0;
    static constexpr uint32_t NUM_GEOMETRY_TYPES      = 1;

    // Calculate SBT indices
    static constexpr uint32_t getSbtOffset(uint32_t rayType, uint32_t geometryType) {
        return rayType + geometryType * NUM_RAY_TYPES;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace sbt {

// Initialize SBT records with proper header packing
inline void packHeader(OptixProgramGroup programGroup, void* record) {
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, record));
}

// Allocate and upload SBT record to device
template <typename RecordType>
inline CUdeviceptr allocateAndUpload(const RecordType& record) {
    CUdeviceptr devicePtr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeof(RecordType)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(devicePtr),
        &record,
        sizeof(RecordType),
        cudaMemcpyHostToDevice
    ));
    return devicePtr;
}

// Allocate and upload array of SBT records to device
template <typename RecordType>
inline CUdeviceptr allocateAndUploadArray(const RecordType* records, size_t count) {
    CUdeviceptr devicePtr;
    const size_t size = sizeof(RecordType) * count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(devicePtr),
        records,
        size,
        cudaMemcpyHostToDevice
    ));
    return devicePtr;
}

// Free device memory
inline void freeDeviceMemory(CUdeviceptr& ptr) {
    if (ptr != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
        ptr = 0;
    }
}

}  // namespace sbt
