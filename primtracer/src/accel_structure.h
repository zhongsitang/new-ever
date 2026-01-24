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
#include "types.h"

class DeviceContext;

// =============================================================================
// Free function for AABB computation (implemented in primitive_kernels.cu)
// =============================================================================

/// Compute AABBs for primitives into pre-allocated buffer
void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs);

// =============================================================================
// AccelStructure - AABB and GAS management with buffer reuse
// =============================================================================

/// Manages acceleration structure: AABB buffer and GAS (Geometry Acceleration Structure).
/// Supports efficient rebuilding with buffer capacity tracking to avoid repeated allocations.
class AccelStructure {
public:
    /// Create an empty acceleration structure (call rebuild() to populate)
    explicit AccelStructure(DeviceContext& ctx);

    /// Create and build acceleration structure for primitives
    AccelStructure(DeviceContext& ctx, const Primitives& prims);

    ~AccelStructure();

    // Non-copyable
    AccelStructure(const AccelStructure&) = delete;
    AccelStructure& operator=(const AccelStructure&) = delete;

    /// Rebuild acceleration structure for new primitives.
    /// Reuses existing buffers when capacity is sufficient.
    void rebuild(const Primitives& prims);

    OptixTraversableHandle handle() const { return gas_handle_; }
    OptixAabb* aabbs() const { return aabb_buffer_; }
    size_t num_prims() const { return num_prims_; }

private:
    void ensure_aabb_capacity(size_t num_prims);
    void ensure_gas_capacity(size_t output_size, size_t temp_size);
    void build_gas(size_t num_prims);

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
