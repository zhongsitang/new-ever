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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "volume_types.h"

using uint = uint32_t;

// Embedded PTX code (generated header)
#include "shaders_ptx.h"

// SBT record types
struct RayGenData {};
struct MissData { float3 bg_color; };
struct HitGroupData {};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

// Launch parameters - must match slang layout exactly
// Note: StructuredBuffer<T> = {T* data, size_t size} (16 bytes with padding)
struct Params {
    StructuredBuffer<float4> image;                    // Rendered RGBA output
    StructuredBuffer<float> depth_out;                 // Rendered depth output
    StructuredBuffer<uint> iters;                      // Iteration count per ray
    StructuredBuffer<uint> last_prim;                  // Last primitive hit
    StructuredBuffer<uint> prim_hits;                  // Hit count per primitive
    StructuredBuffer<float4> last_delta_contrib;       // Last sample delta contribution
    StructuredBuffer<IntegratorState> last_state;      // Final volume state per ray
    StructuredBuffer<int> hit_collection;              // Collected hit IDs for backward pass
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<float3> means;                    // Ellipsoid centers
    StructuredBuffer<float3> scales;                   // Ellipsoid radii
    StructuredBuffer<float4> quats;                    // Ellipsoid rotations (quaternion wxyz)
    StructuredBuffer<float> densities;                 // Ellipsoid densities
    StructuredBuffer<float> features;                  // SH coefficients for color

    size_t sh_degree;                                  // Spherical harmonics degree
    size_t max_iters;                                  // Maximum iterations per ray
    float tmin;                                        // Minimum ray t
    StructuredBuffer<float> tmax;                      // Maximum ray t (per-ray)
    StructuredBuffer<float4> initial_contrib;          // Initial accumulated contribution
    float max_prim_size;                               // Maximum primitive size
    OptixTraversableHandle handle;                     // BVH acceleration structure
};

// Forward declaration
class GAS;

/// State saved during forward pass for backward gradient computation.
struct SavedState {
    IntegratorState* states;        // (M, 12) volume integrator state per ray
    float4* delta_contribs;         // (M, 4) last delta contribution
    uint* iters;                    // (M,) iteration count per ray
    uint* prim_hits;                // (N,) hit count per primitive
    int* hit_collection;            // (M * max_iters,) hit primitive indices
    float4* initial_contrib;        // (M, 4) contribution for rays starting inside
    int* initial_prim_indices;      // (N,) primitives containing ray origins
    int* initial_prim_count;        // (1,) count of initial_prim_indices
};

/// RayPipeline: Complete ray tracing pipeline for ellipsoid volume rendering.
///
/// This class manages:
/// - OptiX context (lazy-initialized, cached per device)
/// - Primitive data and AABB computation
/// - Acceleration structure (GAS)
/// - OptiX pipeline and shader binding table
class RayPipeline {
public:
    /// Construct a ray pipeline with primitive data.
    RayPipeline(
        int device_index,
        float3* means,
        float3* scales,
        float4* quats,
        float* densities,
        float* features,
        size_t num_prims,
        size_t feature_size
    );

    ~RayPipeline() noexcept(false);

    // Non-copyable
    RayPipeline(const RayPipeline&) = delete;
    RayPipeline& operator=(const RayPipeline&) = delete;

    /// Trace rays through the scene.
    ///
    /// @param num_rays Number of rays to trace
    /// @param ray_origins Ray origin positions (M, 3)
    /// @param ray_directions Ray directions, should be normalized (M, 3)
    /// @param color_out Output RGBA colors (M, 4)
    /// @param depth_out Output expected depths (M,)
    /// @param sh_degree Spherical harmonics degree
    /// @param tmin Minimum ray parameter
    /// @param tmax Maximum ray parameter per ray (M,)
    /// @param max_iters Maximum hit iterations per ray
    /// @param saved Output buffers for backward pass (can be nullptr if not needed)
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

    size_t num_prims() const { return model_.num_prims; }

private:
    void create_module(const char* ptx);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    int8_t device_ = -1;
    OptixDeviceContext context_ = nullptr;

    // Primitive data and acceleration structure
    Primitives model_ = {};
    std::unique_ptr<GAS> gas_;

    // OptiX pipeline objects
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
    static constexpr size_t LOG_SIZE = 2048;
    char log_[LOG_SIZE];
    size_t log_size_ = LOG_SIZE;
};
