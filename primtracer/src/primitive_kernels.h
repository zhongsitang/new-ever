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

#include "types.h"

// =============================================================================
// CUDA Kernels for Primitive Operations
// =============================================================================
//
// This module provides efficient CUDA operators for the ray tracing pipeline:
//
// 1. AABB Computation:
//    - compute_primitive_aabbs: Build axis-aligned bounding boxes for ellipsoids
//
// 2. Initial Sample Accumulation:
//    - init_ray_start_samples: Handle rays starting inside primitives
//    - init_ray_start_samples_single: Single-ray optimized version
//    - init_ray_start_samples_zero: No-op placeholder
//
// =============================================================================

/// Compute axis-aligned bounding boxes for ellipsoid primitives.
/// Used by AccelStructure to build the OptiX GAS.
///
/// @param prims  Primitive data (means, scales, quats)
/// @param aabbs  Output buffer for AABBs (must be pre-allocated)
void compute_primitive_aabbs(const Primitives& prims, OptixAabb* aabbs);

/// Initialize contributions for rays starting inside primitives.
/// Finds primitives whose AABBs contain the ray origin and accumulates
/// their density/color contributions to initial_contrib.
///
/// @param params       Launch parameters with ray data and primitives
/// @param aabbs        Pre-computed AABBs for primitives
/// @param d_hit_count  Optional: reusable device buffer for hit count
/// @param d_hit_inds   Optional: reusable device buffer for hit indices
void init_ray_start_samples(Params* params, OptixAabb* aabbs,
                            int* d_hit_count = nullptr,
                            int* d_hit_inds = nullptr);

/// Single-ray version of init_ray_start_samples.
/// Optimized for single ray origin, checks all primitives directly.
void init_ray_start_samples_single(Params* params);

/// No-op version when initial samples are not needed.
void init_ray_start_samples_zero(Params* params);
