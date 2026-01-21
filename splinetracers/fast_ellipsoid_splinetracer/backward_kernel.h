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

#include <cuda_runtime.h>
#include <cstdint>

// Launch the main backward kernel (Slang-compiled)
void launch_backwards_kernel(
    float* last_state,          // [num_rays, 16]
    int* iters,                 // [num_rays]
    int* tri_collection,        // [num_rays * max_iters]
    float* ray_origins,         // [num_rays, 3]
    float* ray_directions,      // [num_rays, 3]
    float* means,               // [num_prims, 3]
    float* scales,              // [num_prims, 3]
    float* quats,               // [num_prims, 4]
    float* densities,           // [num_prims]
    float* features,            // [num_prims, feature_size, 3]
    float* dL_dmeans,           // [num_prims, 3]
    float* dL_dscales,          // [num_prims, 3]
    float* dL_dquats,           // [num_prims, 4]
    float* dL_ddensities,       // [num_prims]
    float* dL_dfeatures,        // [num_prims, feature_size, 3]
    float* dL_drayos,           // [num_rays, 3]
    float* dL_drayds,           // [num_rays, 3]
    float* dL_dmeans2D,         // [num_prims, 2]
    float* dL_dinitial_drgb,    // [num_rays, 4]
    int* touch_count,           // [num_prims]
    float* dL_doutputs,         // [num_rays, 5]
    float* wcts,                // [num_wcts, 4, 4]
    float tmin,
    float tmax,
    float max_prim_size,
    uint32_t max_iters,
    uint32_t num_rays,
    uint32_t num_prims,
    uint32_t feature_size,
    uint32_t num_wcts,
    cudaStream_t stream = nullptr
);

// Launch the initial drgb backward kernel (Slang-compiled)
void launch_backwards_initial_drgb_kernel(
    float* ray_origins,
    float* ray_directions,
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    float* dL_ddensities,
    float* dL_dfeatures,
    int* initial_inds,
    float* dL_dinitial_drgb,
    int* touch_count,
    float tmin,
    uint32_t num_rays,
    uint32_t num_initial_inds,
    uint32_t feature_size,
    cudaStream_t stream = nullptr
);
