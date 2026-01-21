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

// Mirror of DualModel in Slang
struct DualModel {
    // Inputs
    float* means;          // [N, 3]
    float* scales;         // [N, 3]
    float* quats;          // [N, 4]
    float* densities;      // [N]
    float* features;       // [N, feature_size, 3]

    // Gradients (outputs)
    float* dL_dmeans;      // [N, 3]
    float* dL_dscales;     // [N, 3]
    float* dL_dquats;      // [N, 4]
    float* dL_ddensities;  // [N]
    float* dL_dfeatures;   // [N, feature_size, 3]
    float* dL_drayos;      // [num_rays, 3]
    float* dL_drayds;      // [num_rays, 3]
    float* dL_dmeans2D;    // [N, 2]

    // Dimensions
    uint32_t num_prims;
    uint32_t feature_size;
};

// Mirror of BackwardParams in Slang
struct BackwardParams {
    // Saved tensors from forward
    float* last_state;        // [num_rays, 16]
    float* last_dirac;        // [num_rays, 4]
    int* iters;               // [num_rays]
    int* tri_collection;      // [num_rays * max_iters]

    // Ray data
    float* ray_origins;       // [num_rays, 3]
    float* ray_directions;    // [num_rays, 3]

    // Initial touch data
    float* initial_drgb;      // [num_rays, 4]
    float* dL_dinital_drgb;   // [num_rays, 4]
    int* touch_count;         // [num_prims]
    int* initial_inds;        // [num_initial]

    // Gradient input
    float* dL_doutputs;       // [num_rays, 5]

    // World-camera transforms
    float* wcts;              // [num_wcts, 4, 4]

    // Model data
    DualModel model;

    // Scalar params
    float tmin;
    float tmax;
    float max_prim_size;
    uint32_t max_iters;
    uint32_t num_rays;
    uint32_t num_wcts;
    uint32_t num_initial;
};

// Mirror of InitialDrgbParams in Slang
struct InitialDrgbParams {
    float* ray_origins;
    float* ray_directions;
    DualModel model;
    float* initial_drgb;
    int* initial_inds;
    float* dL_dinital_drgb;
    int* touch_count;
    float tmin;
    uint32_t num_rays;
    uint32_t num_initial;
};

// Launch functions (implemented in backward_kernel.cu)
void launch_backwards_kernel(
    const BackwardParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream = nullptr
);

void launch_backwards_initial_drgb_kernel(
    const InitialDrgbParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream = nullptr
);
