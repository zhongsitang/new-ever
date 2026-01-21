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

// Mirror of SHKernelParams in Slang
struct SHKernelParams {
    float* means;        // [N, 3]
    float* features;     // [N, feature_size, 3]
    float* ray_origin;   // [3]
    float* colors;       // [N, 3]
    uint32_t num_prims;
    uint32_t sh_degree;
    uint32_t feature_size;
};

// Mirror of SHBackwardParams in Slang
struct SHBackwardParams {
    float* means;           // [N, 3]
    float* features;        // [N, feature_size, 3]
    float* dL_dfeatures;    // [N, feature_size, 3]
    float* ray_origin;      // [3]
    float* dL_dcolors;      // [N, 3]
    uint32_t num_prims;
    uint32_t sh_degree;
    uint32_t feature_size;
};

// Launch functions (implemented in sh_kernel.cu)
void launch_sh_kernel(
    const SHKernelParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream = nullptr
);

void launch_bw_sh_kernel(
    const SHBackwardParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream = nullptr
);
