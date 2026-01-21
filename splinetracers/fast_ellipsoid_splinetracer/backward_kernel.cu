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

#include "backward_kernel.h"

// Include the Slang-generated CUDA code
#include "backwards_kernel_cuda.h"

void launch_backwards_kernel(
    const BackwardParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream)
{
    backwards_kernel<<<grid_size, block_size, 0, stream>>>(params);
}

void launch_backwards_initial_drgb_kernel(
    const InitialDrgbParams& params,
    dim3 grid_size,
    dim3 block_size,
    cudaStream_t stream)
{
    backwards_initial_drgb_kernel<<<grid_size, block_size, 0, stream>>>(params);
}
