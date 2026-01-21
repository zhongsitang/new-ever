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

#include <torch/torch.h>
#include <vector>

// Simplified C++ API for Python binding
// All internal types (TensorView, DualModel, etc.) are implementation details
// defined in SplineTracerAutograd.cu to avoid conflicts with slang-generated code.
std::vector<torch::Tensor> trace_rays_autograd(
    torch::Tensor mean,
    torch::Tensor scale,
    torch::Tensor quat,
    torch::Tensor density,
    torch::Tensor features,
    torch::Tensor rayo,
    torch::Tensor rayd,
    torch::Tensor wcts,
    float tmin,
    float tmax,
    float max_prim_size,
    int max_iters,
    int64_t optix_context_ptr,
    int64_t gas_handle
);
