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

#include <torch/extension.h>
#include <optix.h>
#include "Forward.h"
#include "GAS.h"

// Simplified input structure for trace_rays
struct TraceRaysInput {
    torch::Tensor mean;
    torch::Tensor scale;
    torch::Tensor quat;
    torch::Tensor density;
    torch::Tensor color;
    torch::Tensor rayo;
    torch::Tensor rayd;
    float tmin;
    float tmax;
    float max_prim_size;
    torch::Tensor wcts;  // Optional, can be empty
    int max_iters;
};

// Output structure
struct TraceRaysOutput {
    torch::Tensor color_and_loss;  // [num_rays, 5] - RGB + depth + distortion
    torch::Tensor tri_collection;
    torch::Tensor iters;
    torch::Tensor opacity;
    torch::Tensor touch_count;
    torch::Tensor distortion_loss;
};

// Saved tensors for backward pass
struct SavedForBackward {
    torch::Tensor states;
    torch::Tensor diracs;
    torch::Tensor iters;
    torch::Tensor tri_collection;
    torch::Tensor initial_drgb;
    torch::Tensor initial_inds;
    torch::Tensor initial_touch_count;

    // Input tensors needed for backward
    torch::Tensor mean;
    torch::Tensor scale;
    torch::Tensor quat;
    torch::Tensor density;
    torch::Tensor color;
    torch::Tensor rayo;
    torch::Tensor rayd;
    torch::Tensor wcts;

    // Scalar params
    float tmin;
    float tmax;
    float max_prim_size;
    int max_iters;
    size_t num_prims;
    size_t num_rays;
    size_t feature_size;
};

class SplineTracerFunction : public torch::autograd::Function<SplineTracerFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        OptixDeviceContext optix_context,
        int8_t device_index,
        const torch::Tensor& mean,
        const torch::Tensor& scale,
        const torch::Tensor& quat,
        const torch::Tensor& density,
        const torch::Tensor& color,
        const torch::Tensor& rayo,
        const torch::Tensor& rayd,
        float tmin,
        float tmax,
        float max_prim_size,
        const torch::Tensor& wcts,
        int max_iters
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    );
};

// Simplified API for Python binding
torch::Tensor trace_rays_autograd(
    OptixDeviceContext optix_context,
    int8_t device_index,
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& quat,
    const torch::Tensor& density,
    const torch::Tensor& color,
    const torch::Tensor& rayo,
    const torch::Tensor& rayd,
    float tmin,
    float tmax,
    float max_prim_size,
    const torch::Tensor& wcts,
    int max_iters
);
