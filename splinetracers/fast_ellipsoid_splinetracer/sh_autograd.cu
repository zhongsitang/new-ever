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

#include "sh_autograd.h"
#include <cmath>

// Include Slang-generated CUDA code (contains kernel implementations and struct definitions)
#include "sh_kernel_cuda.h"

torch::autograd::variable_list EvalSHFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const torch::Tensor& means,
    const torch::Tensor& features,
    const torch::Tensor& ray_origin
) {
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(ray_origin.is_cuda(), "ray_origin must be a CUDA tensor");

    auto means_c = means.contiguous();
    auto features_c = features.contiguous();
    auto ray_origin_c = ray_origin.contiguous();

    const int64_t num_prims = means_c.size(0);
    const int64_t feature_size = features_c.size(1);
    const uint32_t sh_degree = static_cast<uint32_t>(std::sqrt(static_cast<float>(feature_size))) - 1;

    auto device = means_c.device();

    // Allocate output
    auto colors = torch::zeros({num_prims, 3}, torch::device(device).dtype(torch::kFloat32));

    // Setup params
    SHKernelParams params;
    params.means = reinterpret_cast<float*>(means_c.data_ptr());
    params.features = reinterpret_cast<float*>(features_c.data_ptr());
    params.ray_origin = reinterpret_cast<float*>(ray_origin_c.data_ptr());
    params.colors = reinterpret_cast<float*>(colors.data_ptr());
    params.num_prims = static_cast<uint32_t>(num_prims);
    params.sh_degree = sh_degree;
    params.feature_size = static_cast<uint32_t>(feature_size);

    // Launch kernel
    constexpr int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_prims + block_size - 1) / block_size, 1, 1);
    sh_kernel<<<grid, block>>>(params);

    // Save for backward
    ctx->save_for_backward({means_c, features_c, ray_origin_c});
    ctx->saved_data["num_prims"] = static_cast<int64_t>(num_prims);
    ctx->saved_data["feature_size"] = static_cast<int64_t>(feature_size);
    ctx->saved_data["sh_degree"] = static_cast<int64_t>(sh_degree);

    return {colors};
}

torch::autograd::variable_list EvalSHFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto means = saved[0];
    auto features = saved[1];
    auto ray_origin = saved[2];

    int64_t num_prims = ctx->saved_data["num_prims"].toInt();
    int64_t feature_size = ctx->saved_data["feature_size"].toInt();
    int64_t sh_degree = ctx->saved_data["sh_degree"].toInt();

    auto device = means.device();
    auto dL_dcolors = grad_outputs[0].contiguous();

    // Allocate gradient for features
    auto dL_dfeatures = torch::zeros_like(features);

    // Setup params
    SHBackwardParams params;
    params.means = reinterpret_cast<float*>(means.data_ptr());
    params.features = reinterpret_cast<float*>(features.data_ptr());
    params.dL_dfeatures = reinterpret_cast<float*>(dL_dfeatures.data_ptr());
    params.ray_origin = reinterpret_cast<float*>(ray_origin.data_ptr());
    params.dL_dcolors = reinterpret_cast<float*>(dL_dcolors.data_ptr());
    params.num_prims = static_cast<uint32_t>(num_prims);
    params.sh_degree = static_cast<uint32_t>(sh_degree);
    params.feature_size = static_cast<uint32_t>(feature_size);

    // Launch backward kernel
    constexpr int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_prims + block_size - 1) / block_size, 1, 1);
    bw_sh_kernel<<<grid, block>>>(params);

    // Return gradients: means, features, ray_origin
    // Only features has gradient
    return {torch::Tensor(), dL_dfeatures, torch::Tensor()};
}

torch::Tensor eval_sh(
    const torch::Tensor& means,
    const torch::Tensor& features,
    const torch::Tensor& ray_origin
) {
    auto results = EvalSHFunction::apply(means, features, ray_origin);
    return results[0];
}
