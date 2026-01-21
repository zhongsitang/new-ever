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

#include "spline_tracer_autograd.h"
#include "create_aabbs.h"
#include "exception.h"
#include <cmath>

// Global AABB storage (from py_binding.cpp)
extern OptixAabb* D_AABBS;
extern size_t NUM_AABBS;

namespace {

// Helper to ensure tensor is contiguous and on correct device
torch::Tensor ensure_contiguous(const torch::Tensor& t, torch::Device device) {
    auto result = t.contiguous();
    TORCH_CHECK(result.device() == device, "Tensor must be on device ", device);
    return result;
}

// Gradient clipping constants
constexpr float GRAD_CLIP_V = 1e+3f;
constexpr float GRAD_CLIP_MEAN = 1e+3f;
constexpr float GRAD_CLIP_DENSITY = 50.0f;

} // anonymous namespace

torch::autograd::variable_list SplineTracerFunction::forward(
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
) {
    torch::Device device(torch::kCUDA, device_index);
    CUDA_CHECK(cudaSetDevice(device_index));

    // Ensure inputs are contiguous
    auto mean_c = ensure_contiguous(mean, device);
    auto scale_c = ensure_contiguous(scale, device);
    auto quat_c = ensure_contiguous(quat, device);
    auto density_c = ensure_contiguous(density, device);
    auto color_c = ensure_contiguous(color, device);
    auto rayo_c = ensure_contiguous(rayo, device);
    auto rayd_c = ensure_contiguous(rayd, device);

    const size_t num_prims = mean_c.size(0);
    const size_t num_rays = rayo_c.size(0);
    const size_t feature_size = color_c.size(1);
    const uint32_t sh_degree = static_cast<uint32_t>(std::sqrt(static_cast<float>(feature_size))) - 1;

    // Build primitives
    Primitives model;
    model.means = reinterpret_cast<float3*>(mean_c.data_ptr());
    model.scales = reinterpret_cast<float3*>(scale_c.data_ptr());
    model.quats = reinterpret_cast<float4*>(quat_c.data_ptr());
    model.densities = reinterpret_cast<float*>(density_c.data_ptr());
    model.features = reinterpret_cast<float*>(color_c.data_ptr());
    model.num_prims = num_prims;
    model.feature_size = feature_size;
    model.prev_alloc_size = NUM_AABBS;
    model.aabbs = D_AABBS;

    create_aabbs(model);
    D_AABBS = model.aabbs;
    NUM_AABBS = std::max(model.num_prims, NUM_AABBS);

    // Build GAS (enable_backwards=true, fast_build=false)
    GAS gas(optix_context, device_index, model, true, false);

    // Create Forward tracer
    Forward forward(optix_context, device_index, model, true);

    // Allocate output tensors
    auto output_color = torch::zeros({static_cast<long>(num_rays), 4},
        torch::device(device).dtype(torch::kFloat32));
    auto tri_collection = torch::zeros({static_cast<long>(num_rays * max_iters)},
        torch::device(device).dtype(torch::kInt32));
    auto initial_drgb = torch::zeros({static_cast<long>(num_rays), 4},
        torch::device(device).dtype(torch::kFloat32));
    auto initial_touch_count = torch::zeros({1},
        torch::device(device).dtype(torch::kInt32));
    auto initial_touch_inds = torch::zeros({static_cast<long>(num_prims)},
        torch::device(device).dtype(torch::kInt32));

    // Saved for backward
    constexpr size_t state_float_count = sizeof(SplineState) / sizeof(float);
    auto states = torch::zeros({static_cast<long>(num_rays), static_cast<long>(state_float_count)},
        torch::device(device).dtype(torch::kFloat32));
    auto diracs = torch::zeros({static_cast<long>(num_rays), 4},
        torch::device(device).dtype(torch::kFloat32));
    auto iters = torch::zeros({static_cast<long>(num_rays)},
        torch::device(device).dtype(torch::kInt32));
    auto faces = torch::zeros({static_cast<long>(num_rays)},
        torch::device(device).dtype(torch::kInt32));
    auto touch_count = torch::zeros({static_cast<long>(num_prims)},
        torch::device(device).dtype(torch::kInt32));

    // Run forward pass
    forward.trace_rays(
        gas.gas_handle,
        num_rays,
        reinterpret_cast<float3*>(rayo_c.data_ptr()),
        reinterpret_cast<float3*>(rayd_c.data_ptr()),
        reinterpret_cast<void*>(output_color.data_ptr()),
        sh_degree,
        tmin, tmax,
        reinterpret_cast<float4*>(initial_drgb.data_ptr()),
        nullptr,
        max_iters,
        max_prim_size,
        reinterpret_cast<uint32_t*>(iters.data_ptr()),
        reinterpret_cast<uint32_t*>(faces.data_ptr()),
        reinterpret_cast<uint32_t*>(touch_count.data_ptr()),
        reinterpret_cast<float4*>(diracs.data_ptr()),
        reinterpret_cast<SplineState*>(states.data_ptr()),
        reinterpret_cast<int*>(tri_collection.data_ptr()),
        reinterpret_cast<int*>(initial_touch_count.data_ptr()),
        reinterpret_cast<int*>(initial_touch_inds.data_ptr())
    );

    // Compute distortion loss from states
    auto states_reshaped = states.reshape({static_cast<long>(num_rays), -1});
    auto distortion_pt1 = states_reshaped.index({torch::indexing::Slice(), 0});
    auto distortion_pt2 = states_reshaped.index({torch::indexing::Slice(), 1});
    auto distortion_loss = distortion_pt1 - distortion_pt2;

    // Combine color and distortion loss
    auto color_and_loss = torch::cat({output_color, distortion_loss.reshape({-1, 1})}, 1);

    // Get initial indices
    int initial_count = initial_touch_count.item<int>();
    auto initial_inds = initial_touch_inds.index({torch::indexing::Slice(0, initial_count)});

    // Save for backward
    ctx->save_for_backward({
        mean_c, scale_c, quat_c, density_c, color_c, rayo_c, rayd_c,
        tri_collection, wcts.defined() ? wcts : torch::ones({1, 4, 4}, torch::device(device).dtype(torch::kFloat32)),
        initial_drgb, initial_inds, states, diracs, iters
    });

    ctx->saved_data["tmin"] = tmin;
    ctx->saved_data["tmax"] = tmax;
    ctx->saved_data["max_prim_size"] = max_prim_size;
    ctx->saved_data["max_iters"] = max_iters;
    ctx->saved_data["num_prims"] = static_cast<int64_t>(num_prims);
    ctx->saved_data["num_rays"] = static_cast<int64_t>(num_rays);
    ctx->saved_data["feature_size"] = static_cast<int64_t>(feature_size);
    ctx->saved_data["device_index"] = static_cast<int64_t>(device_index);
    ctx->saved_data["has_wcts"] = wcts.defined() && wcts.numel() > 0;

    return {color_and_loss};
}

torch::autograd::variable_list SplineTracerFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto mean = saved[0];
    auto scale = saved[1];
    auto quat = saved[2];
    auto density = saved[3];
    auto features = saved[4];
    auto rayo = saved[5];
    auto rayd = saved[6];
    auto tri_collection = saved[7];
    auto wcts = saved[8];
    auto initial_drgb = saved[9];
    auto initial_inds = saved[10];
    auto states = saved[11];
    auto diracs = saved[12];
    auto iters = saved[13];

    float tmin = ctx->saved_data["tmin"].toDouble();
    float tmax = ctx->saved_data["tmax"].toDouble();
    float max_prim_size = ctx->saved_data["max_prim_size"].toDouble();
    int max_iters = ctx->saved_data["max_iters"].toInt();
    int64_t num_prims = ctx->saved_data["num_prims"].toInt();
    int64_t num_rays = ctx->saved_data["num_rays"].toInt();
    int64_t feature_size = ctx->saved_data["feature_size"].toInt();
    int8_t device_index = static_cast<int8_t>(ctx->saved_data["device_index"].toInt());
    bool has_wcts = ctx->saved_data["has_wcts"].toBool();

    torch::Device device(torch::kCUDA, device_index);
    CUDA_CHECK(cudaSetDevice(device_index));

    auto grad_output = grad_outputs[0].contiguous();

    // Allocate gradient tensors
    auto dL_dmeans = torch::zeros({num_prims, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dscales = torch::zeros({num_prims, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dquats = torch::zeros({num_prims, 4}, torch::device(device).dtype(torch::kFloat32));
    auto dL_ddensities = torch::zeros({num_prims}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dfeatures = torch::zeros_like(features);
    auto dL_drayo = torch::zeros({num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_drayd = torch::zeros({num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dmeans2D = torch::zeros({num_prims, 2}, torch::device(device).dtype(torch::kFloat32));
    auto touch_count_bw = torch::zeros({num_prims}, torch::device(device).dtype(torch::kInt32));
    auto dL_dinital_drgb = torch::zeros({num_rays, 4}, torch::device(device).dtype(torch::kFloat32));

    // Check if we have any iterations
    int64_t total_iters = iters.sum().item<int64_t>();

    if (total_iters > 0) {
        // Setup DualModel
        DualModel dual_model;
        dual_model.means = reinterpret_cast<float*>(mean.data_ptr());
        dual_model.scales = reinterpret_cast<float*>(scale.data_ptr());
        dual_model.quats = reinterpret_cast<float*>(quat.data_ptr());
        dual_model.densities = reinterpret_cast<float*>(density.data_ptr());
        dual_model.features = reinterpret_cast<float*>(features.data_ptr());
        dual_model.dL_dmeans = reinterpret_cast<float*>(dL_dmeans.data_ptr());
        dual_model.dL_dscales = reinterpret_cast<float*>(dL_dscales.data_ptr());
        dual_model.dL_dquats = reinterpret_cast<float*>(dL_dquats.data_ptr());
        dual_model.dL_ddensities = reinterpret_cast<float*>(dL_ddensities.data_ptr());
        dual_model.dL_dfeatures = reinterpret_cast<float*>(dL_dfeatures.data_ptr());
        dual_model.dL_drayos = reinterpret_cast<float*>(dL_drayo.data_ptr());
        dual_model.dL_drayds = reinterpret_cast<float*>(dL_drayd.data_ptr());
        dual_model.dL_dmeans2D = reinterpret_cast<float*>(dL_dmeans2D.data_ptr());
        dual_model.num_prims = static_cast<uint32_t>(num_prims);
        dual_model.feature_size = static_cast<uint32_t>(feature_size);

        // Setup BackwardParams
        BackwardParams params;
        params.last_state = reinterpret_cast<float*>(states.data_ptr());
        params.last_dirac = reinterpret_cast<float*>(diracs.data_ptr());
        params.iters = reinterpret_cast<int*>(iters.data_ptr());
        params.tri_collection = reinterpret_cast<int*>(tri_collection.data_ptr());
        params.ray_origins = reinterpret_cast<float*>(rayo.data_ptr());
        params.ray_directions = reinterpret_cast<float*>(rayd.data_ptr());
        params.initial_drgb = reinterpret_cast<float*>(initial_drgb.data_ptr());
        params.dL_dinital_drgb = reinterpret_cast<float*>(dL_dinital_drgb.data_ptr());
        params.touch_count = reinterpret_cast<int*>(touch_count_bw.data_ptr());
        params.initial_inds = reinterpret_cast<int*>(initial_inds.data_ptr());
        params.dL_doutputs = reinterpret_cast<float*>(grad_output.data_ptr());
        params.wcts = reinterpret_cast<float*>(wcts.data_ptr());
        params.model = dual_model;
        params.tmin = tmin;
        params.tmax = tmax;
        params.max_prim_size = max_prim_size;
        params.max_iters = static_cast<uint32_t>(max_iters);
        params.num_rays = static_cast<uint32_t>(num_rays);
        params.num_wcts = static_cast<uint32_t>(wcts.size(0));
        params.num_initial = static_cast<uint32_t>(initial_inds.size(0));

        // Launch main backward kernel
        constexpr int block_size = 16;
        dim3 block(block_size, 1, 1);
        dim3 grid((num_rays + block_size - 1) / block_size, 1, 1);
        launch_backwards_kernel(params, grid, block, nullptr);

        // Launch initial drgb backward kernel if needed
        if (initial_inds.size(0) > 0) {
            constexpr int ray_block_size = 64;
            constexpr int second_block_size = 16;

            InitialDrgbParams init_params;
            init_params.ray_origins = reinterpret_cast<float*>(rayo.data_ptr());
            init_params.ray_directions = reinterpret_cast<float*>(rayd.data_ptr());
            init_params.model = dual_model;
            init_params.initial_drgb = reinterpret_cast<float*>(initial_drgb.data_ptr());
            init_params.initial_inds = reinterpret_cast<int*>(initial_inds.data_ptr());
            init_params.dL_dinital_drgb = reinterpret_cast<float*>(dL_dinital_drgb.data_ptr());
            init_params.touch_count = reinterpret_cast<int*>(touch_count_bw.data_ptr());
            init_params.tmin = tmin;
            init_params.num_rays = static_cast<uint32_t>(num_rays);
            init_params.num_initial = static_cast<uint32_t>(initial_inds.size(0));

            dim3 init_block(ray_block_size, second_block_size, 1);
            dim3 init_grid(
                (num_rays + ray_block_size - 1) / ray_block_size,
                (initial_inds.size(0) + second_block_size - 1) / second_block_size,
                1
            );
            launch_backwards_initial_drgb_kernel(init_params, init_grid, init_block, nullptr);
        }

        CUDA_SYNC_CHECK();
    }

    // Clip gradients
    dL_dmeans = dL_dmeans.clip(-GRAD_CLIP_MEAN, GRAD_CLIP_MEAN);
    dL_dscales = dL_dscales.clip(-GRAD_CLIP_V, GRAD_CLIP_V);
    dL_dquats = dL_dquats.clip(-GRAD_CLIP_V, GRAD_CLIP_V);
    dL_ddensities = dL_ddensities.clip(-GRAD_CLIP_DENSITY, GRAD_CLIP_DENSITY).reshape(density.sizes());
    dL_dfeatures = dL_dfeatures.clip(-GRAD_CLIP_V, GRAD_CLIP_V);
    dL_drayo = dL_drayo.clip(-GRAD_CLIP_V, GRAD_CLIP_V);
    dL_drayd = dL_drayd.clip(-GRAD_CLIP_V, GRAD_CLIP_V);

    // Return gradients in same order as forward inputs
    // optix_context, device_index have no gradients (first 2)
    // mean, scale, quat, density, color, rayo, rayd (7 tensors)
    // tmin, tmax, max_prim_size (3 scalars - no grad)
    // wcts, max_iters (no grad for these)
    return {
        torch::Tensor(),  // optix_context
        torch::Tensor(),  // device_index
        dL_dmeans,
        dL_dscales,
        dL_dquats,
        dL_ddensities,
        dL_dfeatures,
        dL_drayo,
        dL_drayd,
        torch::Tensor(),  // tmin
        torch::Tensor(),  // tmax
        torch::Tensor(),  // max_prim_size
        has_wcts ? dL_dmeans2D : torch::Tensor(),  // wcts gradient -> dL_dmeans2D
        torch::Tensor(),  // max_iters
    };
}

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
) {
    auto results = SplineTracerFunction::apply(
        optix_context, device_index,
        mean, scale, quat, density, color, rayo, rayd,
        tmin, tmax, max_prim_size, wcts, max_iters
    );
    return results[0];
}
