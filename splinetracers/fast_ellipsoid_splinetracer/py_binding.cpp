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

#include <iostream>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/extension.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "Forward.h"
#include "GAS.h"
#include "create_aabbs.h"
#include "exception.h"
#include "backward_kernel.h"

namespace py = pybind11;
using namespace pybind11::literals;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {}

// Global AABB buffer
static OptixAabb *G_AABBS = nullptr;
static size_t G_NUM_AABBS = 0;

// Global OptixContext (singleton per device)
static std::unordered_map<int, OptixDeviceContext> g_optix_contexts;

static OptixDeviceContext get_optix_context(int device_index) {
    auto it = g_optix_contexts.find(device_index);
    if (it != g_optix_contexts.end()) {
        return it->second;
    }

    CUDA_CHECK(cudaSetDevice(device_index));
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0;
    OptixDeviceContext context;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    g_optix_contexts[device_index] = context;
    return context;
}

// Saved tensors for backward pass
struct SplineTracerSaved : public torch::autograd::AutogradContext {
    torch::Tensor states, iters, tri_collection;
    torch::Tensor means, scales, quats, densities, features;
    torch::Tensor rayo, rayd, wcts;
    torch::Tensor initial_drgb, initial_inds;
    float tmin, tmax, max_prim_size;
    size_t max_iters;
    bool has_wcts;
};

// Forward declaration
class SplineTracerFunction : public torch::autograd::Function<SplineTracerFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor& mean,
        const torch::Tensor& scale,
        const torch::Tensor& quat,
        const torch::Tensor& density,
        const torch::Tensor& features,
        const torch::Tensor& rayo,
        const torch::Tensor& rayd,
        float tmin,
        float tmax,
        float max_prim_size,
        const torch::Tensor& wcts,
        size_t max_iters
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

torch::autograd::variable_list SplineTracerFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& quat,
    const torch::Tensor& density,
    const torch::Tensor& features,
    const torch::Tensor& rayo,
    const torch::Tensor& rayd,
    float tmin,
    float tmax,
    float max_prim_size,
    const torch::Tensor& wcts,
    size_t max_iters
) {
    torch::NoGradGuard no_grad;

    auto device = rayo.device();
    int device_index = device.index();
    OptixDeviceContext optix_ctx = get_optix_context(device_index);

    // Make contiguous
    auto mean_c = mean.contiguous();
    auto scale_c = scale.contiguous();
    auto quat_c = quat.contiguous();
    auto density_c = density.contiguous();
    auto features_c = features.contiguous();
    auto rayo_c = rayo.contiguous();
    auto rayd_c = rayd.contiguous();

    size_t num_prims = mean_c.size(0);
    size_t num_rays = rayo_c.size(0);
    size_t feature_size = features_c.size(1);
    uint sh_degree = static_cast<uint>(sqrt(static_cast<double>(feature_size))) - 1;

    // Create half_attribs for GAS
    auto half_attribs = torch::cat({mean_c, scale_c, quat_c}, 1).to(torch::kFloat16).contiguous();

    // Setup primitives
    Primitives model;
    model.means = reinterpret_cast<float3*>(mean_c.data_ptr<float>());
    model.scales = reinterpret_cast<float3*>(scale_c.data_ptr<float>());
    model.quats = reinterpret_cast<float4*>(quat_c.data_ptr<float>());
    model.densities = density_c.data_ptr<float>();
    model.features = features_c.data_ptr<float>();
    model.half_attribs = reinterpret_cast<half*>(half_attribs.data_ptr<at::Half>());
    model.num_prims = num_prims;
    model.feature_size = feature_size;
    model.prev_alloc_size = G_NUM_AABBS;
    model.aabbs = G_AABBS;

    create_aabbs(model);
    G_AABBS = model.aabbs;
    G_NUM_AABBS = std::max(model.num_prims, G_NUM_AABBS);

    // Build GAS
    GAS gas(optix_ctx, device_index, model, true, false);

    // Create Forward tracer
    Forward forward(optix_ctx, device_index, model, true);

    // Allocate outputs
    auto color = torch::zeros({(long)num_rays, 4}, torch::device(device).dtype(torch::kFloat32));
    auto tri_collection = torch::zeros({(long)(num_rays * max_iters)}, torch::device(device).dtype(torch::kInt32));
    auto initial_drgb = torch::zeros({(long)num_rays, 4}, torch::device(device).dtype(torch::kFloat32));
    auto initial_touch_count = torch::zeros({1}, torch::device(device).dtype(torch::kInt32));
    auto initial_touch_inds = torch::zeros({(long)num_prims}, torch::device(device).dtype(torch::kInt32));

    // State tensors
    size_t num_float_per_state = sizeof(SplineState) / sizeof(float);
    auto states = torch::zeros({(long)num_rays, (long)num_float_per_state}, torch::device(device).dtype(torch::kFloat32));
    auto diracs = torch::zeros({(long)num_rays, 4}, torch::device(device).dtype(torch::kFloat32));
    auto faces = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kInt32));
    auto touch_count = torch::zeros({(long)num_prims}, torch::device(device).dtype(torch::kInt32));
    auto iters = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kInt32));

    // Trace rays
    forward.trace_rays(
        gas.gas_handle, num_rays,
        reinterpret_cast<float3*>(rayo_c.data_ptr<float>()),
        reinterpret_cast<float3*>(rayd_c.data_ptr<float>()),
        color.data_ptr<void>(),
        sh_degree, tmin, tmax,
        reinterpret_cast<float4*>(initial_drgb.data_ptr<float>()),
        nullptr,
        max_iters, max_prim_size,
        reinterpret_cast<uint*>(iters.data_ptr<int>()),
        reinterpret_cast<uint*>(faces.data_ptr<int>()),
        reinterpret_cast<uint*>(touch_count.data_ptr<int>()),
        reinterpret_cast<float4*>(diracs.data_ptr<float>()),
        reinterpret_cast<SplineState*>(states.data_ptr<float>()),
        tri_collection.data_ptr<int>(),
        initial_touch_count.data_ptr<int>(),
        initial_touch_inds.data_ptr<int>()
    );

    // Compute distortion loss
    auto states_reshaped = states.reshape({(long)num_rays, -1});
    auto distortion_pt1 = states_reshaped.select(1, 0);
    auto distortion_pt2 = states_reshaped.select(1, 1);
    auto distortion_loss = distortion_pt1 - distortion_pt2;

    // Get initial_inds
    int touch_cnt = initial_touch_count.item<int>();
    auto initial_inds = initial_touch_inds.slice(0, 0, touch_cnt);

    // Save for backward
    ctx->save_for_backward({
        states, iters, tri_collection,
        mean_c, scale_c, quat_c, density_c, features_c,
        rayo_c, rayd_c, wcts,
        initial_drgb, initial_inds
    });
    ctx->saved_data["tmin"] = tmin;
    ctx->saved_data["tmax"] = tmax;
    ctx->saved_data["max_prim_size"] = max_prim_size;
    ctx->saved_data["max_iters"] = static_cast<int64_t>(max_iters);
    ctx->saved_data["has_wcts"] = wcts.numel() > 0;

    // Output: [color (4), distortion_loss (1)]
    auto output = torch::cat({color, distortion_loss.unsqueeze(1)}, 1);

    return {output};
}

torch::autograd::variable_list SplineTracerFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto states = saved[0];
    auto iters = saved[1];
    auto tri_collection = saved[2];
    auto mean = saved[3];
    auto scale = saved[4];
    auto quat = saved[5];
    auto density = saved[6];
    auto features = saved[7];
    auto rayo = saved[8];
    auto rayd = saved[9];
    auto wcts = saved[10];
    auto initial_drgb = saved[11];
    auto initial_inds = saved[12];

    float tmin = ctx->saved_data["tmin"].toDouble();
    float tmax = ctx->saved_data["tmax"].toDouble();
    float max_prim_size = ctx->saved_data["max_prim_size"].toDouble();
    size_t max_iters = ctx->saved_data["max_iters"].toInt();
    bool has_wcts = ctx->saved_data["has_wcts"].toBool();

    auto device = rayo.device();
    size_t num_prims = mean.size(0);
    size_t num_rays = rayo.size(0);
    size_t feature_size = features.size(1);

    // Allocate gradients
    auto dL_dmeans = torch::zeros({(long)num_prims, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dscales = torch::zeros({(long)num_prims, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dquats = torch::zeros({(long)num_prims, 4}, torch::device(device).dtype(torch::kFloat32));
    auto dL_ddensities = torch::zeros({(long)num_prims}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dfeatures = torch::zeros_like(features);
    auto dL_drayo = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_drayd = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
    auto dL_dmeans2D = torch::zeros({(long)num_prims, 2}, torch::device(device).dtype(torch::kFloat32));
    auto touch_count = torch::zeros({(long)num_prims}, torch::device(device).dtype(torch::kInt32));
    auto dL_dinitial_drgb = torch::zeros({(long)num_rays, 4}, torch::device(device).dtype(torch::kFloat32));

    auto grad_output = grad_outputs[0].contiguous();
    auto wcts_safe = has_wcts ? wcts : torch::ones({1, 4, 4}, torch::device(device).dtype(torch::kFloat32));

    if (iters.sum().item<int>() > 0) {
        launch_backwards_kernel(
            states.data_ptr<float>(),
            iters.data_ptr<int>(),
            tri_collection.data_ptr<int>(),
            rayo.data_ptr<float>(),
            rayd.data_ptr<float>(),
            mean.data_ptr<float>(),
            scale.data_ptr<float>(),
            quat.data_ptr<float>(),
            density.data_ptr<float>(),
            features.data_ptr<float>(),
            dL_dmeans.data_ptr<float>(),
            dL_dscales.data_ptr<float>(),
            dL_dquats.data_ptr<float>(),
            dL_ddensities.data_ptr<float>(),
            dL_dfeatures.data_ptr<float>(),
            dL_drayo.data_ptr<float>(),
            dL_drayd.data_ptr<float>(),
            dL_dmeans2D.data_ptr<float>(),
            dL_dinitial_drgb.data_ptr<float>(),
            touch_count.data_ptr<int>(),
            grad_output.data_ptr<float>(),
            wcts_safe.data_ptr<float>(),
            tmin, tmax, max_prim_size,
            static_cast<uint32_t>(max_iters),
            static_cast<uint32_t>(num_rays),
            static_cast<uint32_t>(num_prims),
            static_cast<uint32_t>(feature_size),
            static_cast<uint32_t>(wcts_safe.size(0)),
            nullptr
        );

        if (initial_inds.size(0) > 0) {
            launch_backwards_initial_drgb_kernel(
                rayo.data_ptr<float>(),
                rayd.data_ptr<float>(),
                mean.data_ptr<float>(),
                scale.data_ptr<float>(),
                quat.data_ptr<float>(),
                density.data_ptr<float>(),
                features.data_ptr<float>(),
                dL_ddensities.data_ptr<float>(),
                dL_dfeatures.data_ptr<float>(),
                initial_inds.data_ptr<int>(),
                dL_dinitial_drgb.data_ptr<float>(),
                touch_count.data_ptr<int>(),
                tmin,
                static_cast<uint32_t>(num_rays),
                static_cast<uint32_t>(initial_inds.size(0)),
                static_cast<uint32_t>(feature_size),
                nullptr
            );
        }
    }

    // Clamp gradients
    float v = 1e3f;
    float mean_v = 1e3f;
    dL_dmeans = dL_dmeans.clamp(-mean_v, mean_v);
    dL_dscales = dL_dscales.clamp(-v, v);
    dL_dquats = dL_dquats.clamp(-v, v);
    dL_ddensities = dL_ddensities.clamp(-50.0f, 50.0f).reshape(density.sizes());
    dL_dfeatures = dL_dfeatures.clamp(-v, v);
    dL_drayo = dL_drayo.clamp(-v, v);
    dL_drayd = dL_drayd.clamp(-v, v);

    if (!has_wcts) {
        dL_dmeans2D = torch::Tensor();  // None
    }

    // Return gradients for: mean, scale, quat, density, features, rayo, rayd, tmin, tmax, max_prim_size, wcts, max_iters
    return {
        dL_dmeans,      // mean
        dL_dscales,     // scale
        dL_dquats,      // quat
        dL_ddensities,  // density
        dL_dfeatures,   // features
        dL_drayo,       // rayo
        dL_drayd,       // rayd
        torch::Tensor(), // tmin (no grad)
        torch::Tensor(), // tmax (no grad)
        torch::Tensor(), // max_prim_size (no grad)
        dL_dmeans2D,     // wcts -> returns dL_dmeans2D
        torch::Tensor()  // max_iters (no grad)
    };
}

// Simple wrapper function
torch::Tensor trace_rays(
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& quat,
    const torch::Tensor& density,
    const torch::Tensor& features,
    const torch::Tensor& rayo,
    const torch::Tensor& rayd,
    float tmin,
    float tmax,
    float max_prim_size,
    const torch::Tensor& wcts,
    size_t max_iters
) {
    auto result = SplineTracerFunction::apply(
        mean, scale, quat, density, features,
        rayo, rayd, tmin, tmax, max_prim_size, wcts, max_iters
    );
    return result[0];
}

PYBIND11_MODULE(ellipsoid_splinetracer, m) {
    m.def("trace_rays", &trace_rays,
          "Trace rays through ellipsoid spline primitives with automatic differentiation",
          "mean"_a, "scale"_a, "quat"_a, "density"_a, "features"_a,
          "rayo"_a, "rayd"_a,
          "tmin"_a = 0.0f, "tmax"_a = 1000.0f, "max_prim_size"_a = 3.0f,
          "wcts"_a = torch::Tensor(), "max_iters"_a = 500);
}
