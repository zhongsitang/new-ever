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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <optix_function_table_definition.h>

#include "ray_pipeline.h"

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Tensor validation macros
// =============================================================================

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x, device) \
    TORCH_CHECK(x.device() == device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x, device, dim) \
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DEVICE(x, device); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == dim, #x " must have last dimension " #dim)

// Helper for casting tensor data pointer to custom types
template<typename T>
T* data_ptr(const torch::Tensor& t) { return static_cast<T*>(t.data_ptr()); }

// =============================================================================
// Main API: trace_rays
// =============================================================================

py::dict trace_rays(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& densities,
    const torch::Tensor& features,
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_directions,
    float tmin,
    const torch::Tensor& tmax,
    size_t max_iters)
{
    torch::AutoGradMode enable_grad(false);

    // Get device
    const auto device = means.device();
    const int device_index = device.index();

    // Validate inputs
    CHECK_INPUT(means, device, 3);
    CHECK_INPUT(scales, device, 3);
    CHECK_INPUT(quats, device, 4);
    CHECK_INPUT(features, device, 3);
    CHECK_INPUT(ray_origins, device, 3);
    CHECK_INPUT(ray_directions, device, 3);
    CHECK_CUDA(densities); CHECK_CONTIGUOUS(densities);
    CHECK_DEVICE(densities, device); CHECK_FLOAT(densities);
    CHECK_CUDA(tmax); CHECK_CONTIGUOUS(tmax);
    CHECK_DEVICE(tmax, device); CHECK_FLOAT(tmax);

    const size_t num_prims = means.size(0);
    const size_t num_rays = ray_origins.size(0);
    const size_t feature_size = features.size(1);
    const uint sh_degree = static_cast<uint>(sqrt(feature_size)) - 1;

    TORCH_CHECK(scales.size(0) == (long)num_prims, "scales must match means count");
    TORCH_CHECK(quats.size(0) == (long)num_prims, "quats must match means count");
    TORCH_CHECK(densities.size(0) == (long)num_prims, "densities must match means count");
    TORCH_CHECK(features.size(0) == (long)num_prims, "features must match means count");
    TORCH_CHECK(tmax.numel() == (long)num_rays, "tmax must have one value per ray");

    // Create pipeline (handles context, primitives, GAS internally)
    Primitives prims = {
        .means = data_ptr<float3>(means),
        .scales = data_ptr<float3>(scales),
        .quats = data_ptr<float4>(quats),
        .densities = data_ptr<float>(densities),
        .num_prims = num_prims,
        .features = data_ptr<float>(features),
        .feature_size = feature_size,
    };
    RayPipeline pipeline(device_index, prims);

    // Allocate output tensors
    auto opts_f = torch::device(device).dtype(torch::kFloat32);
    auto opts_i = torch::device(device).dtype(torch::kInt32);

    torch::Tensor color = torch::zeros({(long)num_rays, 4}, opts_f);
    torch::Tensor depth = torch::zeros({(long)num_rays}, opts_f);

    // Allocate backward state tensors
    constexpr size_t state_floats = sizeof(IntegratorState) / sizeof(float);
    torch::Tensor states = torch::zeros({(long)num_rays, (long)state_floats}, opts_f);
    torch::Tensor delta_contribs = torch::zeros({(long)num_rays, 4}, opts_f);
    torch::Tensor iters = torch::zeros({(long)num_rays}, opts_i);
    torch::Tensor prim_hits = torch::zeros({(long)num_prims}, opts_i);
    torch::Tensor hit_collection = torch::zeros({(long)(num_rays * max_iters)}, opts_i);
    torch::Tensor initial_contrib = torch::zeros({(long)num_rays, 4}, opts_f);
    torch::Tensor initial_prim_indices = torch::zeros({(long)num_prims}, opts_i);
    torch::Tensor initial_prim_count = torch::zeros({1}, opts_i);

    // Setup backward state
    SavedState saved = {
        .states = data_ptr<IntegratorState>(states),
        .delta_contribs = data_ptr<float4>(delta_contribs),
        .iters = data_ptr<uint>(iters),
        .prim_hits = data_ptr<uint>(prim_hits),
        .hit_collection = data_ptr<int>(hit_collection),
        .initial_contrib = data_ptr<float4>(initial_contrib),
        .initial_prim_indices = data_ptr<int>(initial_prim_indices),
        .initial_prim_count = data_ptr<int>(initial_prim_count),
    };

    // Trace rays
    pipeline.trace_rays(
        num_rays,
        data_ptr<float3>(ray_origins),
        data_ptr<float3>(ray_directions),
        data_ptr<float4>(color),
        data_ptr<float>(depth),
        sh_degree,
        tmin,
        data_ptr<float>(tmax),
        max_iters,
        &saved
    );

    return py::dict(
        "color"_a = color,
        "depth"_a = depth,
        // Backward state
        "states"_a = states,
        "delta_contribs"_a = delta_contribs,
        "iters"_a = iters,
        "prim_hits"_a = prim_hits,
        "hit_collection"_a = hit_collection,
        "initial_contrib"_a = initial_contrib,
        "initial_prim_indices"_a = initial_prim_indices,
        "initial_prim_count"_a = initial_prim_count
    );
}

// =============================================================================
// Python module definition
// =============================================================================

PYBIND11_MODULE(ellipsoid_tracer, m) {
    m.doc() = "Differentiable volume rendering for ellipsoid primitives";

    m.def("trace_rays", &trace_rays,
          py::arg("means"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("densities"),
          py::arg("features"),
          py::arg("ray_origins"),
          py::arg("ray_directions"),
          py::arg("tmin"),
          py::arg("tmax"),
          py::arg("max_iters"),
          R"doc(
Trace rays through ellipsoid primitives using volume rendering.

Args:
    means: Primitive centers, shape (N, 3)
    scales: Primitive scales, shape (N, 3)
    quats: Primitive rotations as quaternions (w,x,y,z), shape (N, 4)
    densities: Primitive densities, shape (N,)
    features: SH features, shape (N, C, 3) where C is number of SH coefficients
    ray_origins: Ray origins, shape (M, 3)
    ray_directions: Ray directions (normalized), shape (M, 3)
    tmin: Minimum ray parameter (scalar)
    tmax: Maximum ray parameter per ray, shape (M,)
    max_iters: Maximum hit iterations per ray

Returns:
    Dictionary containing:
        - color: RGBA output, shape (M, 4)
        - depth: Expected depth, shape (M,)
        - states, delta_contribs, iters, prim_hits: Volume integrator state
        - hit_collection, initial_contrib, initial_prim_indices, initial_prim_count: Hit data
)doc");
}
