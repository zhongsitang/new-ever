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

// =============================================================================
// SavedState: Tensors needed for backward pass
// =============================================================================

struct SavedState {
    torch::Tensor states;
    torch::Tensor delta_contribs;
    torch::Tensor iters;
    torch::Tensor prim_hits;

    SavedState(size_t num_rays, size_t num_prims, torch::Device device) {
        constexpr size_t state_floats = sizeof(IntegratorState) / sizeof(float);
        states = torch::zeros({(long)num_rays, (long)state_floats},
                              torch::device(device).dtype(torch::kFloat32));
        delta_contribs = torch::zeros({(long)num_rays, 4},
                                      torch::device(device).dtype(torch::kFloat32));
        iters = torch::zeros({(long)num_rays},
                             torch::device(device).dtype(torch::kInt32));
        prim_hits = torch::zeros({(long)num_prims},
                                 torch::device(device).dtype(torch::kInt32));
    }

    IntegratorState* states_ptr() {
        return reinterpret_cast<IntegratorState*>(states.data_ptr());
    }
    float4* delta_contribs_ptr() {
        return reinterpret_cast<float4*>(delta_contribs.data_ptr());
    }
    uint* iters_ptr() { return reinterpret_cast<uint*>(iters.data_ptr()); }
    uint* prim_hits_ptr() { return reinterpret_cast<uint*>(prim_hits.data_ptr()); }
};

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
    RayPipeline pipeline(
        device_index,
        reinterpret_cast<float*>(means.data_ptr()),
        reinterpret_cast<float*>(scales.data_ptr()),
        reinterpret_cast<float*>(quats.data_ptr()),
        reinterpret_cast<float*>(densities.data_ptr()),
        reinterpret_cast<float*>(features.data_ptr()),
        num_prims,
        feature_size
    );

    // Allocate outputs
    auto opts = torch::device(device).dtype(torch::kFloat32);
    auto opts_int = torch::device(device).dtype(torch::kInt32);

    torch::Tensor color = torch::zeros({(long)num_rays, 4}, opts);
    torch::Tensor depth = torch::zeros({(long)num_rays}, opts);
    torch::Tensor hit_collection = torch::zeros({(long)(num_rays * max_iters)}, opts_int);
    torch::Tensor initial_contrib = torch::zeros({(long)num_rays, 4}, opts);
    torch::Tensor initial_prim_count = torch::zeros({1}, opts_int);
    torch::Tensor initial_prim_indices = torch::zeros({(long)num_prims}, opts_int);
    torch::Tensor last_prim = torch::zeros({(long)num_rays}, opts_int);

    // Allocate backward state
    SavedState saved(num_rays, num_prims, device);

    // Trace rays
    pipeline.trace_rays(
        num_rays,
        reinterpret_cast<float3*>(ray_origins.data_ptr()),
        reinterpret_cast<float3*>(ray_directions.data_ptr()),
        reinterpret_cast<float4*>(color.data_ptr()),
        reinterpret_cast<float*>(depth.data_ptr()),
        sh_degree,
        tmin,
        reinterpret_cast<float*>(tmax.data_ptr()),
        reinterpret_cast<float4*>(initial_contrib.data_ptr()),
        nullptr,  // camera
        max_iters,
        3.0f,     // max_prim_size
        saved.iters_ptr(),
        reinterpret_cast<uint*>(last_prim.data_ptr()),
        saved.prim_hits_ptr(),
        saved.delta_contribs_ptr(),
        saved.states_ptr(),
        reinterpret_cast<int*>(hit_collection.data_ptr()),
        reinterpret_cast<int*>(initial_prim_count.data_ptr()),
        reinterpret_cast<int*>(initial_prim_indices.data_ptr())
    );

    return py::dict(
        "color"_a = color,
        "depth"_a = depth,
        "hit_collection"_a = hit_collection,
        "initial_contrib"_a = initial_contrib,
        "initial_prim_indices"_a = initial_prim_indices,
        "initial_prim_count"_a = initial_prim_count,
        "saved"_a = saved
    );
}

// =============================================================================
// Python module definition
// =============================================================================

PYBIND11_MODULE(ellipsoid_tracer, m) {
    m.doc() = "Differentiable volume rendering for ellipsoid primitives";

    // Expose SavedState for backward pass
    py::class_<SavedState>(m, "SavedState")
        .def_readonly("states", &SavedState::states)
        .def_readonly("delta_contribs", &SavedState::delta_contribs)
        .def_readonly("iters", &SavedState::iters)
        .def_readonly("prim_hits", &SavedState::prim_hits);

    // Main API
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
        - saved: State object for backward pass
        - hit_collection, initial_contrib, etc.: Internal data for gradients
)doc");
}
