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
#include <unordered_map>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "ray_pipeline.h"
#include "accel_structure.h"
#include "cuda_kernels.h"
#include "optix_error.h"

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Tensor validation macros
// =============================================================================

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x, device)                                                \
  TORCH_CHECK(x.device() == device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)                                                         \
  TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_FLOAT_DIM(x, device, dim)                                        \
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DEVICE(x, device); CHECK_FLOAT(x); \
  TORCH_CHECK(x.size(-1) == dim, #x " must have last dimension with size " #dim)

// =============================================================================
// Global OptiX context management (lazy initialization per device)
// =============================================================================

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void * /*cbdata */) {}

static std::unordered_map<int, OptixDeviceContext> g_optix_contexts;
static OptixAabb *g_aabbs = nullptr;
static size_t g_num_aabbs = 0;

static OptixDeviceContext get_optix_context(int device_index) {
    auto it = g_optix_contexts.find(device_index);
    if (it != g_optix_contexts.end()) {
        return it->second;
    }

    CUDA_CHECK(cudaSetDevice(device_index));
    CUDA_CHECK(cudaFree(0));  // Initialize CUDA
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

// =============================================================================
// Internal helper: Build primitives and compute AABBs
// =============================================================================

static Primitives build_primitives(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& densities,
    const torch::Tensor& features)
{
    Primitives model;
    model.means = reinterpret_cast<float3*>(means.data_ptr());
    model.scales = reinterpret_cast<float3*>(scales.data_ptr());
    model.quats = reinterpret_cast<float4*>(quats.data_ptr());
    model.densities = reinterpret_cast<float*>(densities.data_ptr());
    model.features = reinterpret_cast<float*>(features.data_ptr());
    model.num_prims = means.size(0);
    model.feature_size = features.size(1);
    model.prev_alloc_size = g_num_aabbs;
    model.aabbs = g_aabbs;

    build_primitive_aabbs(model);

    g_aabbs = model.aabbs;
    g_num_aabbs = std::max(model.num_prims, g_num_aabbs);

    return model;
}

// =============================================================================
// SavedState: Tensors needed for backward pass
// =============================================================================

struct SavedState {
    torch::Tensor states;
    torch::Tensor delta_contribs;
    torch::Tensor iters;
    torch::Tensor prim_hits;
    torch::Device device;

    SavedState(size_t num_rays, size_t num_prims, torch::Device dev)
        : device(dev)
    {
        constexpr size_t state_floats = sizeof(IntegratorState) / sizeof(float);
        states = torch::zeros({(long)num_rays, (long)state_floats},
                              torch::device(dev).dtype(torch::kFloat32));
        delta_contribs = torch::zeros({(long)num_rays, 4},
                                      torch::device(dev).dtype(torch::kFloat32));
        iters = torch::zeros({(long)num_rays},
                             torch::device(dev).dtype(torch::kInt32));
        prim_hits = torch::zeros({(long)num_prims},
                                 torch::device(dev).dtype(torch::kInt32));
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

/// Trace rays through ellipsoid primitives using volume rendering.
///
/// This is the main entry point that handles all internal resource management
/// (OptiX context, acceleration structure, pipeline) automatically.
///
/// Args:
///     means: Primitive centers (N, 3)
///     scales: Primitive scales (N, 3)
///     quats: Primitive rotations as quaternions (N, 4)
///     densities: Primitive densities (N,)
///     features: SH features (N, C, 3)
///     ray_origins: Ray origins (M, 3)
///     ray_directions: Ray directions (M, 3)
///     tmin: Minimum ray parameter
///     tmax: Maximum ray parameter per ray (M,)
///     max_iters: Maximum hit iterations per ray
///
/// Returns:
///     dict with keys:
///         color: RGBA output (M, 4)
///         depth: Depth output (M,)
///         hit_collection: Hit primitive indices for backward
///         initial_contrib: Initial contribution for rays starting inside primitives
///         initial_prim_indices: Indices of primitives containing ray origins
///         initial_prim_count: Number of such primitives
///         saved: SavedState object for backward pass
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
    CHECK_FLOAT_DIM(means, device, 3);
    CHECK_FLOAT_DIM(scales, device, 3);
    CHECK_FLOAT_DIM(quats, device, 4);
    CHECK_FLOAT_DIM(features, device, 3);
    CHECK_FLOAT_DIM(ray_origins, device, 3);
    CHECK_FLOAT_DIM(ray_directions, device, 3);
    CHECK_CUDA(densities); CHECK_CONTIGUOUS(densities);
    CHECK_DEVICE(densities, device); CHECK_FLOAT(densities);
    CHECK_CUDA(tmax); CHECK_CONTIGUOUS(tmax);
    CHECK_DEVICE(tmax, device); CHECK_FLOAT(tmax);

    const size_t num_prims = means.size(0);
    const size_t num_rays = ray_origins.size(0);
    const uint sh_degree = static_cast<uint>(sqrt(features.size(1))) - 1;

    TORCH_CHECK(scales.size(0) == (long)num_prims, "scales must match means count");
    TORCH_CHECK(quats.size(0) == (long)num_prims, "quats must match means count");
    TORCH_CHECK(densities.size(0) == (long)num_prims, "densities must match means count");
    TORCH_CHECK(features.size(0) == (long)num_prims, "features must match means count");
    TORCH_CHECK(tmax.numel() == (long)num_rays, "tmax must have one value per ray");

    // Build primitives and acceleration structure
    Primitives model = build_primitives(means, scales, quats, densities, features);

    OptixDeviceContext context = get_optix_context(device_index);
    GAS gas(context, device_index, model, /*enable_anyhit=*/true, /*fast_build=*/false);

    // Create pipeline
    RayPipeline pipeline(context, device_index, model, /*enable_backward=*/true);

    // Allocate outputs
    torch::Tensor color = torch::zeros({(long)num_rays, 4},
                                        torch::device(device).dtype(torch::kFloat32));
    torch::Tensor depth = torch::zeros({(long)num_rays},
                                        torch::device(device).dtype(torch::kFloat32));
    torch::Tensor hit_collection = torch::zeros({(long)(num_rays * max_iters)},
                                                 torch::device(device).dtype(torch::kInt32));
    torch::Tensor initial_contrib = torch::zeros({(long)num_rays, 4},
                                                  torch::device(device).dtype(torch::kFloat32));
    torch::Tensor initial_prim_count = torch::zeros({1},
                                                     torch::device(device).dtype(torch::kInt32));
    torch::Tensor initial_prim_indices = torch::zeros({(long)num_prims},
                                                       torch::device(device).dtype(torch::kInt32));

    // Allocate backward state
    SavedState saved(num_rays, num_prims, device);

    // Temporary buffer for last_prim (not exposed to Python)
    torch::Tensor last_prim = torch::zeros({(long)num_rays},
                                            torch::device(device).dtype(torch::kInt32));

    // Trace rays
    pipeline.trace_rays(
        gas.gas_handle,
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
        3.0f,     // max_prim_size (internal detail)
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
