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

#include "ray_tracer.h"

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Tensor validation macros
// =============================================================================

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x, dev) \
    TORCH_CHECK(x.device() == dev, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x, dev, dim) \
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DEVICE(x, dev); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == dim, #x " must have last dimension " #dim)

// Helper for casting tensor data pointer to custom types
template<typename T>
T* data_ptr(const torch::Tensor& t) { return static_cast<T*>(t.data_ptr()); }

/// Pad (N, 3) tensor to (N, 4) with zeros in the last column.
/// This is required for ABI stability: float4 has 16-byte stride vs float3's 12-byte.
inline torch::Tensor pad_to_float4(const torch::Tensor& t) {
    if (t.size(-1) == 4) return t;  // Already padded
    TORCH_CHECK(t.size(-1) == 3, "Expected last dimension 3 or 4");
    return torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, 1}));
}

// =============================================================================
// PyRayTracer - Python wrapper for RayTracer with tensor lifetime management
// =============================================================================

class PyRayTracer {
public:
    explicit PyRayTracer(int device_index)
        : tracer_(std::make_unique<RayTracer>(device_index))
        , device_(torch::Device(torch::kCUDA, device_index))
    {}

    void update_primitives(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& quats,
        const torch::Tensor& densities,
        const torch::Tensor& features)
    {
        // Validate inputs (allow either 3 or 4 for means/scales)
        CHECK_CUDA(means); CHECK_CONTIGUOUS(means); CHECK_DEVICE(means, device_); CHECK_FLOAT(means);
        CHECK_CUDA(scales); CHECK_CONTIGUOUS(scales); CHECK_DEVICE(scales, device_); CHECK_FLOAT(scales);
        CHECK_INPUT(quats, device_, 4);
        CHECK_INPUT(features, device_, 3);
        CHECK_CUDA(densities); CHECK_CONTIGUOUS(densities);
        CHECK_DEVICE(densities, device_); CHECK_FLOAT(densities);
        TORCH_CHECK(means.size(-1) == 3 || means.size(-1) == 4, "means must have last dimension 3 or 4");
        TORCH_CHECK(scales.size(-1) == 3 || scales.size(-1) == 4, "scales must have last dimension 3 or 4");

        const uint64_t num_prims = means.size(0);
        const uint64_t feature_size = features.size(1);

        TORCH_CHECK(scales.size(0) == (long)num_prims, "scales must match means count");
        TORCH_CHECK(quats.size(0) == (long)num_prims, "quats must match means count");
        TORCH_CHECK(densities.size(0) == (long)num_prims, "densities must match means count");
        TORCH_CHECK(features.size(0) == (long)num_prims, "features must match means count");

        // Pad to float4 for ABI stability (16-byte stride)
        means_ = pad_to_float4(means).contiguous();
        scales_ = pad_to_float4(scales).contiguous();
        quats_ = quats;
        densities_ = densities;
        features_ = features;

        // Update tracer (cast float4* to float3* - the stride is what matters)
        Primitives prims = {
            .means = data_ptr<float3>(means_),
            .scales = data_ptr<float3>(scales_),
            .quats = data_ptr<float4>(quats_),
            .densities = data_ptr<float>(densities_),
            .num_prims = num_prims,
            .features = data_ptr<float>(features_),
            .feature_size = feature_size,
        };
        tracer_->update_primitives(prims);
    }

    py::dict trace_rays(
        const torch::Tensor& ray_origins,
        const torch::Tensor& ray_directions,
        float tmin,
        const torch::Tensor& tmax,
        uint32_t max_iters)
    {
        torch::AutoGradMode enable_grad(false);

        if (!tracer_->has_primitives()) {
            throw std::runtime_error("Must call update_primitives() before trace_rays()");
        }

        // Validate ray inputs (allow either 3 or 4)
        CHECK_CUDA(ray_origins); CHECK_CONTIGUOUS(ray_origins);
        CHECK_DEVICE(ray_origins, device_); CHECK_FLOAT(ray_origins);
        CHECK_CUDA(ray_directions); CHECK_CONTIGUOUS(ray_directions);
        CHECK_DEVICE(ray_directions, device_); CHECK_FLOAT(ray_directions);
        CHECK_CUDA(tmax); CHECK_CONTIGUOUS(tmax);
        CHECK_DEVICE(tmax, device_); CHECK_FLOAT(tmax);
        TORCH_CHECK(ray_origins.size(-1) == 3 || ray_origins.size(-1) == 4,
                    "ray_origins must have last dimension 3 or 4");
        TORCH_CHECK(ray_directions.size(-1) == 3 || ray_directions.size(-1) == 4,
                    "ray_directions must have last dimension 3 or 4");

        const uint64_t num_rays = ray_origins.size(0);
        const uint64_t num_prims = tracer_->num_prims();
        const uint64_t feature_size = features_.size(1);
        const auto sh_degree = static_cast<uint32_t>(sqrt(feature_size)) - 1;

        TORCH_CHECK(ray_directions.size(0) == (long)num_rays, "ray_directions must match ray_origins count");
        TORCH_CHECK(tmax.numel() == (long)num_rays, "tmax must have one value per ray");

        // Pad ray data to float4 for ABI stability
        auto ray_origins_f4 = pad_to_float4(ray_origins).contiguous();
        auto ray_directions_f4 = pad_to_float4(ray_directions).contiguous();

        // Allocate output tensors
        auto opts_f = torch::device(device_).dtype(torch::kFloat32);
        auto opts_i = torch::device(device_).dtype(torch::kInt32);

        torch::Tensor color = torch::zeros({(long)num_rays, 4}, opts_f);
        torch::Tensor depth = torch::zeros({(long)num_rays}, opts_f);

        // Allocate backward state tensors
        constexpr uint64_t state_floats = sizeof(IntegratorState) / sizeof(float);
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
            .iters = data_ptr<uint32_t>(iters),
            .prim_hits = data_ptr<uint32_t>(prim_hits),
            .hit_collection = data_ptr<int>(hit_collection),
            .initial_contrib = data_ptr<float4>(initial_contrib),
            .initial_prim_indices = data_ptr<int>(initial_prim_indices),
            .initial_prim_count = data_ptr<int>(initial_prim_count),
        };

        // Trace rays
        tracer_->trace_rays(
            num_rays,
            data_ptr<float4>(ray_origins_f4),
            data_ptr<float4>(ray_directions_f4),
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

    bool has_primitives() const { return tracer_->has_primitives(); }
    uint64_t num_prims() const { return tracer_->num_prims(); }
    int device_index() const { return tracer_->device_index(); }

private:
    std::unique_ptr<RayTracer> tracer_;
    torch::Device device_;

    // Store tensor references to prevent Python GC from releasing underlying data
    torch::Tensor means_;
    torch::Tensor scales_;
    torch::Tensor quats_;
    torch::Tensor densities_;
    torch::Tensor features_;
};

// =============================================================================
// Python module definition
// =============================================================================

PYBIND11_MODULE(ellipsoid_tracer, m) {
    m.doc() = "Differentiable volume rendering for ellipsoid primitives";

    py::class_<PyRayTracer>(m, "RayTracer")
        .def(py::init<int>(), py::arg("device_index") = 0,
             R"doc(
Create a new RayTracer instance.

The OptiX pipeline is compiled once during construction. Subsequent calls to
update_primitives() and trace_rays() reuse the compiled pipeline.

Args:
    device_index: CUDA device index (default: 0)
)doc")
        .def("update_primitives", &PyRayTracer::update_primitives,
             py::arg("means"),
             py::arg("scales"),
             py::arg("quats"),
             py::arg("densities"),
             py::arg("features"),
             R"doc(
Update primitive data and rebuild acceleration structure.

Call this when primitive data changes (count or values). The acceleration
structure buffers are reused when possible to minimize allocations.

Args:
    means: Primitive centers, shape (N, 3)
    scales: Primitive scales, shape (N, 3)
    quats: Primitive rotations as quaternions (w,x,y,z), shape (N, 4)
    densities: Primitive densities, shape (N,)
    features: SH features, shape (N, C, 3) where C is number of SH coefficients
)doc")
        .def("trace_rays", &PyRayTracer::trace_rays,
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("tmin"),
             py::arg("tmax"),
             py::arg("max_iters"),
             R"doc(
Trace rays through the scene.

Requires update_primitives() to be called first.

Args:
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
)doc")
        .def("has_primitives", &PyRayTracer::has_primitives,
             "Returns True if primitives have been set via update_primitives()")
        .def("num_prims", &PyRayTracer::num_prims,
             "Returns the number of primitives")
        .def("device_index", &PyRayTracer::device_index,
             "Returns the CUDA device index");
}
