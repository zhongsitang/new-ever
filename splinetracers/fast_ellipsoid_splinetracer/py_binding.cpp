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

#include "OptixPipeline.h"
#include "GAS.h"
#include "create_aabbs.h"
#include "exception.h"

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Input validation macros
// =============================================================================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x) CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

// =============================================================================
// Global AABB buffer (reused across calls)
// =============================================================================
static OptixAabb* g_d_aabbs = nullptr;
static size_t g_aabb_capacity = 0;

// =============================================================================
// OptiX Context Wrapper
// =============================================================================
class PyOptixContext {
public:
    OptixDeviceContext context = nullptr;
    int device_id;

    PyOptixContext(const torch::Device& device) : device_id(device.index()) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaFree(0));  // Initialize CUDA
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = nullptr;
        options.logCallbackLevel = 0;

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    ~PyOptixContext() {
        if (context) {
            OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context));
        }
    }
};

// =============================================================================
// Primitives Wrapper
// =============================================================================
class PyPrimitives {
public:
    Primitives model = {};
    torch::Device device;

    PyPrimitives(const torch::Device& device) : device(device) {}

    void add_primitives(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& quats,
        const torch::Tensor& half_attribs,
        const torch::Tensor& densities,
        const torch::Tensor& colors
    ) {
        const int64_t num_prims = means.size(0);
        CHECK_FLOAT_DIM3(means);
        CHECK_FLOAT_DIM3(scales);
        CHECK_FLOAT_DIM4(quats);
        CHECK_FLOAT_DIM3(colors);
        TORCH_CHECK(scales.size(0) == num_prims, "scales size mismatch");
        TORCH_CHECK(quats.size(0) == num_prims, "quats size mismatch");
        TORCH_CHECK(colors.size(0) == num_prims, "colors size mismatch");
        TORCH_CHECK(densities.size(0) == num_prims, "densities size mismatch");
        TORCH_CHECK(colors.size(2) == 3, "Features must have 3 channels (N, d, 3)");

        model.feature_size = colors.size(1);
        model.half_attribs = reinterpret_cast<__half*>(half_attribs.data_ptr<torch::Half>());
        model.means = reinterpret_cast<float3*>(means.data_ptr());
        model.scales = reinterpret_cast<float3*>(scales.data_ptr());
        model.quats = reinterpret_cast<float4*>(quats.data_ptr());
        model.densities = reinterpret_cast<float*>(densities.data_ptr());
        model.features = reinterpret_cast<float*>(colors.data_ptr());
        model.num_prims = num_prims;
        model.prev_alloc_size = g_aabb_capacity;
        model.aabbs = g_d_aabbs;

        create_aabbs(model);

        g_d_aabbs = model.aabbs;
        g_aabb_capacity = std::max(model.num_prims, g_aabb_capacity);
    }
};

// =============================================================================
// GAS Wrapper
// =============================================================================
class PyGAS {
public:
    GAS gas;

    PyGAS(
        const PyOptixContext& ctx,
        const torch::Device& device,
        const PyPrimitives& prims,
        bool enable_anyhit,
        bool fast_build,
        bool enable_rebuild
    ) : gas(ctx.context, device.index(), prims.model, enable_anyhit, fast_build) {}
};

// =============================================================================
// Saved state for backward pass
// =============================================================================
class PySavedForBackward {
public:
    torch::Tensor states, diracs, faces, touch_count, iters;
    size_t num_prims;
    size_t num_rays;
    torch::Device device;

    static constexpr size_t num_float_per_state = sizeof(SplineState) / sizeof(float);

    PySavedForBackward(torch::Device device)
        : num_prims(0), num_rays(0), device(device) {}

    PySavedForBackward(size_t num_rays, size_t num_prims, torch::Device device)
        : num_prims(num_prims), num_rays(num_rays), device(device) {
        allocate(num_rays);
    }

    void allocate(size_t num_rays) {
        this->num_rays = num_rays;
        states = torch::zeros({(long)num_rays, (long)num_float_per_state},
                              torch::device(device).dtype(torch::kFloat32));
        diracs = torch::zeros({(long)num_rays, 4},
                              torch::device(device).dtype(torch::kFloat32));
        faces = torch::zeros({(long)num_rays},
                             torch::device(device).dtype(torch::kInt32));
        touch_count = torch::zeros({(long)num_prims},
                                   torch::device(device).dtype(torch::kInt32));
        iters = torch::zeros({(long)num_rays},
                             torch::device(device).dtype(torch::kInt32));
    }

    // Accessors for raw pointers
    uint32_t* iters_ptr() { return reinterpret_cast<uint32_t*>(iters.data_ptr()); }
    uint32_t* touch_count_ptr() { return reinterpret_cast<uint32_t*>(touch_count.data_ptr()); }
    uint32_t* faces_ptr() { return reinterpret_cast<uint32_t*>(faces.data_ptr()); }
    float4* diracs_ptr() { return reinterpret_cast<float4*>(diracs.data_ptr()); }
    SplineState* states_ptr() { return reinterpret_cast<SplineState*>(states.data_ptr()); }
};

// =============================================================================
// Forward Wrapper - Uses new OptixPipeline class
// Keeps the same Python API as before for backward compatibility
// =============================================================================
class PyForward {
public:
    OptixPipeline pipeline;
    torch::Device device;
    size_t num_prims;
    uint32_t sh_degree;
    size_t feature_size;

    // Store primitive pointers (they point to PyTorch tensor data, kept alive by Python)
    Primitives cached_prims = {};

    PyForward(
        const PyOptixContext& ctx,
        const torch::Device& device,
        const PyPrimitives& prims,
        bool backward_mode
    ) : device(device),
        num_prims(prims.model.num_prims),
        sh_degree(static_cast<uint32_t>(sqrt(prims.model.feature_size)) - 1),
        feature_size(prims.model.feature_size),
        cached_prims(prims.model)
    {
        pipeline.init(ctx.context, device.index(), backward_mode);
    }

    // Update primitive features (for when colors change but structure doesn't)
    void update_model(const PyPrimitives& prims) {
        cached_prims.features = prims.model.features;
        cached_prims.densities = prims.model.densities;
    }

    // trace_rays with same signature as original Forward class
    py::dict trace_rays(
        const PyGAS& gas,
        const torch::Tensor& ray_origins,
        const torch::Tensor& ray_directions,
        float tmin,
        float tmax,
        size_t max_iters,
        float max_prim_size
    ) {
        torch::NoGradGuard no_grad;
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);

        const size_t num_rays = ray_origins.numel() / 3;

        // Allocate output tensors
        auto color = torch::zeros({(long)num_rays, 4},
                                  torch::device(device).dtype(torch::kFloat32));
        auto tri_collection = torch::zeros({(long)(num_rays * max_iters)},
                                           torch::device(device).dtype(torch::kInt32));
        auto initial_drgb = torch::zeros({(long)num_rays, 4},
                                         torch::device(device).dtype(torch::kFloat32));
        auto initial_touch_count = torch::zeros({1},
                                                torch::device(device).dtype(torch::kInt32));
        auto initial_touch_inds = torch::zeros({(long)num_prims},
                                               torch::device(device).dtype(torch::kInt32));

        PySavedForBackward saved(num_rays, num_prims, device);

        // Build LaunchParams
        LaunchParams params = {};

        // Output buffers
        params.image = {reinterpret_cast<float4*>(color.data_ptr()), num_rays};
        params.iters = {saved.iters_ptr(), num_rays};
        params.last_face = {saved.faces_ptr(), num_rays};
        params.touch_count = {saved.touch_count_ptr(), num_prims};
        params.last_dirac = {saved.diracs_ptr(), num_rays};
        params.last_state = {saved.states_ptr(), num_rays};
        params.tri_collection = {reinterpret_cast<int32_t*>(tri_collection.data_ptr()), num_rays * max_iters};

        // Input buffers
        params.ray_origins = {reinterpret_cast<float3*>(ray_origins.data_ptr()), num_rays};
        params.ray_directions = {reinterpret_cast<float3*>(ray_directions.data_ptr()), num_rays};

        // Camera (not used in ray mode)
        params.camera = {};

        // Primitive data from cached primitives
        params.half_attribs = {cached_prims.half_attribs, num_prims};
        params.means = {cached_prims.means, num_prims};
        params.scales = {cached_prims.scales, num_prims};
        params.quats = {cached_prims.quats, num_prims};
        params.densities = {cached_prims.densities, num_prims};
        params.features = {cached_prims.features, feature_size * num_prims};

        // Rendering parameters
        params.sh_degree = sh_degree;
        params.max_iters = max_iters;
        params.tmin = tmin;
        params.tmax = tmax;
        params.initial_drgb = {reinterpret_cast<float4*>(initial_drgb.data_ptr()), num_rays};
        params.max_prim_size = max_prim_size;

        // Acceleration structure
        params.handle = gas.gas.gas_handle;

        // Launch
        pipeline.launch(params, static_cast<uint32_t>(num_rays), 1, nullptr);

        return py::dict(
            "color"_a = color,
            "saved"_a = saved,
            "tri_collection"_a = tri_collection,
            "initial_drgb"_a = initial_drgb,
            "initial_touch_inds"_a = initial_touch_inds,
            "initial_touch_count"_a = initial_touch_count
        );
    }
};

// =============================================================================
// Python Module Definition
// =============================================================================
PYBIND11_MODULE(ellipsoid_splinetracer, m) {
    m.doc() = "Spline-based ellipsoid volume rendering with OptiX";

    py::class_<PyOptixContext>(m, "OptixContext")
        .def(py::init<const torch::Device&>());

    py::class_<PySavedForBackward>(m, "SavedForBackward")
        .def_property_readonly("states", [](PySavedForBackward& s) { return s.states; })
        .def_property_readonly("diracs", [](PySavedForBackward& s) { return s.diracs; })
        .def_property_readonly("touch_count", [](PySavedForBackward& s) { return s.touch_count; })
        .def_property_readonly("iters", [](PySavedForBackward& s) { return s.iters; })
        .def_property_readonly("faces", [](PySavedForBackward& s) { return s.faces; });

    py::class_<PyPrimitives>(m, "Primitives")
        .def(py::init<const torch::Device&>())
        .def("add_primitives", &PyPrimitives::add_primitives);

    py::class_<PyGAS>(m, "GAS")
        .def(py::init<const PyOptixContext&, const torch::Device&,
                      const PyPrimitives&, bool, bool, bool>());

    // Forward class - uses new OptixPipeline internally
    py::class_<PyForward>(m, "Forward")
        .def(py::init<const PyOptixContext&, const torch::Device&,
                      const PyPrimitives&, bool>())
        .def("trace_rays", &PyForward::trace_rays)
        .def("update_model", &PyForward::update_model);
}
