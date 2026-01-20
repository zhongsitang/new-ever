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

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Input Validation Macros
// =============================================================================

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) \
    TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) \
    CHECK_INPUT(x); \
    CHECK_DEVICE(x); \
    CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x) \
    CHECK_INPUT(x); \
    CHECK_DEVICE(x); \
    CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

// =============================================================================
// OptiX Context Logging
// =============================================================================

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/) {
    // Silence logging in production
}

// Global AABB storage for reuse
static OptixAabb* g_aabbs = nullptr;
static size_t g_num_aabbs = 0;

// =============================================================================
// OptiX Context Wrapper
// =============================================================================

struct fesOptixContext {
    OptixDeviceContext context = nullptr;
    uint32_t device;

    explicit fesOptixContext(const torch::Device& dev) : device(dev.index()) {
        CUDA_CHECK(cudaSetDevice(device));

        // Initialize CUDA
        CUDA_CHECK(cudaFree(nullptr));

        // Initialize OptiX API
        OPTIX_CHECK(optixInit());

        // Create device context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;

        CUcontext cu_ctx = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
    }

    ~fesOptixContext() {
        OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context));
    }

    // Non-copyable
    fesOptixContext(const fesOptixContext&) = delete;
    fesOptixContext& operator=(const fesOptixContext&) = delete;
};

// =============================================================================
// Primitives Wrapper
// =============================================================================

struct fesPyPrimitives {
    Primitives model = {};
    torch::Device device;

    explicit fesPyPrimitives(const torch::Device& dev) : device(dev) {}

    void add_primitives(const torch::Tensor& means,
                        const torch::Tensor& scales,
                        const torch::Tensor& quats,
                        const torch::Tensor& half_attribs,
                        const torch::Tensor& densities,
                        const torch::Tensor& colors) {
        const int64_t num_prims = means.size(0);

        CHECK_FLOAT_DIM3(means);
        CHECK_FLOAT_DIM3(scales);
        CHECK_FLOAT_DIM4(quats);
        CHECK_FLOAT_DIM3(colors);

        TORCH_CHECK(scales.size(0) == num_prims,
                    "scales must have the same batch dimension");
        TORCH_CHECK(quats.size(0) == num_prims,
                    "quats must have the same batch dimension");
        TORCH_CHECK(colors.size(0) == num_prims,
                    "colors must have the same batch dimension");
        TORCH_CHECK(densities.size(0) == num_prims,
                    "densities must have the same batch dimension");
        TORCH_CHECK(colors.size(2) == 3,
                    "Features must have 3 channels (N, d, 3)");

        model.feature_size = colors.size(1);
        model.half_attribs = reinterpret_cast<half*>(
            half_attribs.data_ptr<torch::Half>());
        model.means = reinterpret_cast<float3*>(means.data_ptr());
        model.scales = reinterpret_cast<float3*>(scales.data_ptr());
        model.quats = reinterpret_cast<float4*>(quats.data_ptr());
        model.densities = reinterpret_cast<float*>(densities.data_ptr());
        model.features = reinterpret_cast<float*>(colors.data_ptr());
        model.num_prims = num_prims;
        model.prev_alloc_size = g_num_aabbs;
        model.aabbs = g_aabbs;

        create_aabbs(model);

        g_aabbs = model.aabbs;
        g_num_aabbs = std::max(static_cast<size_t>(model.num_prims), g_num_aabbs);
    }

    void set_features(const torch::Tensor& colors) {
        CHECK_FLOAT_DIM3(colors);
        TORCH_CHECK(colors.size(0) == static_cast<int64_t>(model.num_prims),
                    "colors must have the same batch dimension");
        model.features = reinterpret_cast<float*>(colors.data_ptr());
    }
};

// =============================================================================
// GAS Wrapper
// =============================================================================

struct fesPyGas {
    GAS gas;

    fesPyGas(const fesOptixContext& ctx,
             const torch::Device& dev,
             const fesPyPrimitives& model,
             bool enable_anyhit,
             bool fast_build,
             bool /*enable_rebuild*/)
        : gas(ctx.context, dev.index(), model.model, enable_anyhit, fast_build) {}
};

// =============================================================================
// Saved State for Backward Pass
// =============================================================================

struct fesSavedForBackward {
    torch::Tensor states, diracs, faces, touch_count, iters;
    size_t num_prims = 0;
    size_t num_rays = 0;
    torch::Device device;

    static constexpr size_t num_float_per_state = sizeof(SplineState) / sizeof(float);

    explicit fesSavedForBackward(torch::Device dev) : device(dev) {}

    fesSavedForBackward(size_t rays, size_t prims, torch::Device dev)
        : num_prims(prims), device(dev) {
        allocate(rays);
    }

    void allocate(size_t rays) {
        auto opts = torch::device(device).dtype(torch::kFloat32);
        auto int_opts = torch::device(device).dtype(torch::kInt32);

        states = torch::zeros({static_cast<long>(rays),
                               static_cast<long>(num_float_per_state)}, opts);
        diracs = torch::zeros({static_cast<long>(rays), 4}, opts);
        faces = torch::zeros({static_cast<long>(rays)}, int_opts);
        touch_count = torch::zeros({static_cast<long>(num_prims)}, int_opts);
        iters = torch::zeros({static_cast<long>(rays)}, int_opts);
        num_rays = rays;
    }

    // Accessors
    uint32_t* iters_data_ptr() {
        return reinterpret_cast<uint32_t*>(iters.data_ptr());
    }
    uint32_t* touch_count_data_ptr() {
        return reinterpret_cast<uint32_t*>(touch_count.data_ptr());
    }
    uint32_t* faces_data_ptr() {
        return reinterpret_cast<uint32_t*>(faces.data_ptr());
    }
    float4* diracs_data_ptr() {
        return reinterpret_cast<float4*>(diracs.data_ptr());
    }
    SplineState* states_data_ptr() {
        return reinterpret_cast<SplineState*>(states.data_ptr());
    }

    // Properties
    torch::Tensor get_states() const { return states; }
    torch::Tensor get_diracs() const { return diracs; }
    torch::Tensor get_faces() const { return faces; }
    torch::Tensor get_iters() const { return iters; }
    torch::Tensor get_touch_count() const { return touch_count; }
};

// =============================================================================
// Forward Pass Wrapper
// =============================================================================

struct fesPyForward {
    optix_pipeline::Forward forward;
    torch::Device device;
    size_t num_prims;
    uint32_t sh_degree;

    fesPyForward(const fesOptixContext& ctx,
                 const torch::Device& dev,
                 const fesPyPrimitives& model,
                 bool enable_backward)
        : device(dev)
        , forward(ctx.context, dev.index(), model.model, enable_backward)
        , num_prims(model.model.num_prims)
        , sh_degree(static_cast<uint32_t>(std::sqrt(model.model.feature_size)) - 1) {}

    void update_model(const fesPyPrimitives& model) {
        forward.reset_features(model.model);
    }

    py::dict trace_rays(const fesPyGas& gas,
                        const torch::Tensor& ray_origins,
                        const torch::Tensor& ray_directions,
                        float tmin, float tmax,
                        size_t max_iters,
                        float max_prim_size) {
        torch::AutoGradMode enable_grad(false);

        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);

        const size_t num_rays = ray_origins.numel() / 3;
        auto opts = torch::device(device).dtype(torch::kFloat32);
        auto int_opts = torch::device(device).dtype(torch::kInt32);

        // Allocate output tensors
        auto color = torch::zeros({static_cast<long>(num_rays), 4}, opts);
        auto tri_collection = torch::zeros(
            {static_cast<long>(num_rays * max_iters)}, int_opts);
        auto initial_drgb = torch::zeros({static_cast<long>(num_rays), 4}, opts);
        auto initial_touch_count = torch::zeros({1}, int_opts);
        auto initial_touch_inds = torch::zeros(
            {static_cast<long>(num_prims)}, int_opts);

        fesSavedForBackward saved(num_rays, num_prims, device);

        // Build launch configuration
        optix_pipeline::LaunchConfig config{
            .handle = gas.gas.gas_handle,
            .num_rays = num_rays,
            .ray_origins = reinterpret_cast<float3*>(ray_origins.data_ptr()),
            .ray_directions = reinterpret_cast<float3*>(ray_directions.data_ptr()),
            .image_out = color.data_ptr(),
            .sh_degree = sh_degree,
            .tmin = tmin,
            .tmax = tmax,
            .initial_drgb = reinterpret_cast<float4*>(initial_drgb.data_ptr()),
            .camera = nullptr,
            .max_iters = max_iters,
            .max_prim_size = max_prim_size,
            .iters = saved.iters_data_ptr(),
            .last_face = saved.faces_data_ptr(),
            .touch_count = saved.touch_count_data_ptr(),
            .last_dirac = saved.diracs_data_ptr(),
            .last_state = saved.states_data_ptr(),
            .tri_collection = reinterpret_cast<int*>(tri_collection.data_ptr()),
            .d_touch_count = reinterpret_cast<int*>(initial_touch_count.data_ptr()),
            .d_touch_inds = reinterpret_cast<int*>(initial_touch_inds.data_ptr())
        };

        forward.trace_rays(config);

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
    m.doc() = "Spline-based ellipsoid volume rendering with OptiX 9.1";

    py::class_<fesOptixContext>(m, "OptixContext")
        .def(py::init<const torch::Device&>());

    py::class_<fesSavedForBackward>(m, "SavedForBackward")
        .def_property_readonly("states", &fesSavedForBackward::get_states)
        .def_property_readonly("diracs", &fesSavedForBackward::get_diracs)
        .def_property_readonly("touch_count", &fesSavedForBackward::get_touch_count)
        .def_property_readonly("iters", &fesSavedForBackward::get_iters)
        .def_property_readonly("faces", &fesSavedForBackward::get_faces);

    py::class_<fesPyPrimitives>(m, "Primitives")
        .def(py::init<const torch::Device&>())
        .def("add_primitives", &fesPyPrimitives::add_primitives)
        .def("set_features", &fesPyPrimitives::set_features);

    py::class_<fesPyGas>(m, "GAS")
        .def(py::init<const fesOptixContext&, const torch::Device&,
                      const fesPyPrimitives&, bool, bool, bool>());

    py::class_<fesPyForward>(m, "Forward")
        .def(py::init<const fesOptixContext&, const torch::Device&,
                      const fesPyPrimitives&, bool>())
        .def("trace_rays", &fesPyForward::trace_rays)
        .def("update_model", &fesPyForward::update_model);
}
