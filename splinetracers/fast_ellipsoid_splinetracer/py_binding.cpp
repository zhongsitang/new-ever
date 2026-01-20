// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#include <pybind11/pybind11.h>
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

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dim = 3")
#define CHECK_FLOAT_DIM4(x) CHECK_INPUT(x); CHECK_DEVICE(x); CHECK_FLOAT(x); \
    TORCH_CHECK(x.size(-1) == 4, #x " must have last dim = 4")

static void context_log_cb(unsigned int, const char*, const char*, void*) {}

static OptixAabb* g_aabbs = nullptr;
static size_t g_num_aabbs = 0;

struct fesOptixContext {
    OptixDeviceContext context = nullptr;
    uint32_t device;

    fesOptixContext(const torch::Device& dev) : device(dev.index()) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(nullptr));
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions opts = { .logCallbackFunction = &context_log_cb, .logCallbackLevel = 4 };
        OPTIX_CHECK(optixDeviceContextCreate(nullptr, &opts, &context));
    }
    ~fesOptixContext() { OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context)); }
};

struct fesPyPrimitives {
    Primitives model = {};
    torch::Device device;

    fesPyPrimitives(const torch::Device& dev) : device(dev) {}

    void add_primitives(const torch::Tensor& means, const torch::Tensor& scales,
                        const torch::Tensor& quats, const torch::Tensor& half_attribs,
                        const torch::Tensor& densities, const torch::Tensor& colors) {
        int64_t n = means.size(0);
        CHECK_FLOAT_DIM3(means); CHECK_FLOAT_DIM3(scales); CHECK_FLOAT_DIM4(quats); CHECK_FLOAT_DIM3(colors);
        TORCH_CHECK(scales.size(0) == n && quats.size(0) == n && colors.size(0) == n && densities.size(0) == n);
        TORCH_CHECK(colors.size(2) == 3, "Features must be (N, d, 3)");

        model.feature_size = colors.size(1);
        model.half_attribs = reinterpret_cast<half*>(half_attribs.data_ptr<torch::Half>());
        model.means = reinterpret_cast<float3*>(means.data_ptr());
        model.scales = reinterpret_cast<float3*>(scales.data_ptr());
        model.quats = reinterpret_cast<float4*>(quats.data_ptr());
        model.densities = reinterpret_cast<float*>(densities.data_ptr());
        model.features = reinterpret_cast<float*>(colors.data_ptr());
        model.num_prims = n;
        model.prev_alloc_size = g_num_aabbs;
        model.aabbs = g_aabbs;
        create_aabbs(model);
        g_aabbs = model.aabbs;
        g_num_aabbs = std::max(static_cast<size_t>(n), g_num_aabbs);
    }

    void set_features(const torch::Tensor& colors) {
        CHECK_FLOAT_DIM3(colors);
        TORCH_CHECK(colors.size(0) == static_cast<int64_t>(model.num_prims));
        model.features = reinterpret_cast<float*>(colors.data_ptr());
    }
};

struct fesPyGas {
    GAS gas;
    fesPyGas(const fesOptixContext& ctx, const torch::Device& dev, const fesPyPrimitives& model,
             bool enable_anyhit, bool fast_build, bool)
        : gas(ctx.context, dev.index(), model.model, enable_anyhit, fast_build) {}
};

struct fesSavedForBackward {
    torch::Tensor states, diracs, faces, touch_count, iters;
    size_t num_prims, num_rays;
    torch::Device device;

    fesSavedForBackward(torch::Device dev) : num_prims(0), num_rays(0), device(dev) {}
    fesSavedForBackward(size_t rays, size_t prims, torch::Device dev) : num_prims(prims), device(dev) {
        auto f = torch::device(device).dtype(torch::kFloat32);
        auto i = torch::device(device).dtype(torch::kInt32);
        states = torch::zeros({(long)rays, (long)(sizeof(SplineState)/sizeof(float))}, f);
        diracs = torch::zeros({(long)rays, 4}, f);
        faces = torch::zeros({(long)rays}, i);
        touch_count = torch::zeros({(long)prims}, i);
        iters = torch::zeros({(long)rays}, i);
        num_rays = rays;
    }

    uint32_t* iters_ptr() { return reinterpret_cast<uint32_t*>(iters.data_ptr()); }
    uint32_t* touch_count_ptr() { return reinterpret_cast<uint32_t*>(touch_count.data_ptr()); }
    uint32_t* faces_ptr() { return reinterpret_cast<uint32_t*>(faces.data_ptr()); }
    float4* diracs_ptr() { return reinterpret_cast<float4*>(diracs.data_ptr()); }
    SplineState* states_ptr() { return reinterpret_cast<SplineState*>(states.data_ptr()); }

    torch::Tensor get_states() { return states; }
    torch::Tensor get_diracs() { return diracs; }
    torch::Tensor get_faces() { return faces; }
    torch::Tensor get_iters() { return iters; }
    torch::Tensor get_touch_count() { return touch_count; }
};

struct fesPyForward {
    Forward forward;
    torch::Device device;
    size_t num_prims;
    uint32_t sh_degree;

    fesPyForward(const fesOptixContext& ctx, const torch::Device& dev,
                 const fesPyPrimitives& model, bool enable_backward)
        : device(dev), forward(ctx.context, dev.index(), model.model, enable_backward),
          num_prims(model.model.num_prims), sh_degree(sqrt(model.model.feature_size) - 1) {}

    void update_model(const fesPyPrimitives& model) { forward.reset_features(model.model); }

    py::dict trace_rays(const fesPyGas& gas, const torch::Tensor& ray_origins,
                        const torch::Tensor& ray_directions, float tmin, float tmax,
                        size_t max_iters, float max_prim_size) {
        torch::AutoGradMode no_grad(false);
        CHECK_FLOAT_DIM3(ray_origins); CHECK_FLOAT_DIM3(ray_directions);

        size_t n = ray_origins.numel() / 3;
        auto f = torch::device(device).dtype(torch::kFloat32);
        auto i = torch::device(device).dtype(torch::kInt32);

        auto color = torch::zeros({(long)n, 4}, f);
        auto tri_collection = torch::zeros({(long)(n * max_iters)}, i);
        auto initial_drgb = torch::zeros({(long)n, 4}, f);
        auto initial_touch_count = torch::zeros({1}, i);
        auto initial_touch_inds = torch::zeros({(long)num_prims}, i);

        fesSavedForBackward saved(n, num_prims, device);

        forward.trace_rays(
            gas.gas.gas_handle, n,
            reinterpret_cast<float3*>(ray_origins.data_ptr()),
            reinterpret_cast<float3*>(ray_directions.data_ptr()),
            color.data_ptr(), sh_degree, tmin, tmax,
            reinterpret_cast<float4*>(initial_drgb.data_ptr()),
            nullptr, max_iters, max_prim_size,
            saved.iters_ptr(), saved.faces_ptr(), saved.touch_count_ptr(),
            saved.diracs_ptr(), saved.states_ptr(),
            reinterpret_cast<int*>(tri_collection.data_ptr()),
            reinterpret_cast<int*>(initial_touch_count.data_ptr()),
            reinterpret_cast<int*>(initial_touch_inds.data_ptr()));

        return py::dict("color"_a=color, "saved"_a=saved, "tri_collection"_a=tri_collection,
                        "initial_drgb"_a=initial_drgb, "initial_touch_inds"_a=initial_touch_inds,
                        "initial_touch_count"_a=initial_touch_count);
    }
};

PYBIND11_MODULE(ellipsoid_splinetracer, m) {
    py::class_<fesOptixContext>(m, "OptixContext").def(py::init<const torch::Device&>());
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
        .def(py::init<const fesOptixContext&, const torch::Device&, const fesPyPrimitives&, bool, bool, bool>());
    py::class_<fesPyForward>(m, "Forward")
        .def(py::init<const fesOptixContext&, const torch::Device&, const fesPyPrimitives&, bool>())
        .def("trace_rays", &fesPyForward::trace_rays)
        .def("update_model", &fesPyForward::update_model);
}
