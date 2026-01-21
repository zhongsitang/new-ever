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
// Input validation macros
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
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x)                                              \
  CHECK_INPUT(x);                                                        \
  CHECK_DEVICE(x);                                                       \
  CHECK_FLOAT(x);                                                        \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x)                                              \
  CHECK_INPUT(x);                                                        \
  CHECK_DEVICE(x);                                                       \
  CHECK_FLOAT(x);                                                        \
  TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

// =============================================================================
// Global AABB cache for acceleration structure
// =============================================================================
static OptixAabb* g_aabb_cache = nullptr;
static size_t g_aabb_cache_size = 0;

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/) {
}

// =============================================================================
// TracerContext - OptiX device context wrapper
// =============================================================================
class TracerContext {
public:
    OptixDeviceContext context = nullptr;
    uint device_id;

    explicit TracerContext(const torch::Device& device) : device_id(device.index()) {
        CUDA_CHECK(cudaSetDevice(device.index()));
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;

        CUcontext cu_ctx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
    }

    ~TracerContext() {
        OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context));
    }
};

// =============================================================================
// PrimitivesWrapper - GPU primitive collection wrapper
// =============================================================================
class PrimitivesWrapper {
public:
    Primitives data;
    torch::Device device;

    explicit PrimitivesWrapper(const torch::Device& device) : device(device) {}

    void add_primitives(const torch::Tensor& means,
                        const torch::Tensor& scales,
                        const torch::Tensor& quats,
                        const torch::Tensor& densities,
                        const torch::Tensor& sh_coeffs) {
        const int64_t num_prims = means.size(0);

        CHECK_FLOAT_DIM3(means);
        CHECK_FLOAT_DIM3(scales);
        CHECK_FLOAT_DIM4(quats);
        CHECK_FLOAT_DIM3(sh_coeffs);
        TORCH_CHECK(scales.size(0) == num_prims,
                    "scales must have the same number of primitives as means");
        TORCH_CHECK(quats.size(0) == num_prims,
                    "quats must have the same number of primitives as means");
        TORCH_CHECK(sh_coeffs.size(0) == num_prims,
                    "sh_coeffs must have the same number of primitives as means");
        TORCH_CHECK(densities.size(0) == num_prims,
                    "densities must have the same number of primitives as means");
        TORCH_CHECK(sh_coeffs.size(2) == 3,
                    "sh_coeffs must have 3 channels in last dimension (N, degree, 3)");

        data.feature_size = sh_coeffs.size(1);
        data.means = reinterpret_cast<float3*>(means.data_ptr());
        data.scales = reinterpret_cast<float3*>(scales.data_ptr());
        data.quats = reinterpret_cast<float4*>(quats.data_ptr());
        data.densities = reinterpret_cast<float*>(densities.data_ptr());
        data.features = reinterpret_cast<float*>(sh_coeffs.data_ptr());
        data.num_prims = num_prims;
        data.prev_alloc_size = g_aabb_cache_size;
        data.aabbs = g_aabb_cache;

        create_aabbs(data);

        g_aabb_cache = data.aabbs;
        g_aabb_cache_size = std::max(data.num_prims, g_aabb_cache_size);
    }

    void set_sh_coeffs(const torch::Tensor& sh_coeffs) {
        CHECK_FLOAT_DIM3(sh_coeffs);
        TORCH_CHECK(sh_coeffs.size(0) == static_cast<int64_t>(data.num_prims),
                    "sh_coeffs must have the same number of primitives");
        data.features = reinterpret_cast<float*>(sh_coeffs.data_ptr());
    }
};

// =============================================================================
// AccelStructWrapper - OptiX acceleration structure wrapper
// =============================================================================
class AccelStructWrapper {
public:
    GAS gas;

    AccelStructWrapper(const TracerContext& ctx,
                       const torch::Device& device,
                       const PrimitivesWrapper& prims,
                       bool enable_anyhit,
                       bool fast_build,
                       bool enable_rebuild)
        : gas(ctx.context, device.index(), prims.data, enable_anyhit, fast_build) {}
};

// =============================================================================
// IntegrationBuffer - Volume integration state buffer for backward pass
// =============================================================================
class IntegrationBuffer {
public:
    torch::Tensor states;
    torch::Tensor samples;
    torch::Tensor last_prim_idx;
    torch::Tensor prim_hit_count;
    torch::Tensor sample_count;

    size_t num_prims;
    size_t num_rays;
    size_t floats_per_state;
    torch::Device device;

    explicit IntegrationBuffer(torch::Device device)
        : num_prims(0), num_rays(0),
          floats_per_state(sizeof(IntegrationState) / sizeof(float)),
          device(device) {}

    IntegrationBuffer(size_t num_rays, size_t num_prims, torch::Device device)
        : num_prims(num_prims),
          floats_per_state(sizeof(IntegrationState) / sizeof(float)),
          device(device) {
        allocate(num_rays);
    }

    uint* sample_count_ptr() { return reinterpret_cast<uint*>(sample_count.data_ptr()); }
    uint* prim_hit_count_ptr() { return reinterpret_cast<uint*>(prim_hit_count.data_ptr()); }
    uint* last_prim_idx_ptr() { return reinterpret_cast<uint*>(last_prim_idx.data_ptr()); }
    float4* samples_ptr() { return reinterpret_cast<float4*>(samples.data_ptr()); }
    IntegrationState* states_ptr() { return reinterpret_cast<IntegrationState*>(states.data_ptr()); }

    torch::Tensor get_states() { return states; }
    torch::Tensor get_samples() { return samples; }
    torch::Tensor get_last_prim_idx() { return last_prim_idx; }
    torch::Tensor get_sample_count() { return sample_count; }
    torch::Tensor get_prim_hit_count() { return prim_hit_count; }

    void allocate(size_t num_rays) {
        states = torch::zeros({static_cast<long>(num_rays), static_cast<long>(floats_per_state)},
                              torch::device(device).dtype(torch::kFloat32));
        samples = torch::zeros({static_cast<long>(num_rays), 4},
                               torch::device(device).dtype(torch::kFloat32));
        last_prim_idx = torch::zeros({static_cast<long>(num_rays)},
                                     torch::device(device).dtype(torch::kInt32));
        prim_hit_count = torch::zeros({static_cast<long>(num_prims)},
                                      torch::device(device).dtype(torch::kInt32));
        sample_count = torch::zeros({static_cast<long>(num_rays)},
                                    torch::device(device).dtype(torch::kInt32));
        this->num_rays = num_rays;
    }
};

// =============================================================================
// RayTracerWrapper - Forward ray tracing wrapper
// =============================================================================
class RayTracerWrapper {
public:
    Forward tracer;
    torch::Device device;
    size_t num_prims;
    uint sh_degree;

    RayTracerWrapper(const TracerContext& ctx,
                     const torch::Device& device,
                     const PrimitivesWrapper& prims,
                     bool enable_backward)
        : device(device),
          tracer(ctx.context, device.index(), prims.data, enable_backward),
          num_prims(prims.data.num_prims),
          sh_degree(sqrt(prims.data.feature_size) - 1) {}

    void update_primitives(const PrimitivesWrapper& prims) {
        tracer.reset_features(prims.data);
    }

    py::dict trace_rays(const AccelStructWrapper& accel,
                        const torch::Tensor& ray_origins,
                        const torch::Tensor& ray_directions,
                        float t_near,
                        float t_far,
                        size_t max_samples,
                        float max_prim_size) {
        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM3(ray_origins);
        CHECK_FLOAT_DIM3(ray_directions);

        const size_t num_rays = ray_origins.numel() / 3;

        torch::Tensor color = torch::zeros(
            {static_cast<long>(num_rays), 4},
            torch::device(device).dtype(torch::kFloat32));

        torch::Tensor prim_sequence = torch::zeros(
            {static_cast<long>(num_rays * max_samples)},
            torch::device(device).dtype(torch::kInt32));

        torch::Tensor initial_sample = torch::zeros(
            {static_cast<long>(num_rays), 4},
            torch::device(device).dtype(torch::kFloat32));

        torch::Tensor initial_hit_count = torch::zeros(
            {1}, torch::device(device).dtype(torch::kInt32));

        torch::Tensor initial_hit_prims = torch::zeros(
            {static_cast<long>(num_prims)},
            torch::device(device).dtype(torch::kInt32));

        IntegrationBuffer buffer(num_rays, num_prims, device);

        tracer.trace_rays(
            accel.gas.gas_handle, num_rays,
            reinterpret_cast<float3*>(ray_origins.data_ptr()),
            reinterpret_cast<float3*>(ray_directions.data_ptr()),
            reinterpret_cast<void*>(color.data_ptr()),
            sh_degree, t_near, t_far,
            reinterpret_cast<float4*>(initial_sample.data_ptr()),
            nullptr,
            max_samples, max_prim_size,
            buffer.sample_count_ptr(),
            buffer.last_prim_idx_ptr(),
            buffer.prim_hit_count_ptr(),
            buffer.samples_ptr(),
            buffer.states_ptr(),
            reinterpret_cast<int*>(prim_sequence.data_ptr()),
            reinterpret_cast<int*>(initial_hit_count.data_ptr()),
            reinterpret_cast<int*>(initial_hit_prims.data_ptr()));

        return py::dict(
            "color"_a = color,
            "saved"_a = buffer,
            "tri_collection"_a = prim_sequence,
            "initial_drgb"_a = initial_sample,
            "initial_touch_inds"_a = initial_hit_prims,
            "initial_touch_count"_a = initial_hit_count);
    }
};

// =============================================================================
// Python module definition
// =============================================================================
PYBIND11_MODULE(primtracer_core, m) {
    m.doc() = "PrimTracer: Primitive-based volume rendering with OptiX";

    py::class_<TracerContext>(m, "OptixContext")
        .def(py::init<const torch::Device&>());

    py::class_<IntegrationBuffer>(m, "SavedForBackward")
        .def_property_readonly("states", &IntegrationBuffer::get_states)
        .def_property_readonly("diracs", &IntegrationBuffer::get_samples)
        .def_property_readonly("touch_count", &IntegrationBuffer::get_prim_hit_count)
        .def_property_readonly("iters", &IntegrationBuffer::get_sample_count)
        .def_property_readonly("faces", &IntegrationBuffer::get_last_prim_idx);

    py::class_<PrimitivesWrapper>(m, "Primitives")
        .def(py::init<const torch::Device&>())
        .def("add_primitives", &PrimitivesWrapper::add_primitives)
        .def("set_features", &PrimitivesWrapper::set_sh_coeffs);

    py::class_<AccelStructWrapper>(m, "GAS")
        .def(py::init<const TracerContext&, const torch::Device&,
                      const PrimitivesWrapper&, bool, bool, bool>());

    py::class_<RayTracerWrapper>(m, "Forward")
        .def(py::init<const TracerContext&, const torch::Device&,
                      const PrimitivesWrapper&, bool>())
        .def("trace_rays", &RayTracerWrapper::trace_rays)
        .def("update_model", &RayTracerWrapper::update_primitives);
}
