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

#include "ray_pipeline.h"
#include "accel_structure.h"
#include "cuda_kernels.h"
#include "optix_error.h"

namespace py = pybind11;
using namespace pybind11::literals;

// =============================================================================
// Input Validation Macros
// =============================================================================

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x)                                                        \
  TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)                                                         \
  TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x)                                                    \
  CHECK_INPUT(x);                                                              \
  CHECK_DEVICE(x);                                                             \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x)                                                    \
  CHECK_INPUT(x);                                                              \
  CHECK_DEVICE(x);                                                             \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")
#define CHECK_FLOAT_DIM4_CPU(x)                                                \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")
#define CHECK_FLOAT_DIM3_CPU(x)                                                \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

// =============================================================================
// OptiX Context Logging
// =============================================================================

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void * /*cbdata */) {
}

OptixAabb *D_AABBS = 0;
size_t NUM_AABBS = 0;

// =============================================================================
// OptiX Context Wrapper
// =============================================================================

struct fesOptixContext {
public:
  OptixDeviceContext context = nullptr;
  uint device;
  fesOptixContext(const torch::Device &device) : device(device.index()) {
    CUDA_CHECK(cudaSetDevice(device.index()));
    {
      // Initialize CUDA
      CUDA_CHECK(cudaFree(0));
      // Initialize the OptiX API, loading all API entry points
      OPTIX_CHECK(optixInit());
      // Specify context options
      OptixDeviceContextOptions options = {};
      options.logCallbackFunction = &context_log_cb;
      options.logCallbackLevel = 4;
      // Associate a CUDA context (and therefore a specific GPU) with this
      // device context
      CUcontext cuCtx = 0; // zero means take the current context
      OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }
  }
  ~fesOptixContext() { OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context)); }
};

// =============================================================================
// Primitives Wrapper
// =============================================================================

struct fesPyPrimitives {
public:
  Primitives model;
  torch::Device device;
  fesPyPrimitives(const torch::Device &device) : device(device) {}
  void add_primitives(const torch::Tensor &means, const torch::Tensor &scales,
                      const torch::Tensor &quats,
                      const torch::Tensor &densities,
                      const torch::Tensor &colors) {
    const int64_t numPrimitives = means.size(0);
    CHECK_FLOAT_DIM3(means);
    CHECK_FLOAT_DIM3(scales);
    CHECK_FLOAT_DIM4(quats);
    CHECK_FLOAT_DIM3(colors);
    TORCH_CHECK(scales.size(0) == numPrimitives,
                "All inputs (scales) must have the same 0 dimension")
    TORCH_CHECK(quats.size(0) == numPrimitives,
                "All inputs (quats) must have the same 0 dimension")
    TORCH_CHECK(colors.size(0) == numPrimitives,
                "All inputs (colors) must have the same 0 dimension")
    TORCH_CHECK(densities.size(0) == numPrimitives,
                "All inputs (densities) must have the same 0 dimension")
    TORCH_CHECK(colors.size(2) == 3, "Features must have 3 channels. (N, d, 3)")
    model.feature_size = colors.size(1);

    model.means = reinterpret_cast<float3 *>(means.data_ptr());
    model.scales = reinterpret_cast<float3 *>(scales.data_ptr());
    model.quats = reinterpret_cast<float4 *>(quats.data_ptr());

    model.densities = reinterpret_cast<float *>(densities.data_ptr());
    model.features = reinterpret_cast<float *>(colors.data_ptr());
    model.num_prims = numPrimitives;
    model.prev_alloc_size = NUM_AABBS;
    model.aabbs = D_AABBS;
    build_primitive_aabbs(model);
    D_AABBS = model.aabbs;
    NUM_AABBS = std::max(model.num_prims, NUM_AABBS);
  }
  void set_features(const torch::Tensor &colors) {
    CHECK_FLOAT_DIM3(colors);
    TORCH_CHECK(colors.size(0) == model.num_prims,
                "All inputs (colors) must have the same 0 dimension");
    model.features = reinterpret_cast<float *>(colors.data_ptr());
  }
};

// =============================================================================
// Acceleration Structure Wrapper
// =============================================================================

struct fesPyGas {
public:
  GAS gas;
  fesPyGas(const fesOptixContext &context, const torch::Device &device,
        const fesPyPrimitives &model, const bool enable_anyhit,
        const bool fast_build, const bool enable_rebuild)
      : gas(context.context, device.index(), model.model, enable_anyhit,
            fast_build) {}
};

// =============================================================================
// Saved State for Backward Pass
// =============================================================================

struct fesSavedForBackward {
public:
  torch::Tensor states, deltas, surfaces, prim_hit_count, iters;
  size_t num_prims;
  size_t num_rays;
  size_t num_float_per_state;
  torch::Device device;

  fesSavedForBackward(torch::Device device)
      : num_prims(0), num_rays(0), num_float_per_state(sizeof(VolumeState) / sizeof(float)),
        device(device) {}

  fesSavedForBackward(size_t num_rays, size_t num_prims, torch::Device device)
      : num_prims(num_prims), num_float_per_state(sizeof(VolumeState) / sizeof(float)),
        device(device) {
    allocate(num_rays);
  }

  uint *iters_data_ptr() { return reinterpret_cast<uint *>(iters.data_ptr()); }
  uint *prim_hit_count_data_ptr() { return reinterpret_cast<uint *>(prim_hit_count.data_ptr()); }
  uint *surfaces_data_ptr() { return reinterpret_cast<uint *>(surfaces.data_ptr()); }
  float4 *deltas_data_ptr() { return reinterpret_cast<float4 *>(deltas.data_ptr()); }
  VolumeState *states_data_ptr() { return reinterpret_cast<VolumeState *>(states.data_ptr()); }

  torch::Tensor get_states() { return states; }
  torch::Tensor get_deltas() { return deltas; }
  torch::Tensor get_surfaces() { return surfaces; }
  torch::Tensor get_iters() { return iters; }
  torch::Tensor get_prim_hit_count() { return prim_hit_count; }

  void allocate(size_t num_rays) {
    states = torch::zeros({(long)num_rays, (long)num_float_per_state},
                          torch::device(device).dtype(torch::kFloat32));
    deltas = torch::zeros({(long)num_rays, 4},
                          torch::device(device).dtype(torch::kFloat32));
    surfaces = torch::zeros({(long)num_rays},
                         torch::device(device).dtype(torch::kInt32));
    prim_hit_count = torch::zeros({(long)num_prims},
                         torch::device(device).dtype(torch::kInt32));
    iters = torch::zeros({(long)num_rays},
                         torch::device(device).dtype(torch::kInt32));
    this->num_rays = num_rays;
  }
};

// =============================================================================
// Ray Tracing Pipeline Wrapper
// =============================================================================

struct fesPyRayPipeline {
public:
  RayPipeline pipeline;
  torch::Device device;
  size_t num_prims;
  uint sh_degree;

  fesPyRayPipeline(const fesOptixContext &context, const torch::Device &device,
            const fesPyPrimitives &model, const bool enable_backward)
      : device(device),
        pipeline(context.context, device.index(), model.model, enable_backward),
        num_prims(model.model.num_prims),
        sh_degree(sqrt(model.model.feature_size) - 1) {}

  void update_model(const fesPyPrimitives &model) {
    pipeline.reset_features(model.model);
  }

  py::dict trace_rays(const fesPyGas &gas, const torch::Tensor &ray_origins,
                      const torch::Tensor &ray_directions, float tmin,
                      float tmax, const size_t max_iters,
                      const float max_prim_size) {
    torch::AutoGradMode enable_grad(false);
    CHECK_FLOAT_DIM3(ray_origins);
    CHECK_FLOAT_DIM3(ray_directions);
    const size_t num_rays = ray_origins.numel() / 3;

    torch::Tensor color = torch::zeros({(long)num_rays, 4},
                         torch::device(device).dtype(torch::kFloat32));
    torch::Tensor hit_log = torch::zeros({(long)(num_rays * max_iters)},
                     torch::device(device).dtype(torch::kInt32));

    torch::Tensor initial_accumulated_drgb = torch::zeros(
        {(long)num_rays, 4},
        torch::device(device).dtype(torch::kFloat32));
    torch::Tensor initial_touch_count = torch::zeros(
        {1},
        torch::device(device).dtype(torch::kInt32));
    torch::Tensor initial_touch_inds = torch::zeros(
        {(long)num_prims},
        torch::device(device).dtype(torch::kInt32));

    fesSavedForBackward saved_for_backward(num_rays, num_prims, device);
    pipeline.trace_rays(gas.gas.gas_handle, num_rays,
                       reinterpret_cast<float3 *>(ray_origins.data_ptr()),
                       reinterpret_cast<float3 *>(ray_directions.data_ptr()),
                       reinterpret_cast<void *>(color.data_ptr()),
                       sh_degree, tmin, tmax,
                       reinterpret_cast<float4 *>(initial_accumulated_drgb.data_ptr()),
                       NULL,
                       max_iters, max_prim_size,
                       saved_for_backward.iters_data_ptr(),
                       saved_for_backward.surfaces_data_ptr(),
                       saved_for_backward.prim_hit_count_data_ptr(),
                       saved_for_backward.deltas_data_ptr(),
                       saved_for_backward.states_data_ptr(),
                       reinterpret_cast<int *>(hit_log.data_ptr()),
                       reinterpret_cast<int *>(initial_touch_count.data_ptr()),
                       reinterpret_cast<int *>(initial_touch_inds.data_ptr()));

    // Return dict with backward-compatible names for Python API
    return py::dict("color"_a = color,
                    "saved"_a = saved_for_backward,
                    "tri_collection"_a = hit_log,  // Legacy name
                    "initial_drgb"_a = initial_accumulated_drgb,  // Legacy name
                    "initial_touch_inds"_a = initial_touch_inds,
                    "initial_touch_count"_a = initial_touch_count);
  }
};

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(ellipsoid_tracer, m) {
  py::class_<fesOptixContext>(m, "OptixContext")
      .def(py::init<const torch::Device &>());

  py::class_<fesSavedForBackward>(m, "SavedForBackward")
      .def_property_readonly("states", &fesSavedForBackward::get_states)
      .def_property_readonly("diracs", &fesSavedForBackward::get_deltas)  // Legacy name
      .def_property_readonly("deltas", &fesSavedForBackward::get_deltas)  // New name
      .def_property_readonly("touch_count", &fesSavedForBackward::get_prim_hit_count)  // Legacy
      .def_property_readonly("prim_hit_count", &fesSavedForBackward::get_prim_hit_count)  // New
      .def_property_readonly("iters", &fesSavedForBackward::get_iters)
      .def_property_readonly("faces", &fesSavedForBackward::get_surfaces)  // Legacy name
      .def_property_readonly("surfaces", &fesSavedForBackward::get_surfaces);  // New name

  py::class_<fesPyPrimitives>(m, "Primitives")
      .def(py::init<const torch::Device &>())
      .def("add_primitives", &fesPyPrimitives::add_primitives)
      .def("set_features", &fesPyPrimitives::set_features);

  py::class_<fesPyGas>(m, "GAS").def(
      py::init<const fesOptixContext &, const torch::Device &,
               const fesPyPrimitives &, const bool, const bool, const bool>());

  py::class_<fesPyRayPipeline>(m, "Forward")
      .def(py::init<const fesOptixContext &, const torch::Device &,
                    const fesPyPrimitives &, const bool>())
      .def("trace_rays", &fesPyRayPipeline::trace_rays)
      .def("update_model", &fesPyRayPipeline::update_model);
}
