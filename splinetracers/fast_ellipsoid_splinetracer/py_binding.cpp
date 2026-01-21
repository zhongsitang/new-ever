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

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "exception.h"
#include "spline_tracer_autograd.h"

namespace py = pybind11;
using namespace pybind11::literals;

// Global AABB storage
OptixAabb *D_AABBS = 0;
size_t NUM_AABBS = 0;

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void * /*cbdata */) {
}

struct fesOptixContext {
public:
  OptixDeviceContext context = nullptr;
  uint device;
  fesOptixContext(const torch::Device &device) : device(device.index()) {
    CUDA_CHECK(cudaSetDevice(device.index()));
    {
      CUDA_CHECK(cudaFree(0));
      OPTIX_CHECK(optixInit());
      OptixDeviceContextOptions options = {};
      options.logCallbackFunction = &context_log_cb;
      options.logCallbackLevel = 4;
      CUcontext cuCtx = 0;
      OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }
  }
  ~fesOptixContext() { OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(context)); }
};

// Simplified trace_rays function with autograd support
torch::Tensor trace_rays(
    const fesOptixContext& ctx,
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& quat,
    const torch::Tensor& density,
    const torch::Tensor& color,
    const torch::Tensor& rayo,
    const torch::Tensor& rayd,
    float tmin,
    float tmax,
    float max_prim_size,
    const torch::Tensor& wcts,
    int max_iters
) {
    return trace_rays_autograd(
        ctx.context,
        static_cast<int8_t>(ctx.device),
        mean, scale, quat, density, color, rayo, rayd,
        tmin, tmax, max_prim_size, wcts, max_iters
    );
}

PYBIND11_MODULE(ellipsoid_splinetracer, m) {
  py::class_<fesOptixContext>(m, "OptixContext")
      .def(py::init<const torch::Device &>());

  m.def("trace_rays", &trace_rays,
        "Trace rays with automatic differentiation support",
        "ctx"_a, "mean"_a, "scale"_a, "quat"_a, "density"_a, "color"_a,
        "rayo"_a, "rayd"_a, "tmin"_a = 0.0f, "tmax"_a = 1000.0f,
        "max_prim_size"_a = 3.0f, "wcts"_a = torch::Tensor(), "max_iters"_a = 500);
}
