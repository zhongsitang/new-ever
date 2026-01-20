// GaussianRT - Python Bindings
// pybind11 interface for PyTorch integration
// Apache License 2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <slangtorch.h>

#include "Device.h"
#include "AccelerationStructure.h"
#include "RayTracer.h"

namespace py = pybind11;
using namespace gaussianrt;

// Convert torch tensor to raw pointer
template<typename T>
T* tensor_ptr(torch::Tensor& t) {
    return t.data_ptr<T>();
}

// Get shader directory path
static std::string get_shader_dir() {
    // Try to find shaders relative to the module
    py::module_ os = py::module_::import("os");
    py::module_ pathlib = py::module_::import("pathlib");

    py::object file_path = py::cast(__FILE__);
    py::object parent = pathlib.attr("Path")(file_path).attr("parent").attr("parent");
    py::object shader_dir = parent / "shaders";

    return shader_dir.attr("__str__")().cast<std::string>();
}

// Static kernel module for backward pass
static slangtorch::Module* backward_kernels = nullptr;

static void ensure_backward_kernels() {
    if (backward_kernels == nullptr) {
        std::string shader_dir = get_shader_dir();
        std::string shader_path = shader_dir + "/backward.slang";

        std::vector<std::string> include_paths = {shader_dir};
        backward_kernels = new slangtorch::Module(
            slangtorch::loadModule(shader_path.c_str(), include_paths)
        );
    }
}

// Wrapper class for Python interface
class GaussianRTContext {
public:
    GaussianRTContext(int device_index = 0)
        : device_(Device::create(device_index)) {
        ensure_backward_kernels();
    }

    // Build acceleration structure from primitives
    void build_accel(
        torch::Tensor means,
        torch::Tensor scales,
        torch::Tensor quats,
        torch::Tensor densities) {

        TORCH_CHECK(means.is_cuda(), "means must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        TORCH_CHECK(quats.is_cuda(), "quats must be on CUDA");
        TORCH_CHECK(densities.is_cuda(), "densities must be on CUDA");

        size_t num_prims = means.size(0);

        // Compute AABBs from ellipsoid bounds
        std::vector<AABB> aabbs(num_prims);
        auto means_cpu = means.cpu();
        auto scales_cpu = scales.cpu();

        auto means_acc = means_cpu.accessor<float, 2>();
        auto scales_acc = scales_cpu.accessor<float, 2>();

        for (size_t i = 0; i < num_prims; i++) {
            float max_scale = std::max({
                std::abs(scales_acc[i][0]),
                std::abs(scales_acc[i][1]),
                std::abs(scales_acc[i][2])
            });
            // Conservative AABB
            aabbs[i].minX = means_acc[i][0] - max_scale;
            aabbs[i].minY = means_acc[i][1] - max_scale;
            aabbs[i].minZ = means_acc[i][2] - max_scale;
            aabbs[i].maxX = means_acc[i][0] + max_scale;
            aabbs[i].maxY = means_acc[i][1] + max_scale;
            aabbs[i].maxZ = means_acc[i][2] + max_scale;
        }

        AccelBuildOptions options;
        options.allow_compaction = true;
        options.prefer_fast_build = false;
        options.allow_anyhit = true;

        accel_ = AccelerationStructure::build(*device_, aabbs, options);
        num_prims_ = num_prims;
    }

    // Create ray tracer with shader
    void create_tracer(const std::string& shader_path) {
        RayTracerConfig config;
        config.shader_path = shader_path;
        config.max_recursion = 1;
        tracer_ = RayTracer::create(*device_, config);
    }

    // Set primitive data
    void set_primitives(
        torch::Tensor means,
        torch::Tensor scales,
        torch::Tensor quats,
        torch::Tensor densities,
        torch::Tensor features) {

        TORCH_CHECK(tracer_, "Tracer not created");

        tracer_->set_primitives(
            *device_,
            reinterpret_cast<const float3*>(means.data_ptr<float>()),
            means.size(0),
            reinterpret_cast<const float3*>(scales.data_ptr<float>()),
            reinterpret_cast<const float4*>(quats.data_ptr<float>()),
            densities.data_ptr<float>(),
            features.data_ptr<float>(),
            features.size(1) * 3  // feature_size
        );
    }

    // Forward pass - trace rays
    py::dict trace_rays(
        torch::Tensor ray_origins,
        torch::Tensor ray_directions,
        float tmin,
        float tmax,
        int max_iters,
        float max_prim_size,
        int sh_degree,
        bool save_for_backward) {

        TORCH_CHECK(accel_, "Acceleration structure not built");
        TORCH_CHECK(tracer_, "Tracer not created");
        TORCH_CHECK(ray_origins.is_cuda(), "ray_origins must be on CUDA");
        TORCH_CHECK(ray_directions.is_cuda(), "ray_directions must be on CUDA");

        size_t ray_count = ray_origins.size(0);

        // Create output tensors
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(ray_origins.device());
        auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(ray_origins.device());

        torch::Tensor color = torch::zeros({(int64_t)ray_count, 4}, options);
        torch::Tensor iters = torch::zeros({(int64_t)ray_count}, int_options);
        torch::Tensor touch_count = torch::zeros({(int64_t)num_prims_}, int_options);
        torch::Tensor states = torch::zeros({(int64_t)ray_count, 16}, options);
        torch::Tensor diracs = torch::zeros({(int64_t)ray_count, 4}, options);
        torch::Tensor initial_drgb = torch::zeros({(int64_t)ray_count, 4}, options);

        torch::Tensor tri_collection;
        if (save_for_backward) {
            tri_collection = torch::zeros({(int64_t)ray_count * max_iters}, int_options);
        }

        TraceParams params;
        params.sh_degree = sh_degree;
        params.max_iters = max_iters;
        params.tmin = tmin;
        params.tmax = tmax;
        params.max_prim_size = max_prim_size;

        RenderOutput output;
        output.image = reinterpret_cast<float4*>(color.data_ptr<float>());
        output.iteration_counts = reinterpret_cast<uint*>(iters.data_ptr<int>());
        output.touch_counts = reinterpret_cast<uint*>(touch_count.data_ptr<int>());
        output.states = reinterpret_cast<VolumeState*>(states.data_ptr<float>());

        tracer_->trace_rays(
            *device_,
            *accel_,
            reinterpret_cast<const float3*>(ray_origins.data_ptr<float>()),
            reinterpret_cast<const float3*>(ray_directions.data_ptr<float>()),
            ray_count,
            params,
            output
        );

        py::dict result;
        result["color"] = color;
        result["iters"] = iters;
        result["touch_count"] = touch_count;
        result["states"] = states;
        result["diracs"] = diracs;
        result["initial_drgb"] = initial_drgb;

        if (save_for_backward) {
            result["tri_collection"] = tri_collection;
        }

        return result;
    }

    // Backward pass
    py::dict backward(
        torch::Tensor dL_doutputs,
        torch::Tensor means,
        torch::Tensor scales,
        torch::Tensor quats,
        torch::Tensor densities,
        torch::Tensor features,
        torch::Tensor ray_origins,
        torch::Tensor ray_directions,
        torch::Tensor states,
        torch::Tensor diracs,
        torch::Tensor iters,
        torch::Tensor tri_collection,
        torch::Tensor initial_drgb,
        torch::Tensor wcts,
        float tmin,
        float tmax,
        float max_prim_size,
        int max_iters,
        int sh_degree) {

        size_t num_prims = means.size(0);
        size_t num_rays = ray_origins.size(0);

        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(means.device());
        auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(means.device());

        // Create gradient tensors
        torch::Tensor dL_dmeans = torch::zeros({(int64_t)num_prims, 3}, options);
        torch::Tensor dL_dscales = torch::zeros({(int64_t)num_prims, 3}, options);
        torch::Tensor dL_dquats = torch::zeros({(int64_t)num_prims, 4}, options);
        torch::Tensor dL_ddensities = torch::zeros({(int64_t)num_prims}, options);
        torch::Tensor dL_dfeatures = torch::zeros_like(features);
        torch::Tensor dL_dray_origins = torch::zeros({(int64_t)num_rays, 3}, options);
        torch::Tensor dL_dray_dirs = torch::zeros({(int64_t)num_rays, 3}, options);
        torch::Tensor dL_dmeans2D = torch::zeros({(int64_t)num_prims, 2}, options);
        torch::Tensor dL_dinitial_drgb = torch::zeros({(int64_t)num_rays, 4}, options);
        torch::Tensor touch_count = torch::zeros({(int64_t)num_prims}, int_options);

        // Skip if no iterations recorded
        if (iters.sum().item<int>() > 0) {
            // Build DualModel tuple for backward kernel
            auto dual_model = std::make_tuple(
                means.contiguous(),
                scales.contiguous(),
                quats.contiguous(),
                densities.contiguous(),
                features.contiguous(),
                dL_dmeans,
                dL_dscales,
                dL_dquats,
                dL_ddensities,
                dL_dfeatures,
                dL_dray_origins,
                dL_dray_dirs,
                dL_dmeans2D
            );

            // Launch backward kernel
            const int block_size = 16;
            const int grid_size = (num_rays + block_size - 1) / block_size;

            backward_kernels->operator()("backwards_kernel")
                .set("last_state", states.contiguous())
                .set("last_dirac", diracs.contiguous())
                .set("iters", iters.contiguous())
                .set("tri_collection", tri_collection.contiguous())
                .set("ray_origins", ray_origins.contiguous())
                .set("ray_directions", ray_directions.contiguous())
                .set("model", dual_model)
                .set("initial_drgb", initial_drgb.contiguous())
                .set("dL_dinital_drgb", dL_dinitial_drgb)
                .set("touch_count", touch_count)
                .set("dL_doutputs", dL_doutputs.contiguous())
                .set("wcts", wcts.contiguous())
                .set("tmin", tmin)
                .set("tmax", tmax)
                .set("max_prim_size", max_prim_size)
                .set("max_iters", (uint32_t)max_iters)
                .launchRaw(
                    dim3(block_size, 1, 1),
                    dim3(grid_size, 1, 1)
                );
        }

        py::dict result;
        result["dL_dmeans"] = dL_dmeans;
        result["dL_dscales"] = dL_dscales;
        result["dL_dquats"] = dL_dquats;
        result["dL_ddensities"] = dL_ddensities;
        result["dL_dfeatures"] = dL_dfeatures;
        result["dL_dray_origins"] = dL_dray_origins;
        result["dL_dray_dirs"] = dL_dray_dirs;
        result["dL_dmeans2D"] = dL_dmeans2D;
        result["dL_dinitial_drgb"] = dL_dinitial_drgb;
        result["touch_count"] = touch_count;

        return result;
    }

private:
    std::unique_ptr<Device> device_;
    std::unique_ptr<AccelerationStructure> accel_;
    std::unique_ptr<RayTracer> tracer_;
    size_t num_prims_ = 0;
};

PYBIND11_MODULE(gaussianrt_ext, m) {
    m.doc() = "GaussianRT - Hardware ray tracing for Gaussian volume rendering";

    py::class_<GaussianRTContext>(m, "Context")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def("build_accel", &GaussianRTContext::build_accel,
             "Build acceleration structure from primitives",
             py::arg("means"), py::arg("scales"), py::arg("quats"), py::arg("densities"))
        .def("create_tracer", &GaussianRTContext::create_tracer,
             "Create ray tracer with shader",
             py::arg("shader_path"))
        .def("set_primitives", &GaussianRTContext::set_primitives,
             "Set primitive data",
             py::arg("means"), py::arg("scales"), py::arg("quats"),
             py::arg("densities"), py::arg("features"))
        .def("trace_rays", &GaussianRTContext::trace_rays,
             "Trace rays through scene",
             py::arg("ray_origins"), py::arg("ray_directions"),
             py::arg("tmin"), py::arg("tmax"), py::arg("max_iters"),
             py::arg("max_prim_size"), py::arg("sh_degree"),
             py::arg("save_for_backward") = true)
        .def("backward", &GaussianRTContext::backward,
             "Backward pass for gradients",
             py::arg("dL_doutputs"), py::arg("means"), py::arg("scales"),
             py::arg("quats"), py::arg("densities"), py::arg("features"),
             py::arg("ray_origins"), py::arg("ray_directions"),
             py::arg("states"), py::arg("diracs"), py::arg("iters"),
             py::arg("tri_collection"), py::arg("initial_drgb"), py::arg("wcts"),
             py::arg("tmin"), py::arg("tmax"), py::arg("max_prim_size"),
             py::arg("max_iters"), py::arg("sh_degree"));
}
