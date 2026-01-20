#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../src/volume_renderer.h"

namespace py = pybind11;
using namespace gaussian_rt;

// Check tensor properties
void check_tensor(const torch::Tensor& t, const std::string& name,
                  std::vector<int64_t> expected_shape = {},
                  bool check_cuda = true, bool check_contiguous = true) {
    if (check_cuda && !t.is_cuda()) {
        throw std::runtime_error(name + " must be a CUDA tensor");
    }
    if (check_contiguous && !t.is_contiguous()) {
        throw std::runtime_error(name + " must be contiguous");
    }
    if (!expected_shape.empty()) {
        auto shape = t.sizes();
        if (shape.size() != expected_shape.size()) {
            throw std::runtime_error(name + " has wrong number of dimensions");
        }
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (expected_shape[i] > 0 && shape[i] != expected_shape[i]) {
                throw std::runtime_error(name + " has wrong shape at dimension " + std::to_string(i));
            }
        }
    }
}

// Python wrapper class
class PyVolumeRenderer {
public:
    PyVolumeRenderer(int device_index = 0, const std::string& shader_dir = "") {
        if (!renderer_.initialize(device_index, shader_dir)) {
            throw std::runtime_error("Failed to initialize VolumeRenderer");
        }
    }

    void build_accel(
        torch::Tensor positions,
        torch::Tensor scales,
        torch::Tensor rotations,
        bool fast_build = false
    ) {
        check_tensor(positions, "positions", {-1, 3});
        check_tensor(scales, "scales", {-1, 3});
        check_tensor(rotations, "rotations", {-1, 4});

        uint32_t num_elements = positions.size(0);

        renderer_.build_accel(
            positions.data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            num_elements,
            fast_build
        );
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    forward(
        torch::Tensor positions,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor features,
        torch::Tensor ray_origins,
        torch::Tensor ray_directions,
        float t_min,
        float t_max,
        int max_samples,
        int sh_degree
    ) {
        // Validate inputs
        check_tensor(positions, "positions", {-1, 3});
        check_tensor(scales, "scales", {-1, 3});
        check_tensor(rotations, "rotations", {-1, 4});
        check_tensor(opacities, "opacities", {-1});
        check_tensor(features, "features");
        check_tensor(ray_origins, "ray_origins", {-1, 3});
        check_tensor(ray_directions, "ray_directions", {-1, 3});

        uint32_t num_elements = positions.size(0);
        uint32_t feature_dim = features.size(1);
        uint32_t num_rays = ray_origins.size(0);

        // Allocate outputs
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(ray_origins.device());
        auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(ray_origins.device());

        torch::Tensor colors = torch::zeros({num_rays, 4}, options);
        torch::Tensor states = torch::zeros({num_rays, 11}, options);  // RenderState size
        torch::Tensor last_points = torch::zeros({num_rays, 5}, options);  // ControlPoint size
        torch::Tensor sample_counts = torch::zeros({num_rays}, int_options);
        torch::Tensor sample_indices = torch::zeros({num_rays * max_samples}, int_options);
        torch::Tensor touch_counts = torch::zeros({num_elements}, int_options);

        // Call forward
        renderer_.forward(
            positions.data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            opacities.data_ptr<float>(),
            features.data_ptr<float>(),
            num_elements,
            feature_dim,
            ray_origins.data_ptr<float>(),
            ray_directions.data_ptr<float>(),
            num_rays,
            t_min, t_max, max_samples, sh_degree,
            colors.data_ptr<float>(),
            states.data_ptr<float>(),
            last_points.data_ptr<float>(),
            sample_counts.data_ptr<int>(),
            sample_indices.data_ptr<int>(),
            touch_counts.data_ptr<int>()
        );

        return std::make_tuple(colors, states, last_points,
                               sample_counts, sample_indices, touch_counts);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward(
        torch::Tensor final_states,
        torch::Tensor last_points,
        torch::Tensor sample_counts,
        torch::Tensor sample_indices,
        torch::Tensor positions,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor opacities,
        torch::Tensor features,
        torch::Tensor ray_origins,
        torch::Tensor ray_directions,
        torch::Tensor grad_colors,
        torch::Tensor grad_depths,
        torch::Tensor grad_distortions,
        float t_min,
        float t_max,
        int max_samples,
        int sh_degree
    ) {
        uint32_t num_elements = positions.size(0);
        uint32_t feature_dim = features.size(1);
        uint32_t num_rays = ray_origins.size(0);

        // Allocate gradient outputs
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(positions.device());

        torch::Tensor grad_positions = torch::zeros({num_elements, 3}, options);
        torch::Tensor grad_scales = torch::zeros({num_elements, 3}, options);
        torch::Tensor grad_rotations = torch::zeros({num_elements, 4}, options);
        torch::Tensor grad_opacities = torch::zeros({num_elements}, options);
        torch::Tensor grad_features = torch::zeros({num_elements, feature_dim}, options);
        torch::Tensor grad_ray_origins = torch::zeros({num_rays, 3}, options);
        torch::Tensor grad_ray_dirs = torch::zeros({num_rays, 3}, options);

        // Call backward
        renderer_.backward(
            final_states.data_ptr<float>(),
            last_points.data_ptr<float>(),
            sample_counts.data_ptr<int>(),
            sample_indices.data_ptr<int>(),
            positions.data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            opacities.data_ptr<float>(),
            features.data_ptr<float>(),
            num_elements,
            feature_dim,
            ray_origins.data_ptr<float>(),
            ray_directions.data_ptr<float>(),
            num_rays,
            grad_colors.data_ptr<float>(),
            grad_depths.data_ptr<float>(),
            grad_distortions.data_ptr<float>(),
            t_min, t_max, max_samples, sh_degree,
            grad_positions.data_ptr<float>(),
            grad_scales.data_ptr<float>(),
            grad_rotations.data_ptr<float>(),
            grad_opacities.data_ptr<float>(),
            grad_features.data_ptr<float>(),
            grad_ray_origins.data_ptr<float>(),
            grad_ray_dirs.data_ptr<float>()
        );

        return std::make_tuple(
            grad_positions, grad_scales, grad_rotations,
            grad_opacities, grad_features, grad_ray_origins, grad_ray_dirs
        );
    }

private:
    VolumeRenderer renderer_;
};

PYBIND11_MODULE(_gaussian_rt, m) {
    m.doc() = "GaussianRT: Differentiable Volume Renderer";

    py::class_<PyVolumeRenderer>(m, "VolumeRenderer")
        .def(py::init<int, const std::string&>(),
             py::arg("device_index") = 0,
             py::arg("shader_dir") = "")
        .def("build_accel", &PyVolumeRenderer::build_accel,
             py::arg("positions"),
             py::arg("scales"),
             py::arg("rotations"),
             py::arg("fast_build") = false,
             "Build acceleration structure from element data")
        .def("forward", &PyVolumeRenderer::forward,
             py::arg("positions"),
             py::arg("scales"),
             py::arg("rotations"),
             py::arg("opacities"),
             py::arg("features"),
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("t_min"),
             py::arg("t_max"),
             py::arg("max_samples"),
             py::arg("sh_degree"),
             "Forward pass: trace rays and compute colors")
        .def("backward", &PyVolumeRenderer::backward,
             py::arg("final_states"),
             py::arg("last_points"),
             py::arg("sample_counts"),
             py::arg("sample_indices"),
             py::arg("positions"),
             py::arg("scales"),
             py::arg("rotations"),
             py::arg("opacities"),
             py::arg("features"),
             py::arg("ray_origins"),
             py::arg("ray_directions"),
             py::arg("grad_colors"),
             py::arg("grad_depths"),
             py::arg("grad_distortions"),
             py::arg("t_min"),
             py::arg("t_max"),
             py::arg("max_samples"),
             py::arg("sh_degree"),
             "Backward pass: compute gradients");
}
