#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gaussian_rt/GaussianRT.h"

#include <cuda_runtime.h>

#ifdef GAUSSIANRT_HAS_TORCH
#include <torch/extension.h>
#endif

namespace py = pybind11;
using namespace gaussian_rt;

//------------------------------------------------------------------------------
// Helper functions for tensor conversion
//------------------------------------------------------------------------------

// Get device pointer from numpy array (copies to GPU)
void* numpyToDevice(Device& device, py::array_t<float> arr) {
    py::buffer_info info = arr.request();
    size_t size = info.size * sizeof(float);
    void* d_ptr = device.createBuffer(size, info.ptr);
    return d_ptr;
}

// Copy GPU data to numpy array
py::array_t<float> deviceToNumpy(Device& device, void* d_ptr, std::vector<ssize_t> shape) {
    size_t totalSize = 1;
    for (auto s : shape) totalSize *= s;

    py::array_t<float> result(shape);
    py::buffer_info info = result.request();

    device.downloadFromBuffer(d_ptr, info.ptr, totalSize * sizeof(float));
    return result;
}

#ifdef GAUSSIANRT_HAS_TORCH
// Get device pointer from PyTorch tensor (no copy if already on GPU)
void* tensorToDevice(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    return tensor.data_ptr();
}

// Create PyTorch tensor from GPU buffer (no copy)
torch::Tensor deviceToTensor(void* d_ptr, std::vector<int64_t> shape, torch::Device device) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);

    // Note: This creates a tensor that doesn't own the memory
    // The buffer must remain valid while the tensor is in use
    return torch::from_blob(d_ptr, shape, options);
}
#endif

//------------------------------------------------------------------------------
// Python wrapper classes
//------------------------------------------------------------------------------

class PyDevice {
public:
    Device device;

    PyDevice() = default;

    bool initialize(int cudaDeviceId = 0, bool enableValidation = false) {
        Result result = device.initialize(cudaDeviceId, enableValidation);
        return result == Result::Success;
    }

    void synchronize() {
        device.synchronize();
    }

    int getCudaDeviceId() const {
        return device.getCudaDeviceId();
    }

    bool isInitialized() const {
        return device.isInitialized();
    }
};

class PyPrimitives {
public:
    std::unique_ptr<GaussianPrimitives> prims;
    Device* devicePtr = nullptr;

    PyPrimitives(PyDevice& pyDevice) {
        devicePtr = &pyDevice.device;
        prims = std::make_unique<GaussianPrimitives>(pyDevice.device);
    }

    bool setData(
        py::array_t<float> means,
        py::array_t<float> scales,
        py::array_t<float> quats,
        py::array_t<float> densities,
        py::array_t<float> features) {

        py::buffer_info meansInfo = means.request();
        py::buffer_info scalesInfo = scales.request();
        py::buffer_info quatsInfo = quats.request();
        py::buffer_info densitiesInfo = densities.request();
        py::buffer_info featuresInfo = features.request();

        if (meansInfo.ndim != 2 || meansInfo.shape[1] != 3) {
            throw std::runtime_error("means must be (N, 3)");
        }

        size_t numPrims = meansInfo.shape[0];
        size_t featureSize = featuresInfo.shape[1];

        Result result = prims->setData(
            numPrims,
            static_cast<const float*>(meansInfo.ptr),
            static_cast<const float*>(scalesInfo.ptr),
            static_cast<const float*>(quatsInfo.ptr),
            static_cast<const float*>(densitiesInfo.ptr),
            static_cast<const float*>(featuresInfo.ptr),
            featureSize
        );

        return result == Result::Success;
    }

#ifdef GAUSSIANRT_HAS_TORCH
    bool setDataFromTensors(
        torch::Tensor means,
        torch::Tensor scales,
        torch::Tensor quats,
        torch::Tensor densities,
        torch::Tensor features) {

        TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
        TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be (N, 3)");
        TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be (N, 4)");
        TORCH_CHECK(densities.dim() == 1, "densities must be (N,)");
        TORCH_CHECK(features.dim() == 2, "features must be (N, F)");

        size_t numPrims = means.size(0);
        size_t featureSize = features.size(1);

        Result result = prims->setDataFromDevice(
            numPrims,
            tensorToDevice(means.contiguous()),
            tensorToDevice(scales.contiguous()),
            tensorToDevice(quats.contiguous()),
            tensorToDevice(densities.contiguous()),
            tensorToDevice(features.contiguous()),
            featureSize
        );

        return result == Result::Success;
    }
#endif

    size_t getNumPrimitives() const {
        return prims->getNumPrimitives();
    }

    size_t getFeatureSize() const {
        return prims->getFeatureSize();
    }

    bool updateAABBs() {
        return prims->updateAABBs() == Result::Success;
    }
};

class PyAccelStruct {
public:
    std::unique_ptr<AccelStruct> accel;

    PyAccelStruct(PyDevice& pyDevice) {
        accel = std::make_unique<AccelStruct>(pyDevice.device);
    }

    bool build(PyPrimitives& prims, bool allowUpdate = true, bool fastBuild = false) {
        return accel->build(*prims.prims, allowUpdate, fastBuild) == Result::Success;
    }

    bool update(PyPrimitives& prims) {
        return accel->update(*prims.prims) == Result::Success;
    }

    bool isValid() const {
        return accel->isValid();
    }
};

class PyForwardRenderer {
public:
    std::unique_ptr<ForwardRenderer> renderer;
    Device* devicePtr = nullptr;

    PyForwardRenderer(PyDevice& pyDevice) {
        devicePtr = &pyDevice.device;
        renderer = std::make_unique<ForwardRenderer>(pyDevice.device);
    }

    bool initialize(bool enableBackward = true, const char* shaderPath = nullptr) {
        return renderer->initialize(enableBackward, shaderPath) == Result::Success;
    }

#ifdef GAUSSIANRT_HAS_TORCH
    py::dict traceRaysTorch(
        PyAccelStruct& accel,
        PyPrimitives& prims,
        torch::Tensor rayOrigins,
        torch::Tensor rayDirections,
        float tmin,
        float tmax,
        uint32_t maxIters,
        uint32_t shDegree,
        float maxPrimSize) {

        TORCH_CHECK(rayOrigins.dim() == 2 && rayOrigins.size(1) == 3, "rayOrigins must be (M, 3)");
        TORCH_CHECK(rayDirections.dim() == 2 && rayDirections.size(1) == 3, "rayDirections must be (M, 3)");

        size_t numRays = rayOrigins.size(0);
        auto device = rayOrigins.device();

        // Setup render params
        RenderParams params;
        params.tmin = tmin;
        params.tmax = tmax;
        params.maxIters = maxIters;
        params.shDegree = shDegree;
        params.maxPrimSize = maxPrimSize;

        // Allocate output
        ForwardOutput output;
        Result result = renderer->allocateOutput(numRays, maxIters, output);
        if (result != Result::Success) {
            throw std::runtime_error("Failed to allocate output buffers");
        }

        // Trace rays
        result = renderer->traceRays(
            *accel.accel,
            *prims.prims,
            tensorToDevice(rayOrigins.contiguous()),
            tensorToDevice(rayDirections.contiguous()),
            numRays,
            params,
            output
        );

        if (result != Result::Success) {
            renderer->freeOutput(output);
            throw std::runtime_error("Ray tracing failed");
        }

        // Create output tensors
        auto colorTensor = deviceToTensor(output.d_color, {(int64_t)numRays, 4}, device);
        auto stateTensor = deviceToTensor(output.d_state, {(int64_t)numRays, 8}, device);
        auto triCollectionTensor = deviceToTensor(output.d_triCollection,
            {(int64_t)numRays, (int64_t)maxIters}, device);
        auto itersTensor = deviceToTensor(output.d_iters, {(int64_t)numRays}, device);

        return py::dict(
            py::arg("color") = colorTensor,
            py::arg("state") = stateTensor,
            py::arg("tri_collection") = triCollectionTensor,
            py::arg("iters") = itersTensor
        );
    }
#endif

    bool isInitialized() const {
        return renderer->isInitialized();
    }
};

class PyBackwardPass {
public:
    std::unique_ptr<BackwardPass> backward;
    Device* devicePtr = nullptr;

    PyBackwardPass(PyDevice& pyDevice) {
        devicePtr = &pyDevice.device;
        backward = std::make_unique<BackwardPass>(pyDevice.device);
    }

    bool initialize() {
        return backward->initialize() == Result::Success;
    }

#ifdef GAUSSIANRT_HAS_TORCH
    py::dict computeTorch(
        torch::Tensor lastState,
        torch::Tensor triCollection,
        torch::Tensor iters,
        torch::Tensor dLdColor,
        PyPrimitives& prims,
        torch::Tensor rayOrigins,
        torch::Tensor rayDirections,
        float tmin,
        float tmax,
        uint32_t maxIters,
        uint32_t shDegree) {

        size_t numRays = rayOrigins.size(0);
        size_t numPrims = prims.getNumPrimitives();
        size_t featureSize = prims.getFeatureSize();
        auto device = rayOrigins.device();

        // Setup params
        RenderParams params;
        params.tmin = tmin;
        params.tmax = tmax;
        params.maxIters = maxIters;
        params.shDegree = shDegree;

        // Setup forward output structure
        ForwardOutput forwardOutput;
        forwardOutput.d_state = tensorToDevice(lastState);
        forwardOutput.d_triCollection = tensorToDevice(triCollection);
        forwardOutput.d_iters = tensorToDevice(iters);
        forwardOutput.numRays = numRays;
        forwardOutput.maxIters = maxIters;

        // Allocate gradients
        GradientOutput gradients;
        Result result = backward->allocateGradients(numPrims, numRays, featureSize, gradients);
        if (result != Result::Success) {
            throw std::runtime_error("Failed to allocate gradient buffers");
        }

        // Compute gradients
        result = backward->compute(
            forwardOutput,
            tensorToDevice(dLdColor.contiguous()),
            *prims.prims,
            tensorToDevice(rayOrigins.contiguous()),
            tensorToDevice(rayDirections.contiguous()),
            numRays,
            params,
            gradients
        );

        if (result != Result::Success) {
            backward->freeGradients(gradients);
            throw std::runtime_error("Backward pass failed");
        }

        // Create output tensors
        auto dMeansTensor = deviceToTensor(gradients.dMeans, {(int64_t)numPrims, 3}, device);
        auto dScalesTensor = deviceToTensor(gradients.dScales, {(int64_t)numPrims, 3}, device);
        auto dQuatsTensor = deviceToTensor(gradients.dQuats, {(int64_t)numPrims, 4}, device);
        auto dDensitiesTensor = deviceToTensor(gradients.dDensities, {(int64_t)numPrims}, device);
        auto dFeaturesTensor = deviceToTensor(gradients.dFeatures,
            {(int64_t)numPrims, (int64_t)featureSize}, device);

        return py::dict(
            py::arg("dMeans") = dMeansTensor,
            py::arg("dScales") = dScalesTensor,
            py::arg("dQuats") = dQuatsTensor,
            py::arg("dDensities") = dDensitiesTensor,
            py::arg("dFeatures") = dFeaturesTensor
        );
    }
#endif

    bool isInitialized() const {
        return backward->isInitialized();
    }
};

//------------------------------------------------------------------------------
// High-level renderer wrapper
//------------------------------------------------------------------------------

class PyGaussianRenderer {
public:
    std::unique_ptr<GaussianRenderer> renderer;

    PyGaussianRenderer() {
        renderer = std::make_unique<GaussianRenderer>();
    }

    bool initialize(int cudaDeviceId = 0, bool enableBackward = true) {
        return renderer->initialize(cudaDeviceId, enableBackward) == Result::Success;
    }

    bool setPrimitives(
        py::array_t<float> means,
        py::array_t<float> scales,
        py::array_t<float> quats,
        py::array_t<float> densities,
        py::array_t<float> features) {

        py::buffer_info meansInfo = means.request();
        size_t numPrims = meansInfo.shape[0];
        size_t featureSize = features.request().shape[1];

        return renderer->setPrimitives(
            numPrims,
            static_cast<const float*>(meansInfo.ptr),
            static_cast<const float*>(scales.request().ptr),
            static_cast<const float*>(quats.request().ptr),
            static_cast<const float*>(densities.request().ptr),
            static_cast<const float*>(features.request().ptr),
            featureSize
        ) == Result::Success;
    }

    bool rebuildAccel() {
        return renderer->rebuildAccel() == Result::Success;
    }

    bool updateAccel() {
        return renderer->updateAccel() == Result::Success;
    }

    void synchronize() {
        renderer->synchronize();
    }
};

//------------------------------------------------------------------------------
// Module definition
//------------------------------------------------------------------------------

PYBIND11_MODULE(gaussian_rt_ext, m) {
    m.doc() = "GaussianRT - Differentiable Gaussian Ray Tracing";

    // Version info
    m.attr("__version__") = getVersionString();

    // Device class
    py::class_<PyDevice>(m, "Device")
        .def(py::init<>())
        .def("initialize", &PyDevice::initialize,
             py::arg("cuda_device_id") = 0,
             py::arg("enable_validation") = false)
        .def("synchronize", &PyDevice::synchronize)
        .def("get_cuda_device_id", &PyDevice::getCudaDeviceId)
        .def("is_initialized", &PyDevice::isInitialized);

    // Primitives class
    py::class_<PyPrimitives>(m, "Primitives")
        .def(py::init<PyDevice&>())
        .def("set_data", &PyPrimitives::setData)
#ifdef GAUSSIANRT_HAS_TORCH
        .def("set_data_from_tensors", &PyPrimitives::setDataFromTensors)
#endif
        .def("get_num_primitives", &PyPrimitives::getNumPrimitives)
        .def("get_feature_size", &PyPrimitives::getFeatureSize)
        .def("update_aabbs", &PyPrimitives::updateAABBs);

    // AccelStruct class
    py::class_<PyAccelStruct>(m, "AccelStruct")
        .def(py::init<PyDevice&>())
        .def("build", &PyAccelStruct::build,
             py::arg("primitives"),
             py::arg("allow_update") = true,
             py::arg("fast_build") = false)
        .def("update", &PyAccelStruct::update)
        .def("is_valid", &PyAccelStruct::isValid);

    // ForwardRenderer class
    py::class_<PyForwardRenderer>(m, "ForwardRenderer")
        .def(py::init<PyDevice&>())
        .def("initialize", &PyForwardRenderer::initialize,
             py::arg("enable_backward") = true,
             py::arg("shader_path") = nullptr)
#ifdef GAUSSIANRT_HAS_TORCH
        .def("trace_rays", &PyForwardRenderer::traceRaysTorch)
#endif
        .def("is_initialized", &PyForwardRenderer::isInitialized);

    // BackwardPass class
    py::class_<PyBackwardPass>(m, "BackwardPass")
        .def(py::init<PyDevice&>())
        .def("initialize", &PyBackwardPass::initialize)
#ifdef GAUSSIANRT_HAS_TORCH
        .def("compute", &PyBackwardPass::computeTorch)
#endif
        .def("is_initialized", &PyBackwardPass::isInitialized);

    // High-level renderer
    py::class_<PyGaussianRenderer>(m, "GaussianRenderer")
        .def(py::init<>())
        .def("initialize", &PyGaussianRenderer::initialize,
             py::arg("cuda_device_id") = 0,
             py::arg("enable_backward") = true)
        .def("set_primitives", &PyGaussianRenderer::setPrimitives)
        .def("rebuild_accel", &PyGaussianRenderer::rebuildAccel)
        .def("update_accel", &PyGaussianRenderer::updateAccel)
        .def("synchronize", &PyGaussianRenderer::synchronize);

    // Utility functions
    m.def("get_version", &getVersionString, "Get library version string");
}
