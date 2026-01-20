// Native OptiX curve acceleration structure implementation

#include "curve_primitives.h"
#include <stdexcept>
#include <cstring>

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            throw std::runtime_error("OptiX call failed: " #call);             \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error("CUDA call failed: " #call);              \
        }                                                                      \
    } while (0)

namespace gaussian_rt {
namespace optix_native {

// ============================================================================
// CurveContext Implementation
// ============================================================================

CurveContext::CurveContext() = default;

CurveContext::~CurveContext() {
    shutdown();
}

bool CurveContext::initialize(int device_id) {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(nullptr));  // Force context creation

    // Get CUDA context
    CUresult cu_res = cuCtxGetCurrent(&cuda_context_);
    if (cu_res != CUDA_SUCCESS || !cuda_context_) {
        return false;
    }

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 0;

    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &context_));

    return true;
}

void CurveContext::shutdown() {
    if (context_) {
        optixDeviceContextDestroy(context_);
        context_ = nullptr;
    }
}

// ============================================================================
// CurveAccelerationStructure Implementation
// ============================================================================

CurveAccelerationStructure::CurveAccelerationStructure(CurveContext& context)
    : context_(context) {}

CurveAccelerationStructure::~CurveAccelerationStructure() {
    if (d_vertices_) cudaFree(reinterpret_cast<void*>(d_vertices_));
    if (d_indices_) cudaFree(reinterpret_cast<void*>(d_indices_));
    if (d_gas_output_) cudaFree(reinterpret_cast<void*>(d_gas_output_));
    if (d_temp_buffer_) cudaFree(reinterpret_cast<void*>(d_temp_buffer_));
}

void CurveAccelerationStructure::build(const CurveData& curves, bool allow_update) {
    // Upload vertices to device
    size_t vertices_size = curves.vertices.size() * sizeof(float4);
    if (d_vertices_) cudaFree(reinterpret_cast<void*>(d_vertices_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices_),
        curves.vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    // Upload indices if present
    CUdeviceptr d_indices_local = 0;
    if (!curves.indices.empty()) {
        size_t indices_size = curves.indices.size() * sizeof(uint32_t);
        if (d_indices_) cudaFree(reinterpret_cast<void*>(d_indices_));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), indices_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_indices_),
            curves.indices.data(),
            indices_size,
            cudaMemcpyHostToDevice
        ));
        d_indices_local = d_indices_;
    }

    // Configure curve build input
    OptixBuildInput curve_input = {};
    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    // Curve array configuration
    curve_input.curveArray.curveType = to_optix_type(curves.type);
    curve_input.curveArray.numPrimitives = curves.num_segments;

    // Vertex buffer: float4 (x, y, z, radius)
    curve_input.curveArray.vertexBuffers = &d_vertices_;
    curve_input.curveArray.numVertices = static_cast<uint32_t>(curves.vertices.size());
    curve_input.curveArray.vertexStrideInBytes = sizeof(float4);

    // Use radius from vertex.w, no separate width buffer
    curve_input.curveArray.widthBuffers = nullptr;
    curve_input.curveArray.widthStrideInBytes = 0;
    curve_input.curveArray.normalizeWidths = 0;

    // Index buffer (optional)
    if (d_indices_local) {
        curve_input.curveArray.indexBuffer = d_indices_local;
        curve_input.curveArray.indexStrideInBytes = sizeof(uint32_t);
    } else {
        curve_input.curveArray.indexBuffer = 0;
        curve_input.curveArray.indexStrideInBytes = 0;
    }

    // Geometry flags
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.primitiveIndexOffset = 0;

    // End caps for round curves
    curve_input.curveArray.endcapFlags = OPTIX_CURVE_ENDCAP_ON;

    // Build flags
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (allow_update) {
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &curve_input,
        1,
        &buffer_sizes
    ));

    // Allocate temp buffer
    if (buffer_sizes.tempSizeInBytes > temp_buffer_size_) {
        if (d_temp_buffer_) cudaFree(reinterpret_cast<void*>(d_temp_buffer_));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer_),
            buffer_sizes.tempSizeInBytes
        ));
        temp_buffer_size_ = buffer_sizes.tempSizeInBytes;
    }

    // Allocate output buffer
    if (buffer_sizes.outputSizeInBytes > gas_output_size_) {
        if (d_gas_output_) cudaFree(reinterpret_cast<void*>(d_gas_output_));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_gas_output_),
            buffer_sizes.outputSizeInBytes
        ));
        gas_output_size_ = buffer_sizes.outputSizeInBytes;
    }

    // Build acceleration structure
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0,  // CUDA stream
        &accel_options,
        &curve_input,
        1,  // num build inputs
        d_temp_buffer_,
        buffer_sizes.tempSizeInBytes,
        d_gas_output_,
        buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        nullptr,  // emitted properties
        0         // num emitted properties
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    built_ = true;
}

void CurveAccelerationStructure::update(const CurveData& curves) {
    if (!built_) {
        throw std::runtime_error("Acceleration structure must be built before update");
    }

    // Update vertex data
    size_t vertices_size = curves.vertices.size() * sizeof(float4);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices_),
        curves.vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    // Configure for update
    OptixBuildInput curve_input = {};
    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    curve_input.curveArray.curveType = to_optix_type(curves.type);
    curve_input.curveArray.numPrimitives = curves.num_segments;
    curve_input.curveArray.vertexBuffers = &d_vertices_;
    curve_input.curveArray.numVertices = static_cast<uint32_t>(curves.vertices.size());
    curve_input.curveArray.vertexStrideInBytes = sizeof(float4);
    curve_input.curveArray.widthBuffers = nullptr;
    curve_input.curveArray.normalizeWidths = 0;
    curve_input.curveArray.indexBuffer = d_indices_;
    curve_input.curveArray.indexStrideInBytes = d_indices_ ? sizeof(uint32_t) : 0;
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.endcapFlags = OPTIX_CURVE_ENDCAP_ON;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    // Refit
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0,
        &accel_options,
        &curve_input,
        1,
        d_temp_buffer_,
        temp_buffer_size_,
        d_gas_output_,
        gas_output_size_,
        &gas_handle_,
        nullptr,
        0
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace optix_native
} // namespace gaussian_rt
