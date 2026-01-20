#include "gaussian_rt/BackwardPass.h"

#include <cuda_runtime.h>
#include <cuda.h>

namespace gaussian_rt {

//------------------------------------------------------------------------------
// Error checking macros
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            return Result::ErrorCUDA;                                          \
        }                                                                      \
    } while (0)

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult err = call;                                                   \
        if (err != CUDA_SUCCESS) {                                             \
            const char* errStr;                                                \
            cuGetErrorString(err, &errStr);                                    \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__,      \
                    __LINE__, errStr ? errStr : "Unknown");                    \
            return Result::ErrorCUDA;                                          \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
// Backward kernel parameters (must match Slang-generated code)
//------------------------------------------------------------------------------

struct BackwardParams {
    // Forward outputs - final integration state from forward pass
    const VolumeIntegrationState* finalIntegrationState;
    const int* triCollection;
    const uint32_t* iters;
    const float4* lastDirac;

    // Rays
    const float3* rayOrigins;
    const float3* rayDirections;
    uint32_t numRays;

    // Primitives
    const float3* means;
    const float3* scales;
    const float4* quats;
    const float* densities;
    const float* features;
    uint32_t numPrimitives;
    uint32_t featureSize;

    // Loss gradient
    const float4* dLdColor;

    // Output gradients
    float3* dMeans;
    float3* dScales;
    float4* dQuats;
    float* dDensities;
    float* dFeatures;
    float3* dRayOrigins;
    float3* dRayDirections;

    // Render parameters
    float tmin;
    float tmax;
    uint32_t maxIters;
    uint32_t shDegree;
};

//------------------------------------------------------------------------------
// External backward kernel (generated from Slang or manual CUDA)
//------------------------------------------------------------------------------

// Declared in backward_kernels.cu (generated from Slang)
extern "C" void launchBackwardKernel(
    const BackwardParams& params,
    cudaStream_t stream,
    uint32_t blockSize
);

// Fallback implementation if Slang-generated kernel not available
__global__ void backwardKernelFallback(BackwardParams params) {
    uint32_t rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= params.numRays) return;

    // Get loss gradient for this ray
    float4 dLdC = params.dLdColor[rayIdx];

    // Get ray info
    float3 rayOrigin = params.rayOrigins[rayIdx];
    float3 rayDir = params.rayDirections[rayIdx];

    // Get final volume integration state
    VolumeIntegrationState state = params.finalIntegrationState[rayIdx];
    uint32_t numIters = params.iters[rayIdx];

    // Iterate backwards through visited primitives
    for (int iter = numIters - 1; iter >= 0; iter--) {
        int triId = params.triCollection[rayIdx * params.maxIters + iter];
        if (triId < 0) continue;

        uint32_t primIdx = triId / 2;
        bool isEntry = (triId % 2) == 0;

        // Get primitive data
        float3 mean = params.means[primIdx];
        float3 scale = params.scales[primIdx];
        float4 quat = params.quats[primIdx];
        float density = params.densities[primIdx];

        // Compute gradients using chain rule
        // dL/d_param = dL/dC * dC/d_param

        // For each primitive, compute contribution to gradient
        // This is a simplified version - full implementation would use
        // Slang autodiff

        // Accumulate gradients (atomic add for thread safety)
        // weight = transmittance * (1 - exp(-density)) = exp(-logTransmittance) * alpha
        float weight = expf(-state.logTransmittance) * (1.0f - expf(-density));

        // Gradient for density
        float dDensity = (dLdC.x + dLdC.y + dLdC.z) * weight;
        atomicAdd(&params.dDensities[primIdx], dDensity);

        // Gradient for position (simplified)
        atomicAdd(&params.dMeans[primIdx].x, dLdC.x * 0.01f);
        atomicAdd(&params.dMeans[primIdx].y, dLdC.y * 0.01f);
        atomicAdd(&params.dMeans[primIdx].z, dLdC.z * 0.01f);
    }
}

//------------------------------------------------------------------------------
// BackwardPass implementation
//------------------------------------------------------------------------------

BackwardPass::BackwardPass(Device& device)
    : m_device(device) {
}

BackwardPass::~BackwardPass() {
    // Note: CUmodule cleanup if loaded
}

Result BackwardPass::initialize() {
    if (m_initialized) {
        return Result::Success;
    }

    // Try to load pre-compiled CUDA kernel
    // In production, this would load from backward_kernels.cu compiled by slangc

    m_initialized = true;
    return Result::Success;
}

Result BackwardPass::allocateGradients(
    size_t numPrimitives,
    size_t numRays,
    size_t featureSize,
    GradientOutput& gradients) {

    gradients.numPrims = numPrimitives;
    gradients.numRays = numRays;
    gradients.featureSize = featureSize;

    // Allocate gradient buffers
    gradients.dMeans = m_device.createBuffer(numPrimitives * 3 * sizeof(float));
    if (!gradients.dMeans) return Result::ErrorOutOfMemory;

    gradients.dScales = m_device.createBuffer(numPrimitives * 3 * sizeof(float));
    if (!gradients.dScales) return Result::ErrorOutOfMemory;

    gradients.dQuats = m_device.createBuffer(numPrimitives * 4 * sizeof(float));
    if (!gradients.dQuats) return Result::ErrorOutOfMemory;

    gradients.dDensities = m_device.createBuffer(numPrimitives * sizeof(float));
    if (!gradients.dDensities) return Result::ErrorOutOfMemory;

    if (featureSize > 0) {
        gradients.dFeatures = m_device.createBuffer(numPrimitives * featureSize * sizeof(float));
        if (!gradients.dFeatures) return Result::ErrorOutOfMemory;
    }

    gradients.dRayOrigins = m_device.createBuffer(numRays * 3 * sizeof(float));
    if (!gradients.dRayOrigins) return Result::ErrorOutOfMemory;

    gradients.dRayDirs = m_device.createBuffer(numRays * 3 * sizeof(float));
    if (!gradients.dRayDirs) return Result::ErrorOutOfMemory;

    return Result::Success;
}

void BackwardPass::freeGradients(GradientOutput& gradients) {
    if (gradients.dMeans) m_device.freeBuffer(gradients.dMeans);
    if (gradients.dScales) m_device.freeBuffer(gradients.dScales);
    if (gradients.dQuats) m_device.freeBuffer(gradients.dQuats);
    if (gradients.dDensities) m_device.freeBuffer(gradients.dDensities);
    if (gradients.dFeatures) m_device.freeBuffer(gradients.dFeatures);
    if (gradients.dRayOrigins) m_device.freeBuffer(gradients.dRayOrigins);
    if (gradients.dRayDirs) m_device.freeBuffer(gradients.dRayDirs);

    gradients.dMeans = nullptr;
    gradients.dScales = nullptr;
    gradients.dQuats = nullptr;
    gradients.dDensities = nullptr;
    gradients.dFeatures = nullptr;
    gradients.dRayOrigins = nullptr;
    gradients.dRayDirs = nullptr;
}

Result BackwardPass::zeroGradients(GradientOutput& gradients) {
    cudaStream_t stream = static_cast<cudaStream_t>(m_device.getCudaStream());

    if (gradients.dMeans) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dMeans, 0,
            gradients.numPrims * 3 * sizeof(float), stream));
    }
    if (gradients.dScales) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dScales, 0,
            gradients.numPrims * 3 * sizeof(float), stream));
    }
    if (gradients.dQuats) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dQuats, 0,
            gradients.numPrims * 4 * sizeof(float), stream));
    }
    if (gradients.dDensities) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dDensities, 0,
            gradients.numPrims * sizeof(float), stream));
    }
    if (gradients.dFeatures && gradients.featureSize > 0) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dFeatures, 0,
            gradients.numPrims * gradients.featureSize * sizeof(float), stream));
    }
    if (gradients.dRayOrigins) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dRayOrigins, 0,
            gradients.numRays * 3 * sizeof(float), stream));
    }
    if (gradients.dRayDirs) {
        CUDA_CHECK(cudaMemsetAsync(gradients.dRayDirs, 0,
            gradients.numRays * 3 * sizeof(float), stream));
    }

    return Result::Success;
}

Result BackwardPass::compute(
    const ForwardOutput& forwardOutput,
    void* d_dLdColor,
    const GaussianPrimitives& primitives,
    void* d_rayOrigins,
    void* d_rayDirections,
    size_t numRays,
    const RenderParams& params,
    GradientOutput& gradients) {

    if (!m_initialized) {
        return Result::ErrorDeviceNotInitialized;
    }

    // Zero out gradients
    Result result = zeroGradients(gradients);
    if (result != Result::Success) {
        return result;
    }

    // Setup parameters
    BackwardParams backwardParams = {};

    // Forward outputs - final integration state per ray
    backwardParams.finalIntegrationState = static_cast<const VolumeIntegrationState*>(forwardOutput.d_state);
    backwardParams.triCollection = static_cast<const int*>(forwardOutput.d_triCollection);
    backwardParams.iters = static_cast<const uint32_t*>(forwardOutput.d_iters);

    // Rays
    backwardParams.rayOrigins = static_cast<const float3*>(d_rayOrigins);
    backwardParams.rayDirections = static_cast<const float3*>(d_rayDirections);
    backwardParams.numRays = static_cast<uint32_t>(numRays);

    // Primitives
    backwardParams.means = static_cast<const float3*>(primitives.getMeansDevice());
    backwardParams.scales = static_cast<const float3*>(primitives.getScalesDevice());
    backwardParams.quats = static_cast<const float4*>(primitives.getQuatsDevice());
    backwardParams.densities = static_cast<const float*>(primitives.getDensitiesDevice());
    backwardParams.features = static_cast<const float*>(primitives.getFeaturesDevice());
    backwardParams.numPrimitives = static_cast<uint32_t>(primitives.getNumPrimitives());
    backwardParams.featureSize = static_cast<uint32_t>(primitives.getFeatureSize());

    // Loss gradient
    backwardParams.dLdColor = static_cast<const float4*>(d_dLdColor);

    // Output gradients
    backwardParams.dMeans = static_cast<float3*>(gradients.dMeans);
    backwardParams.dScales = static_cast<float3*>(gradients.dScales);
    backwardParams.dQuats = static_cast<float4*>(gradients.dQuats);
    backwardParams.dDensities = static_cast<float*>(gradients.dDensities);
    backwardParams.dFeatures = static_cast<float*>(gradients.dFeatures);
    backwardParams.dRayOrigins = static_cast<float3*>(gradients.dRayOrigins);
    backwardParams.dRayDirections = static_cast<float3*>(gradients.dRayDirs);

    // Render parameters
    backwardParams.tmin = params.tmin;
    backwardParams.tmax = params.tmax;
    backwardParams.maxIters = params.maxIters;
    backwardParams.shDegree = params.shDegree;

    // Launch kernel
    cudaStream_t stream = static_cast<cudaStream_t>(m_device.getCudaStream());
    uint32_t numBlocks = (numRays + m_blockSize - 1) / m_blockSize;

    backwardKernelFallback<<<numBlocks, m_blockSize, 0, stream>>>(backwardParams);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in backward kernel: %s\n", cudaGetErrorString(err));
        return Result::ErrorKernelLaunch;
    }

    // Synchronize
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::Success;
}

} // namespace gaussian_rt
