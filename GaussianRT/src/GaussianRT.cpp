#include "gaussian_rt/GaussianRT.h"

namespace gaussian_rt {

//------------------------------------------------------------------------------
// GaussianRenderer implementation
//------------------------------------------------------------------------------

GaussianRenderer::GaussianRenderer() {
}

GaussianRenderer::~GaussianRenderer() {
    if (m_forward) {
        m_forward->freeOutput(m_forwardOutput);
    }
    if (m_backward) {
        m_backward->freeGradients(m_gradients);
    }
}

Result GaussianRenderer::initialize(int cudaDeviceId, bool enableBackward) {
    if (m_initialized) {
        return Result::Success;
    }

    m_enableBackward = enableBackward;

    // Create device
    m_device = std::make_unique<Device>();
    Result result = m_device->initialize(cudaDeviceId);
    if (result != Result::Success) {
        return result;
    }

    // Create primitives manager
    m_primitives = std::make_unique<GaussianPrimitives>(*m_device);

    // Create acceleration structure
    m_accel = std::make_unique<AccelStruct>(*m_device);

    // Create forward renderer
    m_forward = std::make_unique<ForwardRenderer>(*m_device);
    result = m_forward->initialize(enableBackward);
    if (result != Result::Success) {
        return result;
    }

    // Create backward pass
    if (enableBackward) {
        m_backward = std::make_unique<BackwardPass>(*m_device);
        result = m_backward->initialize();
        if (result != Result::Success) {
            return result;
        }
    }

    m_initialized = true;
    return Result::Success;
}

Result GaussianRenderer::setPrimitives(
    size_t numPrimitives,
    const float* means,
    const float* scales,
    const float* quats,
    const float* densities,
    const float* features,
    size_t featureSize) {

    if (!m_initialized) {
        return Result::ErrorDeviceNotInitialized;
    }

    return m_primitives->setData(
        numPrimitives,
        means, scales, quats, densities, features,
        featureSize
    );
}

Result GaussianRenderer::setPrimitivesDevice(
    size_t numPrimitives,
    void* d_means,
    void* d_scales,
    void* d_quats,
    void* d_densities,
    void* d_features,
    size_t featureSize) {

    if (!m_initialized) {
        return Result::ErrorDeviceNotInitialized;
    }

    return m_primitives->setDataFromDevice(
        numPrimitives,
        d_means, d_scales, d_quats, d_densities, d_features,
        featureSize
    );
}

Result GaussianRenderer::rebuildAccel() {
    if (!m_initialized || !m_primitives->isValid()) {
        return Result::ErrorInvalidArgument;
    }

    return m_accel->build(*m_primitives, true, false);
}

Result GaussianRenderer::updateAccel() {
    if (!m_initialized || !m_accel->isValid()) {
        return Result::ErrorInvalidArgument;
    }

    // Update AABBs first
    Result result = m_primitives->updateAABBs();
    if (result != Result::Success) {
        return result;
    }

    return m_accel->update(*m_primitives);
}

Result GaussianRenderer::render(
    void* d_rayOrigins,
    void* d_rayDirections,
    size_t numRays,
    void* d_outputColor,
    const RenderParams& params) {

    if (!m_initialized || !m_accel->isValid()) {
        return Result::ErrorInvalidArgument;
    }

    // Reallocate output if needed
    if (numRays != m_lastNumRays || params.maxIters != m_lastParams.maxIters) {
        m_forward->freeOutput(m_forwardOutput);
        Result result = m_forward->allocateOutput(numRays, params.maxIters, m_forwardOutput);
        if (result != Result::Success) {
            return result;
        }
        m_lastNumRays = numRays;
        m_lastParams = params;
    }

    // Trace rays
    Result result = m_forward->traceRays(
        *m_accel,
        *m_primitives,
        d_rayOrigins,
        d_rayDirections,
        numRays,
        params,
        m_forwardOutput
    );

    if (result != Result::Success) {
        return result;
    }

    // Copy color to output
    cudaMemcpy(
        d_outputColor,
        m_forwardOutput.d_color,
        numRays * 4 * sizeof(float),
        cudaMemcpyDeviceToDevice
    );

    return Result::Success;
}

Result GaussianRenderer::backward(
    void* d_dLdColor,
    void* d_dMeans,
    void* d_dScales,
    void* d_dQuats,
    void* d_dDensities,
    void* d_dFeatures) {

    if (!m_initialized || !m_enableBackward || !m_backward) {
        return Result::ErrorInvalidArgument;
    }

    // Setup gradient output structure
    GradientOutput gradients;
    gradients.dMeans = d_dMeans;
    gradients.dScales = d_dScales;
    gradients.dQuats = d_dQuats;
    gradients.dDensities = d_dDensities;
    gradients.dFeatures = d_dFeatures;
    gradients.numPrims = m_primitives->getNumPrimitives();
    gradients.numRays = m_lastNumRays;
    gradients.featureSize = m_primitives->getFeatureSize();

    // Note: Ray gradients not exposed in this API
    // Allocate temp buffers for them
    void* d_dRayOrigins = m_device->createBuffer(m_lastNumRays * 3 * sizeof(float));
    void* d_dRayDirs = m_device->createBuffer(m_lastNumRays * 3 * sizeof(float));

    gradients.dRayOrigins = d_dRayOrigins;
    gradients.dRayDirs = d_dRayDirs;

    // We need the ray buffers - in practice these would be saved
    // For now, this is a limitation of the simple API

    Result result = m_backward->compute(
        m_forwardOutput,
        d_dLdColor,
        *m_primitives,
        nullptr,  // Ray origins - would need to be saved
        nullptr,  // Ray directions - would need to be saved
        m_lastNumRays,
        m_lastParams,
        gradients
    );

    // Free temp buffers
    m_device->freeBuffer(d_dRayOrigins);
    m_device->freeBuffer(d_dRayDirs);

    return result;
}

void GaussianRenderer::synchronize() {
    if (m_device) {
        m_device->synchronize();
    }
}

} // namespace gaussian_rt
