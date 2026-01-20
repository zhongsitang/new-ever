#pragma once

#include "Types.h"
#include "Device.h"
#include "PrimitiveSet.h"

namespace gaussian_rt {

/**
 * @brief Backward pass for differentiable rendering
 *
 * Computes gradients for all primitive parameters using the
 * forward pass output and loss gradients.
 */
class BackwardPass {
public:
    BackwardPass(Device& device);
    ~BackwardPass();

    // Non-copyable
    BackwardPass(const BackwardPass&) = delete;
    BackwardPass& operator=(const BackwardPass&) = delete;

    /**
     * @brief Initialize the backward pass
     *
     * Loads the pre-compiled CUDA kernel.
     * @return Result code
     */
    Result initialize();

    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Compute gradients
     *
     * @param forwardOutput Output from forward pass
     * @param d_dLdColor Device pointer to loss gradients w.r.t. color (float4 * numRays)
     * @param primitives Volume primitives
     * @param d_rayOrigins Device pointer to ray origins
     * @param d_rayDirections Device pointer to ray directions
     * @param numRays Number of rays
     * @param params Render parameters
     * @param gradients Output gradient structure (pre-allocated)
     * @return Result code
     */
    Result compute(
        const ForwardOutput& forwardOutput,
        void* d_dLdColor,
        const PrimitiveSet& primitives,
        void* d_rayOrigins,
        void* d_rayDirections,
        size_t numRays,
        const RenderParams& params,
        GradientOutput& gradients);

    /**
     * @brief Allocate gradient buffers
     *
     * @param numPrimitives Number of primitives
     * @param numRays Number of rays
     * @param featureSize Feature size per primitive
     * @param gradients Gradient structure to allocate
     * @return Result code
     */
    Result allocateGradients(
        size_t numPrimitives,
        size_t numRays,
        size_t featureSize,
        GradientOutput& gradients);

    /**
     * @brief Free gradient buffers
     */
    void freeGradients(GradientOutput& gradients);

    /**
     * @brief Zero out gradient buffers
     */
    Result zeroGradients(GradientOutput& gradients);

private:
    Device& m_device;
    bool m_initialized = false;

    // CUDA kernel
    void* m_cudaModule = nullptr;
    void* m_backwardKernel = nullptr;

    // Kernel launch parameters
    uint32_t m_blockSize = 256;
};

} // namespace gaussian_rt
