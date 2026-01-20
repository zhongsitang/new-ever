#pragma once

/**
 * @file GaussianRT.h
 * @brief Main header for GaussianRT - Differentiable Gaussian Ray Tracing
 *
 * This library provides a modern implementation of differentiable
 * 3D Gaussian splatting using hardware-accelerated ray tracing.
 *
 * Features:
 * - Hardware ray tracing via Slang-RHI
 * - Differentiable rendering with automatic gradient computation
 * - PyTorch integration for deep learning workflows
 * - Support for spherical harmonics view-dependent colors
 *
 * Basic usage:
 * @code
 * #include <gaussian_rt/GaussianRT.h>
 *
 * using namespace gaussian_rt;
 *
 * // Initialize device
 * initializeGlobalDevice(0);  // CUDA device 0
 *
 * // Create primitives
 * PrimitiveSet prims(getGlobalDevice());
 * prims.setData(numPrims, positions, scales, orientations, densities, features, featureSize);
 *
 * // Build acceleration structure
 * AccelStruct accel(getGlobalDevice());
 * accel.build(prims);
 *
 * // Create renderer
 * ForwardRenderer renderer(getGlobalDevice());
 * renderer.initialize(true);  // enable backward
 *
 * // Render
 * ForwardOutput output;
 * renderer.allocateOutput(numRays, maxIters, output);
 * renderer.traceRays(accel, prims, rayOrigins, rayDirs, numRays, params, output);
 *
 * // Backward (if needed)
 * BackwardPass backward(getGlobalDevice());
 * backward.initialize();
 * GradientOutput gradients;
 * backward.allocateGradients(numPrims, numRays, featureSize, gradients);
 * backward.compute(output, dLdColor, prims, rayOrigins, rayDirs, numRays, params, gradients);
 * @endcode
 */

#include "Types.h"
#include "Device.h"
#include "PrimitiveSet.h"
#include "AccelStruct.h"
#include "ForwardRenderer.h"
#include "BackwardPass.h"

namespace gaussian_rt {

/**
 * @brief Library version
 */
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Get library version string
 */
inline const char* getVersionString() {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return version;
}

/**
 * @brief High-level renderer class combining all components
 *
 * Convenience class for simple use cases.
 */
class GaussianRenderer {
public:
    GaussianRenderer();
    ~GaussianRenderer();

    /**
     * @brief Initialize renderer
     * @param cudaDeviceId CUDA device to use
     * @param enableBackward Enable gradient computation
     */
    Result initialize(int cudaDeviceId = 0, bool enableBackward = true);

    /**
     * @brief Set volume primitive data
     */
    Result setPrimitives(
        size_t numPrimitives,
        const float* positions,
        const float* scales,
        const float* orientations,
        const float* densities,
        const float* features,
        size_t featureSize);

    /**
     * @brief Set primitives from device pointers
     */
    Result setPrimitivesDevice(
        size_t numPrimitives,
        void* d_positions,
        void* d_scales,
        void* d_orientations,
        void* d_densities,
        void* d_features,
        size_t featureSize);

    /**
     * @brief Rebuild acceleration structure
     */
    Result rebuildAccel();

    /**
     * @brief Update acceleration structure (refit)
     */
    Result updateAccel();

    /**
     * @brief Render image
     *
     * @param d_rayOrigins Device pointer to ray origins
     * @param d_rayDirections Device pointer to ray directions
     * @param numRays Number of rays
     * @param d_outputColor Device pointer for output color (float4 * numRays)
     * @param params Render parameters
     */
    Result render(
        void* d_rayOrigins,
        void* d_rayDirections,
        size_t numRays,
        void* d_outputColor,
        const RenderParams& params);

    /**
     * @brief Compute gradients
     *
     * @param d_dLdColor Device pointer to loss gradients
     * @param d_dPositions Output: gradients for positions
     * @param d_dScales Output: gradients for scales
     * @param d_dOrientations Output: gradients for orientations
     * @param d_dDensities Output: gradients for densities
     * @param d_dFeatures Output: gradients for features
     */
    Result backward(
        void* d_dLdColor,
        void* d_dPositions,
        void* d_dScales,
        void* d_dOrientations,
        void* d_dDensities,
        void* d_dFeatures);

    /**
     * @brief Synchronize
     */
    void synchronize();

    // Accessors
    Device& getDevice() { return *m_device; }
    PrimitiveSet& getPrimitives() { return *m_primitives; }
    AccelStruct& getAccelStruct() { return *m_accel; }
    ForwardRenderer& getForwardRenderer() { return *m_forward; }
    BackwardPass& getBackwardPass() { return *m_backward; }

private:
    std::unique_ptr<Device> m_device;
    std::unique_ptr<PrimitiveSet> m_primitives;
    std::unique_ptr<AccelStruct> m_accel;
    std::unique_ptr<ForwardRenderer> m_forward;
    std::unique_ptr<BackwardPass> m_backward;

    ForwardOutput m_forwardOutput;
    GradientOutput m_gradients;

    RenderParams m_lastParams;
    size_t m_lastNumRays = 0;
    bool m_enableBackward = true;
    bool m_initialized = false;
};

} // namespace gaussian_rt
