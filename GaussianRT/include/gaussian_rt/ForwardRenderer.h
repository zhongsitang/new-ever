#pragma once

#include "Types.h"
#include "Device.h"
#include "AccelStruct.h"
#include "GaussianPrimitives.h"

namespace gaussian_rt {

/**
 * @brief Forward ray tracing renderer
 *
 * Implements the forward pass of differentiable Gaussian rendering
 * using hardware-accelerated ray tracing.
 */
class ForwardRenderer {
public:
    ForwardRenderer(Device& device);
    ~ForwardRenderer();

    // Non-copyable
    ForwardRenderer(const ForwardRenderer&) = delete;
    ForwardRenderer& operator=(const ForwardRenderer&) = delete;

    /**
     * @brief Initialize the renderer
     *
     * @param enableBackward Enable saving data for backward pass
     * @param shaderPath Optional path to shader directory
     * @return Result code
     */
    Result initialize(bool enableBackward = true, const char* shaderPath = nullptr);

    /**
     * @brief Check if renderer is initialized
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Execute ray tracing
     *
     * @param accel Acceleration structure
     * @param primitives Gaussian primitives
     * @param d_rayOrigins Device pointer to ray origins (float3 * numRays)
     * @param d_rayDirections Device pointer to ray directions (float3 * numRays)
     * @param numRays Number of rays to trace
     * @param params Render parameters
     * @param output Output structure (pre-allocated)
     * @return Result code
     */
    Result traceRays(
        const AccelStruct& accel,
        const GaussianPrimitives& primitives,
        void* d_rayOrigins,
        void* d_rayDirections,
        size_t numRays,
        const RenderParams& params,
        ForwardOutput& output);

    /**
     * @brief Allocate output buffers
     *
     * @param numRays Number of rays
     * @param maxIters Maximum iterations per ray
     * @param output Output structure to allocate
     * @return Result code
     */
    Result allocateOutput(size_t numRays, size_t maxIters, ForwardOutput& output);

    /**
     * @brief Free output buffers
     */
    void freeOutput(ForwardOutput& output);

private:
    Device& m_device;
    bool m_initialized = false;
    bool m_enableBackward = true;

    // Pipeline state
    void* m_pipeline = nullptr;
    void* m_shaderTable = nullptr;
    void* m_program = nullptr;

    // Shader binding table buffers
    void* m_raygenRecord = nullptr;
    void* m_missRecord = nullptr;
    void* m_hitgroupRecord = nullptr;

    // Uniform buffer
    void* m_uniformBuffer = nullptr;

    // Internal helpers
    Result createPipeline(const char* shaderPath);
    Result createShaderTable();
    void freePipeline();
};

} // namespace gaussian_rt
