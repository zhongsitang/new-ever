#pragma once

#include "Types.h"
#include "Device.h"
#include "GaussianPrimitives.h"

namespace gaussian_rt {

/**
 * @brief Manages ray tracing acceleration structure
 *
 * Builds and maintains BLAS (Bottom Level Acceleration Structure) and
 * TLAS (Top Level Acceleration Structure) for efficient ray tracing.
 */
class AccelStruct {
public:
    AccelStruct(Device& device);
    ~AccelStruct();

    // Non-copyable
    AccelStruct(const AccelStruct&) = delete;
    AccelStruct& operator=(const AccelStruct&) = delete;

    // Movable
    AccelStruct(AccelStruct&&) noexcept;
    AccelStruct& operator=(AccelStruct&&) noexcept;

    /**
     * @brief Build acceleration structure from primitives
     *
     * @param primitives Gaussian primitives to build from
     * @param allowUpdate If true, structure can be updated later
     * @param fastBuild If true, prioritize build speed over trace speed
     * @return Result code
     */
    Result build(const GaussianPrimitives& primitives,
                 bool allowUpdate = true,
                 bool fastBuild = false);

    /**
     * @brief Update acceleration structure (refit)
     *
     * Only works if built with allowUpdate=true.
     * Use when primitive positions change but topology stays the same.
     */
    Result update(const GaussianPrimitives& primitives);

    /**
     * @brief Rebuild acceleration structure
     *
     * Use when primitive count changes.
     */
    Result rebuild(const GaussianPrimitives& primitives);

    /**
     * @brief Check if structure is valid
     */
    bool isValid() const { return m_built; }

    /**
     * @brief Check if structure supports update
     */
    bool canUpdate() const { return m_allowUpdate; }

    //--------------------------------------------------------------------------
    // Accessors for rendering
    //--------------------------------------------------------------------------

    // Get traversable handle for ray tracing
    void* getTraversableHandle() const { return m_traversableHandle; }

    // Get BLAS buffer for binding
    void* getBLASBuffer() const { return m_blasBuffer; }

    // Get TLAS buffer for binding
    void* getTLASBuffer() const { return m_tlasBuffer; }

    size_t getNumPrimitives() const { return m_numPrimitives; }

private:
    Device& m_device;

    bool m_built = false;
    bool m_allowUpdate = false;
    size_t m_numPrimitives = 0;

    // Acceleration structure buffers
    void* m_blasBuffer = nullptr;
    void* m_tlasBuffer = nullptr;
    void* m_instanceBuffer = nullptr;
    void* m_scratchBuffer = nullptr;

    // Traversable handle for ray tracing
    void* m_traversableHandle = nullptr;

    // Build sizes
    size_t m_blasSize = 0;
    size_t m_tlasSize = 0;
    size_t m_scratchSize = 0;

    // Internal helpers
    Result buildBLAS(const GaussianPrimitives& primitives, bool fastBuild);
    Result buildTLAS(bool fastBuild);
    void freeBuffers();
};

} // namespace gaussian_rt
