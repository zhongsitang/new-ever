#pragma once

#include "Types.h"
#include "Device.h"
#include <vector>

namespace gaussian_rt {

/**
 * @brief Manages primitive data on GPU
 *
 * Stores and manages volume primitives (currently ellipsoids) including:
 * - Position (center)
 * - Scale (radii)
 * - Orientation (quaternion)
 * - Density/opacity
 * - Features (RGB or spherical harmonics coefficients)
 *
 * This class is primitive-type agnostic and can be extended to support
 * different primitive types (sphere, capsule, etc.)
 */
class PrimitiveSet {
public:
    PrimitiveSet(Device& device);
    ~PrimitiveSet();

    // Non-copyable
    PrimitiveSet(const PrimitiveSet&) = delete;
    PrimitiveSet& operator=(const PrimitiveSet&) = delete;

    // Movable
    PrimitiveSet(PrimitiveSet&&) noexcept;
    PrimitiveSet& operator=(PrimitiveSet&&) noexcept;

    /**
     * @brief Set primitive data from raw pointers
     *
     * @param numPrimitives Number of primitives
     * @param positions Center positions (float3 * N)
     * @param scales Scale factors (float3 * N)
     * @param orientations Rotation quaternions (float4 * N)
     * @param densities Opacity values (float * N)
     * @param features Color/SH features (float * N * featureSize)
     * @param featureSize Size of features per primitive
     * @return Result code
     */
    Result setData(
        size_t numPrimitives,
        const float* positions,
        const float* scales,
        const float* orientations,
        const float* densities,
        const float* features,
        size_t featureSize);

    /**
     * @brief Set primitive data from device pointers (no copy)
     *
     * The pointers must remain valid for the lifetime of this object.
     */
    Result setDataFromDevice(
        size_t numPrimitives,
        void* d_positions,
        void* d_scales,
        void* d_orientations,
        void* d_densities,
        void* d_features,
        size_t featureSize);

    /**
     * @brief Update AABBs for acceleration structure
     *
     * Call this after modifying primitive positions/scales.
     */
    Result updateAABBs();

    //--------------------------------------------------------------------------
    // Accessors
    //--------------------------------------------------------------------------

    size_t getNumPrimitives() const { return m_numPrimitives; }
    size_t getFeatureSize() const { return m_featureSize; }

    // Device pointers
    void* getPositionsDevice() const { return m_d_positions; }
    void* getScalesDevice() const { return m_d_scales; }
    void* getOrientationsDevice() const { return m_d_orientations; }
    void* getDensitiesDevice() const { return m_d_densities; }
    void* getFeaturesDevice() const { return m_d_features; }
    void* getAABBsDevice() const { return m_d_aabbs; }

    // Legacy accessors for compatibility
    void* getMeansDevice() const { return m_d_positions; }
    void* getQuatsDevice() const { return m_d_orientations; }

    // Get spherical harmonics degree based on feature size
    uint32_t getSHDegree() const;

    // Check if data is valid
    bool isValid() const { return m_numPrimitives > 0 && m_d_positions != nullptr; }

private:
    Device& m_device;

    size_t m_numPrimitives = 0;
    size_t m_featureSize = 0;

    // Device pointers (owned or external)
    void* m_d_positions = nullptr;
    void* m_d_scales = nullptr;
    void* m_d_orientations = nullptr;
    void* m_d_densities = nullptr;
    void* m_d_features = nullptr;
    void* m_d_aabbs = nullptr;

    // Ownership flags
    bool m_ownsPositions = false;
    bool m_ownsScales = false;
    bool m_ownsOrientations = false;
    bool m_ownsDensities = false;
    bool m_ownsFeatures = false;
    bool m_ownsAABBs = false;

    // Helper to free owned buffers
    void freeOwnedBuffers();
};

// Legacy alias for compatibility
using GaussianPrimitives = PrimitiveSet;

} // namespace gaussian_rt
