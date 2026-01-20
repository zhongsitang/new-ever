#pragma once

#include "Types.h"
#include "Device.h"
#include <vector>

namespace gaussian_rt {

/**
 * @brief Manages Gaussian primitive data on GPU
 *
 * Stores and manages the 3D Gaussian splat primitives including:
 * - Position (mean)
 * - Scale
 * - Rotation (quaternion)
 * - Density/opacity
 * - Features (RGB or spherical harmonics coefficients)
 */
class GaussianPrimitives {
public:
    GaussianPrimitives(Device& device);
    ~GaussianPrimitives();

    // Non-copyable
    GaussianPrimitives(const GaussianPrimitives&) = delete;
    GaussianPrimitives& operator=(const GaussianPrimitives&) = delete;

    // Movable
    GaussianPrimitives(GaussianPrimitives&&) noexcept;
    GaussianPrimitives& operator=(GaussianPrimitives&&) noexcept;

    /**
     * @brief Set primitive data from raw pointers
     *
     * @param numPrimitives Number of primitives
     * @param means Center positions (float3 * N)
     * @param scales Scale factors (float3 * N)
     * @param quats Rotation quaternions (float4 * N)
     * @param densities Opacity values (float * N)
     * @param features Color/SH features (float * N * featureSize)
     * @param featureSize Size of features per primitive
     * @return Result code
     */
    Result setData(
        size_t numPrimitives,
        const float* means,
        const float* scales,
        const float* quats,
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
        void* d_means,
        void* d_scales,
        void* d_quats,
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
    void* getMeansDevice() const { return m_d_means; }
    void* getScalesDevice() const { return m_d_scales; }
    void* getQuatsDevice() const { return m_d_quats; }
    void* getDensitiesDevice() const { return m_d_densities; }
    void* getFeaturesDevice() const { return m_d_features; }
    void* getAABBsDevice() const { return m_d_aabbs; }

    // Get spherical harmonics degree based on feature size
    uint32_t getSHDegree() const;

    // Check if data is valid
    bool isValid() const { return m_numPrimitives > 0 && m_d_means != nullptr; }

private:
    Device& m_device;

    size_t m_numPrimitives = 0;
    size_t m_featureSize = 0;

    // Device pointers (owned or external)
    void* m_d_means = nullptr;
    void* m_d_scales = nullptr;
    void* m_d_quats = nullptr;
    void* m_d_densities = nullptr;
    void* m_d_features = nullptr;
    void* m_d_aabbs = nullptr;

    // Ownership flags
    bool m_ownsMeans = false;
    bool m_ownsScales = false;
    bool m_ownsQuats = false;
    bool m_ownsDensities = false;
    bool m_ownsFeatures = false;
    bool m_ownsAABBs = false;

    // Helper to free owned buffers
    void freeOwnedBuffers();
};

} // namespace gaussian_rt
