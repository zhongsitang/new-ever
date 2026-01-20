#include "gaussian_rt/PrimitiveSet.h"

#include <cuda_runtime.h>
#include <cstring>

namespace gaussian_rt {

//------------------------------------------------------------------------------
// CUDA kernel for AABB computation
//------------------------------------------------------------------------------

__global__ void computeAABBsKernel(
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ orientations,
    float* __restrict__ aabbMins,
    float* __restrict__ aabbMaxs,
    size_t numPrimitives,
    float scaleFactor = 3.0f)  // 3-sigma coverage for Gaussian falloff
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPrimitives) return;

    // Load position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // Load scale
    float sx = scales[idx * 3 + 0];
    float sy = scales[idx * 3 + 1];
    float sz = scales[idx * 3 + 2];

    // Load quaternion (x, y, z, w)
    float qx = orientations[idx * 4 + 0];
    float qy = orientations[idx * 4 + 1];
    float qz = orientations[idx * 4 + 2];
    float qw = orientations[idx * 4 + 3];

    // Compute rotation matrix columns
    float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    float r01 = 2.0f * (qx * qy - qw * qz);
    float r02 = 2.0f * (qx * qz + qw * qy);

    float r10 = 2.0f * (qx * qy + qw * qz);
    float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    float r12 = 2.0f * (qy * qz - qw * qx);

    float r20 = 2.0f * (qx * qz - qw * qy);
    float r21 = 2.0f * (qy * qz + qw * qx);
    float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    // Scale factors
    sx *= scaleFactor;
    sy *= scaleFactor;
    sz *= scaleFactor;

    // Compute AABB extents using rotated scaled axes
    float ex = fabsf(r00 * sx) + fabsf(r01 * sy) + fabsf(r02 * sz);
    float ey = fabsf(r10 * sx) + fabsf(r11 * sy) + fabsf(r12 * sz);
    float ez = fabsf(r20 * sx) + fabsf(r21 * sy) + fabsf(r22 * sz);

    // Write AABB
    aabbMins[idx * 3 + 0] = px - ex;
    aabbMins[idx * 3 + 1] = py - ey;
    aabbMins[idx * 3 + 2] = pz - ez;

    aabbMaxs[idx * 3 + 0] = px + ex;
    aabbMaxs[idx * 3 + 1] = py + ey;
    aabbMaxs[idx * 3 + 2] = pz + ez;
}

//------------------------------------------------------------------------------
// PrimitiveSet implementation
//------------------------------------------------------------------------------

PrimitiveSet::PrimitiveSet(Device& device)
    : m_device(device) {
}

PrimitiveSet::~PrimitiveSet() {
    freeOwnedBuffers();
}

PrimitiveSet::PrimitiveSet(PrimitiveSet&& other) noexcept
    : m_device(other.m_device)
    , m_numPrimitives(other.m_numPrimitives)
    , m_featureSize(other.m_featureSize)
    , m_d_positions(other.m_d_positions)
    , m_d_scales(other.m_d_scales)
    , m_d_orientations(other.m_d_orientations)
    , m_d_densities(other.m_d_densities)
    , m_d_features(other.m_d_features)
    , m_d_aabbs(other.m_d_aabbs)
    , m_ownsPositions(other.m_ownsPositions)
    , m_ownsScales(other.m_ownsScales)
    , m_ownsOrientations(other.m_ownsOrientations)
    , m_ownsDensities(other.m_ownsDensities)
    , m_ownsFeatures(other.m_ownsFeatures)
    , m_ownsAABBs(other.m_ownsAABBs) {
    // Clear other's pointers to prevent double-free
    other.m_d_positions = nullptr;
    other.m_d_scales = nullptr;
    other.m_d_orientations = nullptr;
    other.m_d_densities = nullptr;
    other.m_d_features = nullptr;
    other.m_d_aabbs = nullptr;
    other.m_numPrimitives = 0;
}

PrimitiveSet& PrimitiveSet::operator=(PrimitiveSet&& other) noexcept {
    if (this != &other) {
        freeOwnedBuffers();

        m_numPrimitives = other.m_numPrimitives;
        m_featureSize = other.m_featureSize;
        m_d_positions = other.m_d_positions;
        m_d_scales = other.m_d_scales;
        m_d_orientations = other.m_d_orientations;
        m_d_densities = other.m_d_densities;
        m_d_features = other.m_d_features;
        m_d_aabbs = other.m_d_aabbs;
        m_ownsPositions = other.m_ownsPositions;
        m_ownsScales = other.m_ownsScales;
        m_ownsOrientations = other.m_ownsOrientations;
        m_ownsDensities = other.m_ownsDensities;
        m_ownsFeatures = other.m_ownsFeatures;
        m_ownsAABBs = other.m_ownsAABBs;

        other.m_d_positions = nullptr;
        other.m_d_scales = nullptr;
        other.m_d_orientations = nullptr;
        other.m_d_densities = nullptr;
        other.m_d_features = nullptr;
        other.m_d_aabbs = nullptr;
        other.m_numPrimitives = 0;
    }
    return *this;
}

void PrimitiveSet::freeOwnedBuffers() {
    if (m_ownsPositions && m_d_positions) m_device.freeBuffer(m_d_positions);
    if (m_ownsScales && m_d_scales) m_device.freeBuffer(m_d_scales);
    if (m_ownsOrientations && m_d_orientations) m_device.freeBuffer(m_d_orientations);
    if (m_ownsDensities && m_d_densities) m_device.freeBuffer(m_d_densities);
    if (m_ownsFeatures && m_d_features) m_device.freeBuffer(m_d_features);
    if (m_ownsAABBs && m_d_aabbs) m_device.freeBuffer(m_d_aabbs);

    m_d_positions = nullptr;
    m_d_scales = nullptr;
    m_d_orientations = nullptr;
    m_d_densities = nullptr;
    m_d_features = nullptr;
    m_d_aabbs = nullptr;
    m_ownsPositions = false;
    m_ownsScales = false;
    m_ownsOrientations = false;
    m_ownsDensities = false;
    m_ownsFeatures = false;
    m_ownsAABBs = false;
}

Result PrimitiveSet::setData(
    size_t numPrimitives,
    const float* positions,
    const float* scales,
    const float* orientations,
    const float* densities,
    const float* features,
    size_t featureSize) {

    if (numPrimitives == 0 || !positions || !scales || !orientations || !densities) {
        return Result::ErrorInvalidArgument;
    }

    freeOwnedBuffers();

    m_numPrimitives = numPrimitives;
    m_featureSize = featureSize;

    // Allocate and upload positions (float3 * N)
    m_d_positions = m_device.createBuffer(numPrimitives * 3 * sizeof(float), positions);
    if (!m_d_positions) return Result::ErrorOutOfMemory;
    m_ownsPositions = true;

    // Allocate and upload scales (float3 * N)
    m_d_scales = m_device.createBuffer(numPrimitives * 3 * sizeof(float), scales);
    if (!m_d_scales) return Result::ErrorOutOfMemory;
    m_ownsScales = true;

    // Allocate and upload orientations (float4 * N)
    m_d_orientations = m_device.createBuffer(numPrimitives * 4 * sizeof(float), orientations);
    if (!m_d_orientations) return Result::ErrorOutOfMemory;
    m_ownsOrientations = true;

    // Allocate and upload densities (float * N)
    m_d_densities = m_device.createBuffer(numPrimitives * sizeof(float), densities);
    if (!m_d_densities) return Result::ErrorOutOfMemory;
    m_ownsDensities = true;

    // Allocate and upload features (float * N * featureSize)
    if (features && featureSize > 0) {
        m_d_features = m_device.createBuffer(numPrimitives * featureSize * sizeof(float), features);
        if (!m_d_features) return Result::ErrorOutOfMemory;
        m_ownsFeatures = true;
    }

    // Update AABBs
    return updateAABBs();
}

Result PrimitiveSet::setDataFromDevice(
    size_t numPrimitives,
    void* d_positions,
    void* d_scales,
    void* d_orientations,
    void* d_densities,
    void* d_features,
    size_t featureSize) {

    if (numPrimitives == 0 || !d_positions || !d_scales || !d_orientations || !d_densities) {
        return Result::ErrorInvalidArgument;
    }

    freeOwnedBuffers();

    m_numPrimitives = numPrimitives;
    m_featureSize = featureSize;

    // Use external pointers (no ownership)
    m_d_positions = d_positions;
    m_d_scales = d_scales;
    m_d_orientations = d_orientations;
    m_d_densities = d_densities;
    m_d_features = d_features;

    // Update AABBs (we own the AABB buffer)
    return updateAABBs();
}

Result PrimitiveSet::updateAABBs() {
    if (m_numPrimitives == 0 || !m_d_positions || !m_d_scales || !m_d_orientations) {
        return Result::ErrorInvalidArgument;
    }

    // Allocate AABB buffer if needed (2 * float3 per primitive for min/max)
    size_t aabbSize = m_numPrimitives * 6 * sizeof(float);
    if (!m_d_aabbs) {
        m_d_aabbs = m_device.createBuffer(aabbSize);
        if (!m_d_aabbs) return Result::ErrorOutOfMemory;
        m_ownsAABBs = true;
    }

    // Launch AABB computation kernel
    int blockSize = 256;
    int numBlocks = (m_numPrimitives + blockSize - 1) / blockSize;

    float* aabbMins = static_cast<float*>(m_d_aabbs);
    float* aabbMaxs = aabbMins + m_numPrimitives * 3;

    computeAABBsKernel<<<numBlocks, blockSize, 0, static_cast<cudaStream_t>(m_device.getCudaStream())>>>(
        static_cast<const float*>(m_d_positions),
        static_cast<const float*>(m_d_scales),
        static_cast<const float*>(m_d_orientations),
        aabbMins,
        aabbMaxs,
        m_numPrimitives
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in computeAABBsKernel: %s\n", cudaGetErrorString(err));
        return Result::ErrorCUDA;
    }

    return Result::Success;
}

uint32_t PrimitiveSet::getSHDegree() const {
    // Infer SH degree from feature size
    // featureSize = (degree+1)^2 * 3
    if (m_featureSize == 3) return 0;   // DC only
    if (m_featureSize == 12) return 1;  // 4 coeffs * 3
    if (m_featureSize == 27) return 2;  // 9 coeffs * 3
    if (m_featureSize == 48) return 3;  // 16 coeffs * 3
    return 0;
}

} // namespace gaussian_rt
