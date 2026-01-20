#pragma once

#include "device.h"
#include "types.h"
#include <slang-rhi.h>
#include <slang-com-ptr.h>
#include <vector>

namespace gaussian_rt {

// Axis-Aligned Bounding Box
struct AABB {
    float3 min_bound;
    float3 max_bound;
};

// Acceleration structure for volume elements
class AccelerationStructure {
public:
    AccelerationStructure(Device& device);
    ~AccelerationStructure();

    // Build acceleration structure from ellipsoid elements
    void build(
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        uint32_t num_elements,
        bool fast_build = false
    );

    // Rebuild with updated element positions (for dynamic scenes)
    void rebuild();

    // Update (refit) acceleration structure for small changes
    void update();

    // Get RHI acceleration structure handle
    rhi::IAccelerationStructure* get_handle() const { return tlas_.get(); }

    // Get AABB buffer
    rhi::IBuffer* get_aabb_buffer() const { return aabb_buffer_.get(); }

    uint32_t get_num_elements() const { return num_elements_; }

private:
    Device& device_;

    // Acceleration structures
    Slang::ComPtr<rhi::IAccelerationStructure> blas_;  // Bottom-level (geometry)
    Slang::ComPtr<rhi::IAccelerationStructure> tlas_;  // Top-level (instances)

    // Buffers
    Slang::ComPtr<rhi::IBuffer> aabb_buffer_;
    Slang::ComPtr<rhi::IBuffer> blas_scratch_buffer_;
    Slang::ComPtr<rhi::IBuffer> tlas_scratch_buffer_;
    Slang::ComPtr<rhi::IBuffer> instance_buffer_;

    // Element data references
    const float3* positions_ = nullptr;
    const float3* scales_ = nullptr;
    const float4* rotations_ = nullptr;
    uint32_t num_elements_ = 0;

    bool built_ = false;
    bool allow_update_ = false;

    void compute_aabbs(AABB* output);
    void build_blas(bool fast_build);
    void build_tlas(bool fast_build);
};

// CUDA kernel to compute AABBs from ellipsoid parameters
void launch_compute_aabbs(
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    AABB* aabbs,
    uint32_t num_elements,
    cudaStream_t stream = nullptr
);

} // namespace gaussian_rt
