// GaussianRT - Acceleration Structure Builder
// BLAS/TLAS construction using slang-rhi
// Apache License 2.0

#pragma once

#include <memory>
#include <vector>
#include "slang-com-ptr.h"
#include "slang-rhi.h"
#include "slang-rhi/acceleration-structure-utils.h"
#include "Types.h"
#include "Device.h"

namespace gaussianrt {

// Build flags for acceleration structures
struct AccelBuildOptions {
    bool allow_compaction = true;
    bool prefer_fast_build = false;  // false = prefer fast trace
    bool allow_anyhit = true;        // Required for volume rendering
};

// Manages BLAS and TLAS for procedural primitives
class AccelerationStructure {
public:
    // Build acceleration structure from AABBs
    static std::unique_ptr<AccelerationStructure> build(
        const Device& device,
        const std::vector<AABB>& aabbs,
        const AccelBuildOptions& options = {});

    // Build from GPU memory
    static std::unique_ptr<AccelerationStructure> build_from_gpu(
        const Device& device,
        const AABB* d_aabbs,
        size_t count,
        const AccelBuildOptions& options = {});

    // Get the top-level acceleration structure for ray tracing
    rhi::IAccelerationStructure* tlas() const { return tlas_.get(); }

    // Get primitive count
    size_t primitive_count() const { return primitive_count_; }

    ~AccelerationStructure() = default;
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

private:
    AccelerationStructure() = default;

    void build_blas(
        const Device& device,
        rhi::IBuffer* aabb_buffer,
        size_t count,
        const AccelBuildOptions& options);

    void build_tlas(const Device& device);

    Slang::ComPtr<rhi::IBuffer> aabb_buffer_;
    Slang::ComPtr<rhi::IBuffer> instance_buffer_;
    Slang::ComPtr<rhi::IAccelerationStructure> blas_;
    Slang::ComPtr<rhi::IAccelerationStructure> tlas_;
    size_t primitive_count_ = 0;
};

} // namespace gaussianrt
