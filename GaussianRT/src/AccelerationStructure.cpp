// GaussianRT - Acceleration Structure Implementation
// Apache License 2.0

#include "AccelerationStructure.h"
#include <stdexcept>
#include <cstring>

#ifndef GAUSSIANRT_NO_CUDA
#include <cuda_runtime.h>
#endif

namespace gaussianrt {

std::unique_ptr<AccelerationStructure> AccelerationStructure::build(
    const Device& device,
    const std::vector<AABB>& aabbs,
    const AccelBuildOptions& options) {

    if (aabbs.empty()) {
        throw std::invalid_argument("Cannot build acceleration structure with no AABBs");
    }

    auto accel = std::unique_ptr<AccelerationStructure>(new AccelerationStructure());
    accel->primitive_count_ = aabbs.size();

    // Create AABB buffer
    rhi::BufferDesc aabb_desc = {};
    aabb_desc.size = aabbs.size() * sizeof(AABB);
    aabb_desc.usage = rhi::BufferUsage::AccelerationStructureBuildInput |
                      rhi::BufferUsage::ShaderResource;
    aabb_desc.defaultState = rhi::ResourceState::AccelerationStructureBuildInput;

    accel->aabb_buffer_ = Slang::ComPtr<rhi::IBuffer>(
        device.get()->createBuffer(aabb_desc, aabbs.data()));
    if (!accel->aabb_buffer_) {
        throw std::runtime_error("Failed to create AABB buffer");
    }

    accel->build_blas(device, accel->aabb_buffer_.get(), aabbs.size(), options);
    accel->build_tlas(device);

    return accel;
}

std::unique_ptr<AccelerationStructure> AccelerationStructure::build_from_gpu(
    const Device& device,
    const AABB* d_aabbs,
    size_t count,
    const AccelBuildOptions& options) {

    if (count == 0 || d_aabbs == nullptr) {
        throw std::invalid_argument("Invalid AABB data");
    }

#ifndef GAUSSIANRT_NO_CUDA
    // Copy from GPU to create proper RHI buffer
    std::vector<AABB> aabbs(count);
    cudaMemcpy(aabbs.data(), d_aabbs, count * sizeof(AABB), cudaMemcpyDeviceToHost);
    return build(device, aabbs, options);
#else
    throw std::runtime_error("CUDA not available for GPU memory access");
#endif
}

void AccelerationStructure::build_blas(
    const Device& device,
    rhi::IBuffer* aabb_buffer,
    size_t count,
    const AccelBuildOptions& options) {

    using namespace rhi;

    // Configure build input for procedural primitives (AABBs)
    AccelerationStructureBuildInput build_input = {};
    build_input.type = AccelerationStructureBuildInputType::ProceduralPrimitives;
    build_input.proceduralPrimitives.aabbBuffer = BufferOffsetPair(aabb_buffer, 0);
    build_input.proceduralPrimitives.primitiveCount = static_cast<uint32_t>(count);
    build_input.proceduralPrimitives.aabbStride = sizeof(AABB);
    build_input.proceduralPrimitives.flags = options.allow_anyhit
        ? AccelerationStructureGeometryFlags::None
        : AccelerationStructureGeometryFlags::Opaque;

    // Configure build descriptor
    AccelerationStructureBuildDesc build_desc = {};
    build_desc.inputs = &build_input;
    build_desc.inputCount = 1;
    build_desc.flags = AccelerationStructureBuildFlags::None;
    if (options.allow_compaction) {
        build_desc.flags = build_desc.flags | AccelerationStructureBuildFlags::AllowCompaction;
    }
    if (options.prefer_fast_build) {
        build_desc.flags = build_desc.flags | AccelerationStructureBuildFlags::PreferFastBuild;
    } else {
        build_desc.flags = build_desc.flags | AccelerationStructureBuildFlags::PreferFastTrace;
    }

    // Query required sizes
    AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.get()->getAccelerationStructureSizes(build_desc, &sizes))) {
        throw std::runtime_error("Failed to query BLAS sizes");
    }

    // Create scratch buffer
    BufferDesc scratch_desc = {};
    scratch_desc.size = sizes.scratchSize;
    scratch_desc.usage = BufferUsage::UnorderedAccess;
    scratch_desc.defaultState = ResourceState::UnorderedAccess;
    auto scratch_buffer = Slang::ComPtr<IBuffer>(device.get()->createBuffer(scratch_desc));

    // Create draft AS for building
    Slang::ComPtr<IAccelerationStructure> draft_as;
    AccelerationStructureDesc draft_desc = {};
    draft_desc.size = sizes.accelerationStructureSize;
    if (SLANG_FAILED(device.get()->createAccelerationStructure(draft_desc, draft_as.writeRef()))) {
        throw std::runtime_error("Failed to create draft BLAS");
    }

    // Create query pool for compaction
    Slang::ComPtr<IQueryPool> query_pool;
    if (options.allow_compaction) {
        QueryPoolDesc query_desc = {};
        query_desc.count = 1;
        query_desc.type = QueryType::AccelerationStructureCompactedSize;
        if (SLANG_FAILED(device.get()->createQueryPool(query_desc, query_pool.writeRef()))) {
            throw std::runtime_error("Failed to create query pool");
        }
        query_pool->reset();
    }

    // Build the acceleration structure
    auto encoder = device.create_command_encoder();
    AccelerationStructureQueryDesc query_desc = {};
    if (options.allow_compaction) {
        query_desc.queryPool = query_pool.get();
        query_desc.queryType = QueryType::AccelerationStructureCompactedSize;
    }
    encoder->buildAccelerationStructure(
        build_desc, draft_as.get(), nullptr,
        BufferOffsetPair(scratch_buffer.get(), 0),
        options.allow_compaction ? 1 : 0,
        options.allow_compaction ? &query_desc : nullptr);
    device.submit_and_wait(encoder.get());

    // Compact if requested
    if (options.allow_compaction) {
        uint64_t compacted_size = 0;
        query_pool->getResult(0, 1, &compacted_size);

        AccelerationStructureDesc compact_desc = {};
        compact_desc.size = compacted_size;
        if (SLANG_FAILED(device.get()->createAccelerationStructure(compact_desc, blas_.writeRef()))) {
            throw std::runtime_error("Failed to create compacted BLAS");
        }

        encoder = device.create_command_encoder();
        encoder->copyAccelerationStructure(blas_.get(), draft_as.get(), AccelerationStructureCopyMode::Compact);
        device.submit_and_wait(encoder.get());
    } else {
        blas_ = draft_as;
    }
}

void AccelerationStructure::build_tlas(const Device& device) {
    using namespace rhi;

    // Get native instance descriptor type for this device
    AccelerationStructureInstanceDescType native_type =
        getAccelerationStructureInstanceDescType(device.get());
    Size native_size = getAccelerationStructureInstanceDescSize(native_type);

    // Create instance descriptor (identity transform, single instance)
    AccelerationStructureInstanceDescGeneric instance = {};
    float transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    std::memcpy(&instance.transform[0][0], transform, sizeof(transform));
    instance.instanceID = 0;
    instance.instanceMask = 0xFF;
    instance.instanceContributionToHitGroupIndex = 0;
    instance.flags = AccelerationStructureInstanceFlags::TriangleFacingCullDisable;
    instance.accelerationStructure = blas_->getHandle();

    // Convert to native format
    std::vector<uint8_t> native_data(native_size);
    convertAccelerationStructureInstanceDescs(
        1, native_type, native_data.data(), native_size,
        &instance, sizeof(instance));

    // Create instance buffer
    BufferDesc instance_desc = {};
    instance_desc.size = native_size;
    instance_desc.usage = BufferUsage::ShaderResource;
    instance_desc.defaultState = ResourceState::ShaderResource;
    instance_buffer_ = Slang::ComPtr<IBuffer>(
        device.get()->createBuffer(instance_desc, native_data.data()));

    // Configure TLAS build
    AccelerationStructureBuildInput build_input = {};
    build_input.type = AccelerationStructureBuildInputType::Instances;
    build_input.instances.instanceBuffer = BufferOffsetPair(instance_buffer_.get(), 0);
    build_input.instances.instanceCount = 1;
    build_input.instances.instanceStride = native_size;

    AccelerationStructureBuildDesc build_desc = {};
    build_desc.inputs = &build_input;
    build_desc.inputCount = 1;

    // Query sizes
    AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.get()->getAccelerationStructureSizes(build_desc, &sizes))) {
        throw std::runtime_error("Failed to query TLAS sizes");
    }

    // Create scratch buffer
    BufferDesc scratch_desc = {};
    scratch_desc.size = sizes.scratchSize;
    scratch_desc.usage = BufferUsage::UnorderedAccess;
    scratch_desc.defaultState = ResourceState::UnorderedAccess;
    auto scratch_buffer = Slang::ComPtr<IBuffer>(device.get()->createBuffer(scratch_desc));

    // Create TLAS
    AccelerationStructureDesc tlas_desc = {};
    tlas_desc.size = sizes.accelerationStructureSize;
    if (SLANG_FAILED(device.get()->createAccelerationStructure(tlas_desc, tlas_.writeRef()))) {
        throw std::runtime_error("Failed to create TLAS");
    }

    // Build TLAS
    auto encoder = device.create_command_encoder();
    encoder->buildAccelerationStructure(
        build_desc, tlas_.get(), nullptr,
        BufferOffsetPair(scratch_buffer.get(), 0),
        0, nullptr);
    device.submit_and_wait(encoder.get());
}

} // namespace gaussianrt
