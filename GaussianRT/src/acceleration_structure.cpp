#include "acceleration_structure.h"
#include <cstring>
#include <stdexcept>

namespace gaussian_rt {

AccelerationStructure::AccelerationStructure(Device& device)
    : device_(device) {}

AccelerationStructure::~AccelerationStructure() = default;

void AccelerationStructure::build(
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    uint32_t num_elements,
    bool fast_build
) {
    positions_ = positions;
    scales_ = scales;
    rotations_ = rotations;
    num_elements_ = num_elements;
    allow_update_ = !fast_build;

    // Create AABB buffer
    size_t aabb_size = num_elements * sizeof(AABB);
    aabb_buffer_ = device_.create_buffer(
        aabb_size,
        rhi::BufferUsage::AccelerationStructureBuildInput |
        rhi::BufferUsage::ShaderResource |
        rhi::BufferUsage::UnorderedAccess,
        rhi::MemoryType::DeviceLocal
    );

    if (!aabb_buffer_) {
        throw std::runtime_error("Failed to create AABB buffer");
    }

    // Compute AABBs on GPU
    AABB* aabb_ptr = nullptr;
    aabb_buffer_->map(nullptr, reinterpret_cast<void**>(&aabb_ptr));
    if (aabb_ptr) {
        compute_aabbs(aabb_ptr);
        aabb_buffer_->unmap(nullptr);
    } else {
        // Use separate computation with GPU buffer
        std::vector<AABB> host_aabbs(num_elements);
        compute_aabbs(host_aabbs.data());

        // Copy to GPU
        auto staging = device_.create_buffer(
            aabb_size,
            rhi::BufferUsage::CopySource,
            rhi::MemoryType::Upload,
            host_aabbs.data()
        );

        // Create command buffer and copy
        Slang::ComPtr<rhi::ICommandEncoder> encoder;
        device_.get_queue()->createCommandEncoder(encoder.writeRef());
        encoder->copyBuffer(aabb_buffer_, 0, staging, 0, aabb_size);

        Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
        encoder->finish(cmd_buffer.writeRef());
        device_.submit_and_wait(cmd_buffer);
    }

    // Build acceleration structures
    build_blas(fast_build);
    build_tlas(fast_build);

    built_ = true;
}

void AccelerationStructure::rebuild() {
    if (!built_) {
        throw std::runtime_error("Acceleration structure must be built first");
    }

    // Recompute AABBs and rebuild
    build(positions_, scales_, rotations_, num_elements_, !allow_update_);
}

void AccelerationStructure::update() {
    if (!built_ || !allow_update_) {
        throw std::runtime_error("Acceleration structure cannot be updated");
    }

    // Recompute AABBs
    std::vector<AABB> host_aabbs(num_elements_);
    compute_aabbs(host_aabbs.data());

    // Update AABB buffer
    size_t aabb_size = num_elements_ * sizeof(AABB);
    auto staging = device_.create_buffer(
        aabb_size,
        rhi::BufferUsage::CopySource,
        rhi::MemoryType::Upload,
        host_aabbs.data()
    );

    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    device_.get_queue()->createCommandEncoder(encoder.writeRef());
    encoder->copyBuffer(aabb_buffer_, 0, staging, 0, aabb_size);

    Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
    encoder->finish(cmd_buffer.writeRef());
    device_.submit_and_wait(cmd_buffer);

    // Refit BLAS (update operation)
    // Note: RHI may need extension for refit support
    // For now, rebuild
    build_blas(true);
    build_tlas(true);
}

void AccelerationStructure::compute_aabbs(AABB* output) {
    // Launch CUDA kernel for AABB computation
    launch_compute_aabbs(
        positions_,
        scales_,
        rotations_,
        output,
        num_elements_,
        nullptr  // default stream
    );

    cudaDeviceSynchronize();
}

void AccelerationStructure::build_blas(bool fast_build) {
    auto rhi_device = device_.get_device();

    // Setup AABB geometry input
    rhi::AccelerationStructureBuildInput build_input = {};
    build_input.type = rhi::AccelerationStructureBuildInputType::ProceduralPrimitives;
    build_input.proceduralPrimitives.aabbBuffer = aabb_buffer_;
    build_input.proceduralPrimitives.aabbBufferOffset = 0;
    build_input.proceduralPrimitives.aabbBufferStride = sizeof(AABB);
    build_input.proceduralPrimitives.primitiveCount = num_elements_;
    build_input.proceduralPrimitives.flags = rhi::AccelerationStructureGeometryFlags::NoDuplicateAnyHitInvocation;

    // Query prebuild info
    rhi::AccelerationStructurePrebuildInfo prebuild_info = {};
    rhi_device->getAccelerationStructurePrebuildInfo(
        &build_input, 1,
        fast_build ? rhi::AccelerationStructureBuildFlags::PreferFastBuild
                   : rhi::AccelerationStructureBuildFlags::PreferFastTrace,
        &prebuild_info
    );

    // Create scratch buffer
    blas_scratch_buffer_ = device_.create_buffer(
        prebuild_info.scratchDataSize,
        rhi::BufferUsage::UnorderedAccess,
        rhi::MemoryType::DeviceLocal
    );

    // Create BLAS
    rhi::AccelerationStructureDesc as_desc = {};
    as_desc.type = rhi::AccelerationStructureType::BottomLevel;
    as_desc.size = prebuild_info.resultDataMaxSize;

    if (SLANG_FAILED(rhi_device->createAccelerationStructure(as_desc, blas_.writeRef()))) {
        throw std::runtime_error("Failed to create BLAS");
    }

    // Build BLAS
    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    device_.get_queue()->createCommandEncoder(encoder.writeRef());

    auto ray_tracing_encoder = encoder->beginRayTracingPass();
    ray_tracing_encoder->buildAccelerationStructure(
        blas_,
        &build_input, 1,
        fast_build ? rhi::AccelerationStructureBuildFlags::PreferFastBuild
                   : rhi::AccelerationStructureBuildFlags::PreferFastTrace,
        blas_scratch_buffer_
    );
    ray_tracing_encoder->end();

    Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
    encoder->finish(cmd_buffer.writeRef());
    device_.submit_and_wait(cmd_buffer);
}

void AccelerationStructure::build_tlas(bool fast_build) {
    auto rhi_device = device_.get_device();

    // Create instance descriptor
    rhi::AccelerationStructureInstanceDescGeneric instance_desc = {};
    instance_desc.accelerationStructure = blas_->getDeviceAddress();
    instance_desc.instanceID = 0;
    instance_desc.instanceMask = 0xFF;
    instance_desc.instanceContributionToHitGroupIndex = 0;
    instance_desc.flags = rhi::AccelerationStructureInstanceFlags::None;

    // Identity transform
    instance_desc.transform[0][0] = 1.0f;
    instance_desc.transform[1][1] = 1.0f;
    instance_desc.transform[2][2] = 1.0f;

    // Create instance buffer
    instance_buffer_ = device_.create_buffer(
        sizeof(rhi::AccelerationStructureInstanceDescGeneric),
        rhi::BufferUsage::AccelerationStructureBuildInput,
        rhi::MemoryType::DeviceLocal,
        &instance_desc
    );

    // Setup instance input
    rhi::AccelerationStructureBuildInput build_input = {};
    build_input.type = rhi::AccelerationStructureBuildInputType::Instances;
    build_input.instances.instanceBuffer = instance_buffer_;
    build_input.instances.instanceBufferOffset = 0;
    build_input.instances.instanceCount = 1;

    // Query prebuild info
    rhi::AccelerationStructurePrebuildInfo prebuild_info = {};
    rhi_device->getAccelerationStructurePrebuildInfo(
        &build_input, 1,
        fast_build ? rhi::AccelerationStructureBuildFlags::PreferFastBuild
                   : rhi::AccelerationStructureBuildFlags::PreferFastTrace,
        &prebuild_info
    );

    // Create scratch buffer
    tlas_scratch_buffer_ = device_.create_buffer(
        prebuild_info.scratchDataSize,
        rhi::BufferUsage::UnorderedAccess,
        rhi::MemoryType::DeviceLocal
    );

    // Create TLAS
    rhi::AccelerationStructureDesc as_desc = {};
    as_desc.type = rhi::AccelerationStructureType::TopLevel;
    as_desc.size = prebuild_info.resultDataMaxSize;

    if (SLANG_FAILED(rhi_device->createAccelerationStructure(as_desc, tlas_.writeRef()))) {
        throw std::runtime_error("Failed to create TLAS");
    }

    // Build TLAS
    Slang::ComPtr<rhi::ICommandEncoder> encoder;
    device_.get_queue()->createCommandEncoder(encoder.writeRef());

    auto ray_tracing_encoder = encoder->beginRayTracingPass();
    ray_tracing_encoder->buildAccelerationStructure(
        tlas_,
        &build_input, 1,
        fast_build ? rhi::AccelerationStructureBuildFlags::PreferFastBuild
                   : rhi::AccelerationStructureBuildFlags::PreferFastTrace,
        tlas_scratch_buffer_
    );
    ray_tracing_encoder->end();

    Slang::ComPtr<rhi::ICommandBuffer> cmd_buffer;
    encoder->finish(cmd_buffer.writeRef());
    device_.submit_and_wait(cmd_buffer);
}

} // namespace gaussian_rt
