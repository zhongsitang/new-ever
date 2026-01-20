// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "GAS.h"

#include <chrono>
#include <stdexcept>

#include "exception.h"

using namespace rhi;

GAS::GAS(GAS &&other) noexcept
    : device(std::exchange(other.device, nullptr)),
      queue(std::exchange(other.queue, nullptr)),
      aabb_buffer(std::exchange(other.aabb_buffer, nullptr)),
      instance_buffer(std::exchange(other.instance_buffer, nullptr)),
      blas(std::exchange(other.blas, nullptr)),
      tlas(std::exchange(other.tlas, nullptr)),
      device_index(std::exchange(other.device_index, -1)),
      enable_anyhit(std::exchange(other.enable_anyhit, false)),
      fast_build(std::exchange(other.fast_build, false)) {}

void GAS::release() {
  aabb_buffer = nullptr;
  instance_buffer = nullptr;
  blas = nullptr;
  tlas = nullptr;
}

GAS::~GAS() noexcept(false) {
  release();
  device = nullptr;
  queue = nullptr;
}

void GAS::build(const Primitives &model) {
  release();
  CUDA_CHECK(cudaSetDevice(device_index));

  std::vector<OptixAabb> aabbs(model.num_prims);
  CUDA_CHECK(cudaMemcpy(aabbs.data(), model.aabbs,
                        model.num_prims * sizeof(OptixAabb), cudaMemcpyDeviceToHost));

  BufferDesc aabb_buffer_desc = {};
  aabb_buffer_desc.size = aabbs.size() * sizeof(OptixAabb);
  aabb_buffer_desc.defaultState = ResourceState::AccelerationStructureBuildInput;
  aabb_buffer_desc.usage =
      BufferUsage::AccelerationStructureBuildInput | BufferUsage::ShaderResource;
  aabb_buffer = device->createBuffer(aabb_buffer_desc, aabbs.data());

  AccelerationStructureBuildInput build_input = {};
  build_input.type = AccelerationStructureBuildInputType::ProceduralPrimitives;
  build_input.proceduralPrimitives.aabbBuffer = BufferOffsetPair(aabb_buffer, 0);
  build_input.proceduralPrimitives.primitiveCount = model.num_prims;
  build_input.proceduralPrimitives.aabbStride = sizeof(OptixAabb);
  build_input.proceduralPrimitives.flags =
      enable_anyhit ? AccelerationStructureGeometryFlags::None
                    : AccelerationStructureGeometryFlags::Opaque;

  AccelerationStructureBuildDesc build_desc = {};
  build_desc.inputs = &build_input;
  build_desc.inputCount = 1;
  build_desc.flags = AccelerationStructureBuildFlags::AllowCompaction |
                     (fast_build ? AccelerationStructureBuildFlags::PreferFastBuild
                                 : AccelerationStructureBuildFlags::PreferFastTrace);

  AccelerationStructureSizes sizes;
  if (SLANG_FAILED(device->getAccelerationStructureSizes(build_desc, &sizes))) {
    throw std::runtime_error("Failed to query BLAS build sizes.");
  }

  BufferDesc scratch_buffer_desc = {};
  scratch_buffer_desc.defaultState = ResourceState::UnorderedAccess;
  scratch_buffer_desc.size = sizes.scratchSize;
  scratch_buffer_desc.usage = BufferUsage::UnorderedAccess;
  auto scratch_buffer = device->createBuffer(scratch_buffer_desc);

  ComPtr<IQueryPool> compacted_size_query;
  QueryPoolDesc query_pool_desc = {};
  query_pool_desc.count = 1;
  query_pool_desc.type = QueryType::AccelerationStructureCompactedSize;
  if (SLANG_FAILED(device->createQueryPool(query_pool_desc, compacted_size_query.writeRef()))) {
    throw std::runtime_error("Failed to create acceleration structure query pool.");
  }

  ComPtr<IAccelerationStructure> draft_as;
  AccelerationStructureDesc draft_create_desc = {};
  draft_create_desc.size = sizes.accelerationStructureSize;
  if (SLANG_FAILED(device->createAccelerationStructure(draft_create_desc, draft_as.writeRef()))) {
    throw std::runtime_error("Failed to create draft BLAS.");
  }

  compacted_size_query->reset();
  auto command_encoder = queue->createCommandEncoder();
  AccelerationStructureQueryDesc compacted_query_desc = {};
  compacted_query_desc.queryPool = compacted_size_query;
  compacted_query_desc.queryType = QueryType::AccelerationStructureCompactedSize;
  command_encoder->buildAccelerationStructure(
      build_desc, draft_as, nullptr, BufferOffsetPair(scratch_buffer, 0), 1,
      &compacted_query_desc);
  queue->submit(command_encoder->finish());
  queue->waitOnHost();

  uint64_t compacted_size = 0;
  compacted_size_query->getResult(0, 1, &compacted_size);

  AccelerationStructureDesc create_desc = {};
  create_desc.size = compacted_size;
  if (SLANG_FAILED(device->createAccelerationStructure(create_desc, blas.writeRef()))) {
    throw std::runtime_error("Failed to create compacted BLAS.");
  }

  command_encoder = queue->createCommandEncoder();
  command_encoder->copyAccelerationStructure(blas, draft_as,
                                             AccelerationStructureCopyMode::Compact);
  queue->submit(command_encoder->finish());
  queue->waitOnHost();

  AccelerationStructureInstanceDescType native_instance_desc_type =
      getAccelerationStructureInstanceDescType(device);
  Size native_instance_desc_size =
      getAccelerationStructureInstanceDescSize(native_instance_desc_type);

  std::vector<AccelerationStructureInstanceDescGeneric> instance_descs(1);
  float transform_matrix[] =
      {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  memcpy(&instance_descs[0].transform[0][0], transform_matrix, sizeof(float) * 12);
  instance_descs[0].instanceID = 0;
  instance_descs[0].instanceMask = 0xFF;
  instance_descs[0].instanceContributionToHitGroupIndex = 0;
  instance_descs[0].flags = AccelerationStructureInstanceFlags::TriangleFacingCullDisable;
  instance_descs[0].accelerationStructure = blas->getHandle();

  std::vector<uint8_t> native_instance_descs(instance_descs.size() * native_instance_desc_size);
  convertAccelerationStructureInstanceDescs(
      instance_descs.size(),
      native_instance_desc_type,
      native_instance_descs.data(),
      native_instance_desc_size,
      instance_descs.data(),
      sizeof(AccelerationStructureInstanceDescGeneric));

  BufferDesc instance_buffer_desc = {};
  instance_buffer_desc.size = native_instance_descs.size();
  instance_buffer_desc.defaultState = ResourceState::ShaderResource;
  instance_buffer_desc.usage = BufferUsage::ShaderResource;
  instance_buffer = device->createBuffer(instance_buffer_desc, native_instance_descs.data());

  AccelerationStructureBuildInput tlas_build_input = {};
  tlas_build_input.type = AccelerationStructureBuildInputType::Instances;
  tlas_build_input.instances.instanceBuffer = BufferOffsetPair(instance_buffer, 0);
  tlas_build_input.instances.instanceCount = 1;
  tlas_build_input.instances.instanceStride = native_instance_desc_size;

  AccelerationStructureBuildDesc tlas_build_desc = {};
  tlas_build_desc.inputs = &tlas_build_input;
  tlas_build_desc.inputCount = 1;

  AccelerationStructureSizes tlas_sizes;
  if (SLANG_FAILED(device->getAccelerationStructureSizes(tlas_build_desc, &tlas_sizes))) {
    throw std::runtime_error("Failed to query TLAS build sizes.");
  }

  BufferDesc tlas_scratch_desc = {};
  tlas_scratch_desc.defaultState = ResourceState::UnorderedAccess;
  tlas_scratch_desc.size = tlas_sizes.scratchSize;
  tlas_scratch_desc.usage = BufferUsage::UnorderedAccess;
  auto tlas_scratch_buffer = device->createBuffer(tlas_scratch_desc);

  AccelerationStructureDesc tlas_create_desc = {};
  tlas_create_desc.size = tlas_sizes.accelerationStructureSize;
  if (SLANG_FAILED(device->createAccelerationStructure(tlas_create_desc, tlas.writeRef()))) {
    throw std::runtime_error("Failed to create TLAS.");
  }

  command_encoder = queue->createCommandEncoder();
  command_encoder->buildAccelerationStructure(
      tlas_build_desc, tlas, nullptr, BufferOffsetPair(tlas_scratch_buffer, 0), 0, nullptr);
  queue->submit(command_encoder->finish());
  queue->waitOnHost();
}
