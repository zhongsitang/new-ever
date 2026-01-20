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

#include "Forward.h"

#include <filesystem>
#include <stdexcept>

#include "exception.h"
#include "initialize_density.h"
#include "slang-com-ptr.h"
#include "slang-rhi/shader-cursor.h"
#include "slang.h"

using namespace rhi;
using namespace Slang;

namespace {
constexpr size_t kPayloadSlotCount = 2 * 16;
constexpr size_t kPayloadSize = sizeof(uint32_t) * kPayloadSlotCount;

std::string shader_path_for_mode(bool enable_backward) {
  const std::filesystem::path base_path = std::filesystem::path(__FILE__).parent_path();
  const char* shader_name = enable_backward ? "shaders.slang" : "fast_shaders.slang";
  return (base_path / "slang" / shader_name).string();
}
}  // namespace

Forward::Forward(IDevice *device,
                 ICommandQueue *queue,
                 int8_t device_index,
                 const Primitives &model,
                 const bool enable_backward)
    : enable_backward(enable_backward),
      rhi_device(device),
      queue(queue),
      device_index(device_index),
      model(&model) {
  if (!rhi_device || !queue) {
    throw std::invalid_argument("Forward requires a valid slang-rhi device and queue.");
  }
  CUDA_CHECK(cudaSetDevice(device_index));
  update_model_buffers(model);
  if (SLANG_FAILED(load_shader_program(shader_program.writeRef()))) {
    throw std::runtime_error("Failed to load Slang ray tracing shader program.");
  }
  if (SLANG_FAILED(create_ray_tracing_pipeline())) {
    throw std::runtime_error("Failed to create Slang ray tracing pipeline.");
  }
}

Slang::Result Forward::load_shader_program(IShaderProgram **out_program) const {
  ComPtr<slang::ISession> slang_session = rhi_device->getSlangSession();
  ComPtr<slang::IBlob> diagnostics_blob;

  std::string shader_path = shader_path_for_mode(enable_backward);
  slang::IModule *module =
      slang_session->loadModule(shader_path.c_str(), diagnostics_blob.writeRef());
  if (diagnostics_blob) {
    printf("%s", static_cast<const char *>(diagnostics_blob->getBufferPointer()));
  }
  if (!module) {
    return SLANG_FAIL;
  }

  Slang::List<slang::IComponentType *> component_types;
  component_types.add(module);

  ComPtr<slang::IEntryPoint> entry_point;
  SLANG_RETURN_ON_FAIL(
      module->findEntryPointByName("rayGenShader", entry_point.writeRef()));
  component_types.add(entry_point);
  SLANG_RETURN_ON_FAIL(
      module->findEntryPointByName("missShader", entry_point.writeRef()));
  component_types.add(entry_point);
  SLANG_RETURN_ON_FAIL(
      module->findEntryPointByName("anyHitShader", entry_point.writeRef()));
  component_types.add(entry_point);
  SLANG_RETURN_ON_FAIL(
      module->findEntryPointByName("intersectionShader", entry_point.writeRef()));
  component_types.add(entry_point);

  ComPtr<slang::IComponentType> linked_program;
  SlangResult result = slang_session->createCompositeComponentType(
      component_types.getBuffer(),
      component_types.getCount(),
      linked_program.writeRef(),
      diagnostics_blob.writeRef());
  if (diagnostics_blob) {
    printf("%s", static_cast<const char *>(diagnostics_blob->getBufferPointer()));
  }
  SLANG_RETURN_ON_FAIL(result);

  ShaderProgramDesc program_desc = {};
  program_desc.slangGlobalScope = linked_program;
  SLANG_RETURN_ON_FAIL(rhi_device->createShaderProgram(program_desc, out_program));

  return SLANG_OK;
}

Slang::Result Forward::create_ray_tracing_pipeline() {
  const char *hit_group_name = "ellipsoidHitGroup";
  HitGroupDesc hit_group = {};
  hit_group.hitGroupName = hit_group_name;
  hit_group.anyHitEntryPoint = "anyHitShader";
  hit_group.intersectionEntryPoint = "intersectionShader";

  RayTracingPipelineDesc pipeline_desc = {};
  pipeline_desc.program = shader_program;
  pipeline_desc.hitGroupCount = 1;
  pipeline_desc.hitGroups = &hit_group;
  pipeline_desc.maxRayPayloadSize = kPayloadSize;
  pipeline_desc.maxRecursion = 1;
  SLANG_RETURN_ON_FAIL(
      rhi_device->createRayTracingPipeline(pipeline_desc, pipeline.writeRef()));

  ShaderTableDesc shader_table_desc = {};
  const char *raygen_name = "rayGenShader";
  const char *miss_name = "missShader";
  shader_table_desc.program = shader_program;
  shader_table_desc.hitGroupCount = 1;
  shader_table_desc.hitGroupNames = &hit_group_name;
  shader_table_desc.rayGenShaderCount = 1;
  shader_table_desc.rayGenShaderEntryPointNames = &raygen_name;
  shader_table_desc.missShaderCount = 1;
  shader_table_desc.missShaderEntryPointNames = &miss_name;
  SLANG_RETURN_ON_FAIL(
      rhi_device->createShaderTable(shader_table_desc, shader_table.writeRef()));

  return SLANG_OK;
}

Slang::ComPtr<IBuffer> Forward::create_buffer_from_cuda_pointer(
    void *pointer,
    size_t size,
    size_t element_size,
    BufferUsage usage,
    ResourceState default_state) const {
  if (!pointer || size == 0) {
    return nullptr;
  }
  BufferDesc buffer_desc = {};
  buffer_desc.size = size;
  buffer_desc.elementSize = element_size;
  buffer_desc.usage = usage;
  buffer_desc.defaultState = default_state;

  NativeHandle native_handle = {};
  native_handle.type = NativeHandleType::CUDADevicePointer;
  native_handle.value = reinterpret_cast<uint64_t>(pointer);

  ComPtr<IBuffer> buffer;
  if (SLANG_FAILED(
          rhi_device->createBufferFromNativeHandle(native_handle, buffer_desc, buffer.writeRef()))) {
    return nullptr;
  }
  return buffer;
}

void Forward::update_model_buffers(const Primitives &model) {
  CUDA_CHECK(cudaSetDevice(device_index));

  std::vector<float3> mean_data(model.num_prims);
  std::vector<float3> scale_data(model.num_prims);
  std::vector<float4> quat_data(model.num_prims);
  std::vector<float> density_data(model.num_prims);
  std::vector<float> feature_data(model.num_prims * model.feature_size);

  CUDA_CHECK(cudaMemcpy(mean_data.data(), model.means,
                        model.num_prims * sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scale_data.data(), model.scales,
                        model.num_prims * sizeof(float3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(quat_data.data(), model.quats,
                        model.num_prims * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(density_data.data(), model.densities,
                        model.num_prims * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(feature_data.data(), model.features,
                        model.num_prims * model.feature_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  BufferDesc buffer_desc = {};
  buffer_desc.usage = BufferUsage::ShaderResource;
  buffer_desc.defaultState = ResourceState::ShaderResource;

  buffer_desc.size = mean_data.size() * sizeof(float3);
  buffer_desc.elementSize = sizeof(float3);
  means_buffer = rhi_device->createBuffer(buffer_desc, mean_data.data());
  if (!means_buffer) {
    throw std::runtime_error("Failed to create means buffer.");
  }

  buffer_desc.size = scale_data.size() * sizeof(float3);
  scales_buffer = rhi_device->createBuffer(buffer_desc, scale_data.data());
  if (!scales_buffer) {
    throw std::runtime_error("Failed to create scales buffer.");
  }

  buffer_desc.size = quat_data.size() * sizeof(float4);
  buffer_desc.elementSize = sizeof(float4);
  quats_buffer = rhi_device->createBuffer(buffer_desc, quat_data.data());
  if (!quats_buffer) {
    throw std::runtime_error("Failed to create quaternions buffer.");
  }

  buffer_desc.size = density_data.size() * sizeof(float);
  buffer_desc.elementSize = sizeof(float);
  densities_buffer = rhi_device->createBuffer(buffer_desc, density_data.data());
  if (!densities_buffer) {
    throw std::runtime_error("Failed to create densities buffer.");
  }

  buffer_desc.size = feature_data.size() * sizeof(float);
  buffer_desc.elementSize = sizeof(float);
  features_buffer = rhi_device->createBuffer(buffer_desc, feature_data.data());
  if (!features_buffer) {
    throw std::runtime_error("Failed to create features buffer.");
  }

  params.means.data = reinterpret_cast<float3 *>(model.means);
  params.means.size = model.num_prims;
  params.scales.data = reinterpret_cast<float3 *>(model.scales);
  params.scales.size = model.num_prims;
  params.quats.data = reinterpret_cast<float4 *>(model.quats);
  params.quats.size = model.num_prims;
  params.densities.data = model.densities;
  params.densities.size = model.num_prims;
  params.features.data = model.features;
  params.features.size = model.num_prims * model.feature_size;
  num_prims = model.num_prims;
}

void Forward::trace_rays(rhi::IAccelerationStructure *top_level_as,
                         const size_t ray_count,
                         float3 *ray_origins,
                         float3 *ray_directions,
                         void *output_image,
                         uint sh_deg,
                         float tmin,
                         float tmax,
                         float4 *initial_drgb,
                         Cam *camera,
                         const size_t max_iters,
                         const float max_prim_size,
                         uint *iteration_counts,
                         uint *last_faces,
                         uint *touch_counts,
                         float4 *last_diracs,
                         SplineState *last_states,
                         int *hit_triangles,
                         int *d_touch_count,
                         int *d_touch_indices) {
  if (!top_level_as) {
    throw std::invalid_argument("trace_rays requires a valid TLAS.");
  }
  if (!ray_origins || !ray_directions || !output_image || !initial_drgb || !iteration_counts ||
      !last_faces || !touch_counts || !last_diracs || !last_states || !hit_triangles) {
    throw std::invalid_argument("trace_rays received null output or ray buffers.");
  }
  if (ray_count == 0) {
    return;
  }
  if (!pipeline || !shader_table) {
    throw std::runtime_error("trace_rays called before pipeline initialization.");
  }
  CUDA_CHECK(cudaSetDevice(device_index));

  params.image.data = reinterpret_cast<float4 *>(output_image);
  params.last_state.data = last_states;
  params.last_state.size = ray_count;
  params.last_dirac.data = last_diracs;
  params.last_dirac.size = ray_count;
  params.tri_collection.data = hit_triangles;
  params.tri_collection.size = ray_count * max_iters;
  params.iters.data = iteration_counts;
  params.iters.size = ray_count;
  params.last_face.data = last_faces;
  params.last_face.size = ray_count;
  params.touch_count.data = touch_counts;
  params.sh_degree = static_cast<uint>(sh_deg);
  params.max_prim_size = max_prim_size;
  params.max_iters = static_cast<uint>(max_iters);
  params.ray_origins.data = ray_origins;
  params.ray_origins.size = ray_count;
  params.ray_directions.data = ray_directions;
  params.ray_directions.size = ray_count;
  if (camera != nullptr) {
    params.camera = *camera;
  }
  params.tmin = tmin;
  params.tmax = tmax;

  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(initial_drgb), 0,
                        ray_count * sizeof(float4)));
  params.initial_drgb.data = initial_drgb;
  params.initial_drgb.size = ray_count;

  if (num_prims == 0) {
    throw std::runtime_error("trace_rays requires at least one primitive.");
  }

  initialize_density(&params, model->aabbs, d_touch_count, d_touch_indices);

  auto image_buffer = create_buffer_from_cuda_pointer(
      output_image, ray_count * sizeof(float4), sizeof(float4),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto iters_buffer = create_buffer_from_cuda_pointer(
      iteration_counts, ray_count * sizeof(uint), sizeof(uint),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto last_face_buffer = create_buffer_from_cuda_pointer(
      last_faces, ray_count * sizeof(uint), sizeof(uint),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto touch_count_buffer = create_buffer_from_cuda_pointer(
      touch_counts, num_prims * sizeof(uint), sizeof(uint),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto last_dirac_buffer = create_buffer_from_cuda_pointer(
      last_diracs, ray_count * sizeof(float4), sizeof(float4),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto last_state_buffer = create_buffer_from_cuda_pointer(
      last_states, ray_count * sizeof(SplineState), sizeof(SplineState),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto tri_collection_buffer = create_buffer_from_cuda_pointer(
      hit_triangles, ray_count * max_iters * sizeof(int), sizeof(int),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  auto ray_origins_buffer = create_buffer_from_cuda_pointer(
      ray_origins, ray_count * sizeof(float3), sizeof(float3),
      BufferUsage::ShaderResource, ResourceState::ShaderResource);
  auto ray_directions_buffer = create_buffer_from_cuda_pointer(
      ray_directions, ray_count * sizeof(float3), sizeof(float3),
      BufferUsage::ShaderResource, ResourceState::ShaderResource);
  auto initial_drgb_buffer = create_buffer_from_cuda_pointer(
      initial_drgb, ray_count * sizeof(float4), sizeof(float4),
      BufferUsage::UnorderedAccess | BufferUsage::ShaderResource,
      ResourceState::UnorderedAccess);
  if (!image_buffer || !iters_buffer || !last_face_buffer || !touch_count_buffer ||
      !last_dirac_buffer || !last_state_buffer || !tri_collection_buffer ||
      !ray_origins_buffer || !ray_directions_buffer || !initial_drgb_buffer) {
    throw std::runtime_error("Failed to create one or more slang-rhi buffers.");
  }

  auto command_encoder = queue->createCommandEncoder();
  auto ray_tracing_encoder = command_encoder->beginRayTracingPass();
  auto root_object = ray_tracing_encoder->bindPipeline(pipeline, shader_table);
  ShaderCursor cursor(root_object);

  cursor["image"].setBinding(image_buffer);
  cursor["iters"].setBinding(iters_buffer);
  cursor["last_face"].setBinding(last_face_buffer);
  cursor["touch_count"].setBinding(touch_count_buffer);
  cursor["last_dirac"].setBinding(last_dirac_buffer);
  cursor["last_state"].setBinding(last_state_buffer);
  cursor["tri_collection"].setBinding(tri_collection_buffer);
  cursor["ray_origins"].setBinding(ray_origins_buffer);
  cursor["ray_directions"].setBinding(ray_directions_buffer);
  cursor["means"].setBinding(means_buffer);
  cursor["scales"].setBinding(scales_buffer);
  cursor["quats"].setBinding(quats_buffer);
  cursor["densities"].setBinding(densities_buffer);
  cursor["features"].setBinding(features_buffer);
  cursor["initial_drgb"].setBinding(initial_drgb_buffer);
  cursor["traversable"].setBinding(top_level_as);

  Cam local_camera = camera ? *camera : Cam{};
  uint sh_degree_value = static_cast<uint>(sh_deg);
  uint max_iters_value = static_cast<uint>(max_iters);
  cursor["camera"].setData(&local_camera, sizeof(Cam));
  cursor["sh_degree"].setData(&sh_degree_value, sizeof(sh_degree_value));
  cursor["max_iters"].setData(&max_iters_value, sizeof(max_iters_value));
  cursor["tmin"].setData(&tmin, sizeof(tmin));
  cursor["tmax"].setData(&tmax, sizeof(tmax));
  cursor["max_prim_size"].setData(&max_prim_size, sizeof(max_prim_size));

  const uint32_t dispatch_width =
      camera ? static_cast<uint32_t>(camera->width) : static_cast<uint32_t>(ray_count);
  const uint32_t dispatch_height =
      camera ? static_cast<uint32_t>(camera->height) : 1;
  ray_tracing_encoder->dispatchRays(0, dispatch_width, dispatch_height, 1);
  ray_tracing_encoder->end();
  queue->submit(command_encoder->finish());
  queue->waitOnHost();
}

void Forward::reset_features(const Primitives &model) {
  params.features.data = model.features;
  params.features.size = model.num_prims * model.feature_size;
  update_model_buffers(model);
}

Forward::~Forward() noexcept(false) {
  shader_table = nullptr;
  pipeline = nullptr;
  shader_program = nullptr;
  queue = nullptr;
  rhi_device = nullptr;
}
