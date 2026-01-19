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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "slang-com-ptr.h"
#include "slang-rhi.h"
#include "structs.h"
#include "TraceParams.h"

using uint = uint32_t;

class Forward {
   public:
    Forward() = default;
    Forward(
        rhi::IDevice *device,
        rhi::ICommandQueue *queue,
        int8_t device_index,
        const Primitives &model,
        const bool enable_backward);
    ~Forward() noexcept(false);
    void trace_rays(rhi::IAccelerationStructure *top_level_as,
                    const size_t ray_count,
                    float3 *ray_origins,
                    float3 *ray_directions,
                    void *output_image,
                    uint sh_degree,
                    float tmin,
                    float tmax,
                    float4 *initial_drgb,
                    Cam *camera=NULL,
                    const size_t max_iters=10000,
                    const float max_prim_size=3,
                    uint *iteration_counts=NULL,
                    uint *last_faces=NULL,
                    uint *touch_counts=NULL,
                    float4 *last_diracs=NULL,
                    SplineState *last_states=NULL,
                    int *hit_triangles=NULL,
                    int *d_touch_count=NULL,
                    int *d_touch_indices=NULL);
   void reset_features(const Primitives &model);
   bool enable_backward = false;
   size_t num_prims = 0;
   private:
    Slang::Result create_ray_tracing_pipeline();
    Slang::Result load_shader_program(rhi::IShaderProgram **out_program) const;
    void update_model_buffers(const Primitives &model);
    Slang::ComPtr<rhi::IBuffer> create_buffer_from_cuda_pointer(
        void *pointer,
        size_t size,
        size_t element_size,
        rhi::BufferUsage usage,
        rhi::ResourceState default_state) const;

    Params params;
    Slang::ComPtr<rhi::IDevice> rhi_device;
    Slang::ComPtr<rhi::ICommandQueue> queue;
    Slang::ComPtr<rhi::IRayTracingPipeline> pipeline;
    Slang::ComPtr<rhi::IShaderProgram> shader_program;
    Slang::ComPtr<rhi::IShaderTable> shader_table;
    Slang::ComPtr<rhi::IBuffer> means_buffer;
    Slang::ComPtr<rhi::IBuffer> scales_buffer;
    Slang::ComPtr<rhi::IBuffer> quats_buffer;
    Slang::ComPtr<rhi::IBuffer> densities_buffer;
    Slang::ComPtr<rhi::IBuffer> features_buffer;
    int8_t device_index = -1;
    const Primitives *model = nullptr;
};
