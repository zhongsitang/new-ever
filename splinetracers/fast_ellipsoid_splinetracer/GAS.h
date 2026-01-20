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
#include "slang-rhi/acceleration-structure-utils.h"
#include "structs.h"

class GAS {
   public:
    GAS() noexcept = default;
    GAS(rhi::IDevice *device,
        rhi::ICommandQueue *queue,
        const uint8_t device_index,
        const Primitives &model,
        const bool enable_anyhit = false,
        const bool fast_build = false)
        : device(device),
          queue(queue),
          device_index(device_index),
          enable_anyhit(enable_anyhit),
          fast_build(fast_build) {
      build(model);
    }

    ~GAS() noexcept(false);
    GAS(const GAS &) = delete;
    GAS &operator=(const GAS &) = delete;
    GAS(GAS &&other) noexcept;
    GAS &operator=(GAS &&other) {
      using std::swap;
      if (this != &other) {
        GAS tmp(std::move(other));
        swap(tmp, *this);
      }
      return *this;
    }
    friend void swap(GAS &first, GAS &second) {
      using std::swap;
      swap(first.device, second.device);
      swap(first.queue, second.queue);
      swap(first.device_index, second.device_index);
      swap(first.blas, second.blas);
      swap(first.tlas, second.tlas);
      swap(first.aabb_buffer, second.aabb_buffer);
      swap(first.instance_buffer, second.instance_buffer);
    }

    bool defined() const { return tlas != nullptr; }
    rhi::IAccelerationStructure *get_tlas() const { return tlas.get(); }

   private:
    void build(const Primitives &model);
    void release();

    Slang::ComPtr<rhi::IDevice> device;
    Slang::ComPtr<rhi::ICommandQueue> queue;
    Slang::ComPtr<rhi::IBuffer> aabb_buffer;
    Slang::ComPtr<rhi::IBuffer> instance_buffer;
    Slang::ComPtr<rhi::IAccelerationStructure> blas;
    Slang::ComPtr<rhi::IAccelerationStructure> tlas;
    int8_t device_index = -1;
    bool enable_anyhit = false;
    bool fast_build = false;
};
