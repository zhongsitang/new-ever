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
#include <optix.h>
#include <optix_stubs.h>
#include <memory>
#include <unordered_map>
#include "optix_error.h"

// =============================================================================
// DeviceContext - Per-device OptiX context (header-only, cached globally)
// =============================================================================

namespace detail {

inline void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void* /*cbdata*/) {
    // Silently ignore OptiX log messages
}

}  // namespace detail

/// Manages per-device OptiX context. Cached globally and shared.
class DeviceContext {
public:
    /// Get or create context for a specific device (cached globally)
    static DeviceContext& get(int device_index) {
        auto it = contexts().find(device_index);
        if (it != contexts().end()) {
            return *it->second;
        }

        auto ctx = std::unique_ptr<DeviceContext>(new DeviceContext(device_index));
        auto& ref = *ctx;
        contexts()[device_index] = std::move(ctx);
        return ref;
    }

    OptixDeviceContext context() const { return context_; }
    int device() const { return device_; }

    // Non-copyable
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    ~DeviceContext() {
        if (context_) optixDeviceContextDestroy(context_);
    }

private:
    explicit DeviceContext(int device_index) : device_(device_index) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaFree(0));  // Initialize CUDA context
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &detail::context_log_cb;
        options.logCallbackLevel = 4;

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
    }

    static std::unordered_map<int, std::unique_ptr<DeviceContext>>& contexts() {
        static std::unordered_map<int, std::unique_ptr<DeviceContext>> g_contexts;
        return g_contexts;
    }

    int device_ = -1;
    OptixDeviceContext context_ = nullptr;
};
