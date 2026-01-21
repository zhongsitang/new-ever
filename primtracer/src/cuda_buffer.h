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

// =============================================================================
// CudaBuffer - Simple RAII wrapper for CUDA device memory
// =============================================================================

#pragma once

#include <cassert>
#include <vector>
#include "exception.h"

struct CudaBuffer {
    CUdeviceptr device_ptr() const { return reinterpret_cast<CUdeviceptr>(data_); }

    void resize(size_t size) {
        if (data_) {
            free();
        }
        alloc(size);
    }

    void alloc(size_t size) {
        assert(data_ == nullptr);
        size_bytes_ = size;
        CUDA_CHECK(cudaMalloc(&data_, size_bytes_));
    }

    void free() {
        CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
        size_bytes_ = 0;
    }

    template <typename T>
    void alloc_and_upload(const std::vector<T>& vec) {
        alloc(vec.size() * sizeof(T));
        upload(vec.data(), vec.size());
    }

    template <typename T>
    void upload(const T* src, size_t count) {
        assert(data_ != nullptr);
        assert(size_bytes_ == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(data_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void download(T* dst, size_t count) {
        assert(data_ != nullptr);
        assert(size_bytes_ == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(dst, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    size_t size_bytes() const { return size_bytes_; }

private:
    size_t size_bytes_ = 0;
    void* data_ = nullptr;
};

// Legacy alias
using CUDABuffer = CudaBuffer;
