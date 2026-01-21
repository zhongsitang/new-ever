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

#include <optix.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------
namespace detail {

inline std::string optix_error_name(OptixResult res) {
#if !defined(__CUDACC__)
    const char* name = optixGetErrorName(res);
    return name ? name : "OPTIX_ERROR_UNKNOWN";
#else
    (void)res;
    return "OPTIX_ERROR";
#endif
}

inline const char* cuda_error_string(cudaError_t err) {
    const char* s = cudaGetErrorString(err);
    return s ? s : "cudaErrorUnknown";
}

inline std::string make_location(const char* file, int line) {
    std::ostringstream out;
    out << file << ":" << line;
    return out.str();
}

} // namespace detail

// -----------------------------------------------------------------------------
// Exception class
// -----------------------------------------------------------------------------
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
    explicit Exception(const char* msg) : std::runtime_error(msg ? msg : "Exception") {}

    Exception(OptixResult res, const std::string& msg)
        : std::runtime_error(detail::optix_error_name(res) + ": " + msg) {}
};

// -----------------------------------------------------------------------------
// OptiX macros
// -----------------------------------------------------------------------------
#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult res_ = (call);                                              \
        if (res_ != OPTIX_SUCCESS) {                                            \
            std::ostringstream ss_;                                             \
            ss_ << "OptiX call '" << #call << "' failed at "                    \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(res_, ss_.str());                                   \
        }                                                                       \
    } while (0)

// Self-contained: defines log_ and log_size_ internally
// Caller uses these names in the call expression
#define OPTIX_CHECK_LOG(call)                                                   \
    do {                                                                        \
        char log_[8192];                                                        \
        size_t log_size_ = sizeof(log_);                                        \
        OptixResult res_ = (call);                                              \
        if (res_ != OPTIX_SUCCESS) {                                            \
            std::ostringstream ss_;                                             \
            ss_ << "OptiX call '" << #call << "' failed at "                    \
                << detail::make_location(__FILE__, __LINE__)                    \
                << "\nLog: " << log_;                                           \
            throw Exception(res_, ss_.str());                                   \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK_NOTHROW(call)                                               \
    do {                                                                        \
        OptixResult res_ = (call);                                              \
        if (res_ != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX call '" << #call << "' failed at "              \
                      << detail::make_location(__FILE__, __LINE__) << "\n";     \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// CUDA macros
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA call (" << #call << ") failed: '"                      \
                << detail::cuda_error_string(err_) << "' at "                   \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
    } while (0)

#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                        \
        cudaError_t sync_err_ = cudaDeviceSynchronize();                        \
        cudaError_t last_err_ = cudaGetLastError();                             \
        if (sync_err_ != cudaSuccess) {                                         \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA sync failed: '"                                        \
                << detail::cuda_error_string(sync_err_) << "' at "              \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
        if (last_err_ != cudaSuccess) {                                         \
            std::ostringstream ss_;                                             \
            ss_ << "CUDA last error: '"                                         \
                << detail::cuda_error_string(last_err_) << "' at "              \
                << detail::make_location(__FILE__, __LINE__);                   \
            throw Exception(ss_.str());                                         \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_NOTHROW(call)                                                \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::cerr << "CUDA call (" << #call << ") failed: '"                \
                      << detail::cuda_error_string(err_) << "' at "             \
                      << detail::make_location(__FILE__, __LINE__) << "\n";     \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// PyTorch tensor check macros (requires torch/extension.h)
// -----------------------------------------------------------------------------
#ifdef TORCH_EXTENSION_H

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_DEVICE(x)                                                        \
    TORCH_CHECK(x.device() == this->device, #x " must be on the same device")

#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_FLOAT(x)                                                         \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")

#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT_DIM3(x)                                                    \
    CHECK_INPUT(x);                                                            \
    CHECK_DEVICE(x);                                                           \
    CHECK_FLOAT(x);                                                            \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

#define CHECK_FLOAT_DIM4(x)                                                    \
    CHECK_INPUT(x);                                                            \
    CHECK_DEVICE(x);                                                           \
    CHECK_FLOAT(x);                                                            \
    TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

#define CHECK_FLOAT_DIM4_CPU(x)                                                \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_FLOAT(x);                                                            \
    TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")

#define CHECK_FLOAT_DIM3_CPU(x)                                                \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_FLOAT(x);                                                            \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

#endif // TORCH_EXTENSION_H
