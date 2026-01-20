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

// This header is safe to include from both .cpp and .cu files.
// Host-only OptiX helpers (e.g., optixGetErrorName) are guarded for __CUDACC__.

#include <optix.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// ============================================================================
// Internal Helpers
// ============================================================================

namespace detail {

inline std::string optixErrorName(OptixResult res) {
#if !defined(__CUDACC__)
    const char* name = optixGetErrorName(res);
    return name ? std::string(name) : "OPTIX_ERROR_UNKNOWN";
#else
    (void)res;
    return "OPTIX_ERROR";
#endif
}

inline const char* cudaErrorString(cudaError_t err) {
    const char* s = cudaGetErrorString(err);
    return s ? s : "cudaErrorUnknown";
}

inline std::string makeLocation(const char* file, int line) {
    std::ostringstream out;
    out << file << ":" << line;
    return out.str();
}

} // namespace detail

// ============================================================================
// Exception Class
// ============================================================================

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg)
        : std::runtime_error(msg) {}

    explicit Exception(const char* msg)
        : std::runtime_error(msg ? msg : "Exception") {}

    Exception(OptixResult res, const std::string& msg)
        : std::runtime_error(detail::optixErrorName(res) + ": " + msg) {}

    Exception(OptixResult res, const char* msg)
        : Exception(res, std::string(msg ? msg : "OptiX error")) {}
};

// ============================================================================
// OptiX Macros
// ============================================================================

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult _res = (call);                                             \
        if (_res != OPTIX_SUCCESS) {                                           \
            std::ostringstream _ss;                                            \
            _ss << "OptiX call '" << #call << "' failed at "                   \
                << detail::makeLocation(__FILE__, __LINE__);                   \
            throw Exception(_res, _ss.str());                                  \
        }                                                                      \
    } while (0)

// Self-contained macro with built-in log buffer.
// Usage: OPTIX_CHECK_LOG(optixXxx(..., log, &logSize, ...))
// The variables 'log' and 'logSize' are defined by this macro.
#define OPTIX_CHECK_LOG(call)                                                  \
    do {                                                                       \
        char   log[16384];                                                     \
        size_t logSize = sizeof(log);                                          \
        OptixResult _res = (call);                                             \
        if (_res != OPTIX_SUCCESS) {                                           \
            std::ostringstream _ss;                                            \
            _ss << "OptiX call '" << #call << "' failed at "                   \
                << detail::makeLocation(__FILE__, __LINE__)                    \
                << "\nLog:\n" << log                                           \
                << (logSize < sizeof(log) ? "" : "<TRUNCATED>");               \
            throw Exception(_res, _ss.str());                                  \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK_NOTHROW(call)                                              \
    do {                                                                       \
        OptixResult _res = (call);                                             \
        if (_res != OPTIX_SUCCESS) {                                           \
            std::cerr << "OptiX call '" << #call << "' failed at "             \
                      << detail::makeLocation(__FILE__, __LINE__) << "\n";     \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA Macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _ss;                                            \
            _ss << "CUDA call (" << #call << ") failed: '"                     \
                << detail::cudaErrorString(_err) << "' at "                    \
                << detail::makeLocation(__FILE__, __LINE__);                   \
            throw Exception(_ss.str());                                        \
        }                                                                      \
    } while (0)

#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaError_t _sync = cudaDeviceSynchronize();                           \
        cudaError_t _last = cudaGetLastError();                                \
        if (_sync != cudaSuccess) {                                            \
            std::ostringstream _ss;                                            \
            _ss << "CUDA sync failed: '"                                       \
                << detail::cudaErrorString(_sync) << "' at "                   \
                << detail::makeLocation(__FILE__, __LINE__);                   \
            throw Exception(_ss.str());                                        \
        }                                                                      \
        if (_last != cudaSuccess) {                                            \
            std::ostringstream _ss;                                            \
            _ss << "CUDA last error: '"                                        \
                << detail::cudaErrorString(_last) << "' at "                   \
                << detail::makeLocation(__FILE__, __LINE__);                   \
            throw Exception(_ss.str());                                        \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_NOTHROW(call)                                               \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA call (" << #call << ") failed: '"               \
                      << detail::cudaErrorString(_err) << "' at "              \
                      << detail::makeLocation(__FILE__, __LINE__) << "\n";     \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)
