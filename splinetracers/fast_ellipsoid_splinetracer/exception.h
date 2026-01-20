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

// -----------------------------------------------------------------------------
// Internal helpers (not part of public API)
// -----------------------------------------------------------------------------
inline std::string detail_optix_error_name_safe(OptixResult res)
{
#if !defined(__CUDACC__)
    const char* name = optixGetErrorName(res);
    return name ? std::string(name) : std::string("OPTIX_ERROR_UNKNOWN");
#else
    // In CUDA compilation units, avoid host-only OptiX APIs.
    (void)res;
    return std::string("OPTIX_ERROR");
#endif
}

inline const char* detail_cuda_error_string_safe(cudaError_t err)
{
    const char* s = cudaGetErrorString(err);
    return s ? s : "cudaErrorUnknown";
}

inline std::string detail_make_location(const char* file, int line)
{
    std::ostringstream out;
    out << file << ":" << line;
    return out.str();
}

// -----------------------------------------------------------------------------
// Unified, more robust Exception
// -----------------------------------------------------------------------------
class Exception : public std::runtime_error
{
public:
    explicit Exception(const std::string& msg)
        : std::runtime_error(msg)
    {}

    explicit Exception(const char* msg)
        : std::runtime_error(msg ? msg : "Exception")
    {}

    // OptiX-aware constructor (gracefully degrades in .cu files)
    Exception(OptixResult res, const std::string& msg)
        : std::runtime_error(detail_optix_error_name_safe(res) + ": " + msg)
    {}

    Exception(OptixResult res, const char* msg)
        : Exception(res, std::string(msg ? msg : "OptiX error"))
    {}
};

// -----------------------------------------------------------------------------
// OptiX error-checking macros
// -----------------------------------------------------------------------------

#ifndef OPTIX_CHECK
#define OPTIX_CHECK(call)                                                         \
    do                                                                            \
    {                                                                             \
        OptixResult _res = (call);                                                \
        if (_res != OPTIX_SUCCESS)                                                \
        {                                                                         \
            std::ostringstream _ss;                                               \
            _ss << "OptiX call '" << #call << "' failed at "                      \
                << detail_make_location(__FILE__, __LINE__);                     \
            throw Exception(_res, _ss.str());                                     \
        }                                                                         \
    } while (0)
#endif

// OPTIX_CHECK_LOG: call must have (log, &log_size) as last two args
// Example: OPTIX_CHECK_LOG(optixModuleCreate(ctx, &opts, &pco, ptx, len, _log, &_log_size, &module))
#ifndef OPTIX_CHECK_LOG
#define OPTIX_CHECK_LOG(call)                                                     \
    do                                                                            \
    {                                                                             \
        char   _log[4096];                                                        \
        size_t _log_size = sizeof(_log);                                          \
        OptixResult _res = (call);                                                \
        if (_res != OPTIX_SUCCESS)                                                \
        {                                                                         \
            std::ostringstream _ss;                                               \
            _ss << "OptiX call '" << #call << "' failed at "                      \
                << detail_make_location(__FILE__, __LINE__)                       \
                << "\nLog: " << _log;                                             \
            throw Exception(_res, _ss.str());                                     \
        }                                                                         \
    } while (0)
#endif

#ifndef OPTIX_CHECK_NOTHROW
#define OPTIX_CHECK_NOTHROW(call)                                                 \
    do                                                                            \
    {                                                                             \
        OptixResult _res = (call);                                                \
        if (_res != OPTIX_SUCCESS)                                                \
        {                                                                         \
            std::cerr << "OptiX call '" << #call << "' failed at "                \
                      << detail_make_location(__FILE__, __LINE__) << "\n";       \
            std::terminate();                                                     \
        }                                                                         \
    } while (0)
#endif

// -----------------------------------------------------------------------------
// CUDA error-checking macros
// -----------------------------------------------------------------------------

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess)                                                  \
        {                                                                         \
            std::ostringstream _ss;                                               \
            _ss << "CUDA call (" << #call << ") failed with error: '"             \
                << detail_cuda_error_string_safe(_err) << "' at "                 \
                << detail_make_location(__FILE__, __LINE__);                     \
            throw Exception(_ss.str());                                           \
        }                                                                         \
    } while (0)
#endif

#ifndef CUDA_SYNC_CHECK
#define CUDA_SYNC_CHECK()                                                         \
    do                                                                            \
    {                                                                             \
        cudaError_t _sync_err = cudaDeviceSynchronize();                          \
        cudaError_t _last_err = cudaGetLastError();                               \
        if (_sync_err != cudaSuccess)                                             \
        {                                                                         \
            std::ostringstream _ss;                                               \
            _ss << "CUDA synchronize failed with error: '"                        \
                << detail_cuda_error_string_safe(_sync_err) << "' at "            \
                << detail_make_location(__FILE__, __LINE__);                     \
            throw Exception(_ss.str());                                           \
        }                                                                         \
        if (_last_err != cudaSuccess)                                             \
        {                                                                         \
            std::ostringstream _ss;                                               \
            _ss << "CUDA last error after synchronize: '"                         \
                << detail_cuda_error_string_safe(_last_err) << "' at "            \
                << detail_make_location(__FILE__, __LINE__);                     \
            throw Exception(_ss.str());                                           \
        }                                                                         \
    } while (0)
#endif

#ifndef CUDA_CHECK_NOTHROW
#define CUDA_CHECK_NOTHROW(call)                                                  \
    do                                                                            \
    {                                                                             \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess)                                                  \
        {                                                                         \
            std::cerr << "CUDA call (" << #call << ") failed with error: '"       \
                      << detail_cuda_error_string_safe(_err) << "' at "           \
                      << detail_make_location(__FILE__, __LINE__) << "\n";       \
            std::terminate();                                                     \
        }                                                                         \
    } while (0)
#endif
