#pragma once

#include <cstdint>
#include <cstddef>

namespace gaussian_rt {

// Forward declarations
class Device;
class AccelStruct;
class GaussianPrimitives;
class ForwardRenderer;
class BackwardPass;

//------------------------------------------------------------------------------
// Basic types aligned with Slang shaders
//------------------------------------------------------------------------------

struct alignas(16) Float2 {
    float x, y;

    Float2() : x(0), y(0) {}
    Float2(float x_, float y_) : x(x_), y(y_) {}
};

struct alignas(16) Float3 {
    float x, y, z;

    Float3() : x(0), y(0), z(0) {}
    Float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float* data() { return &x; }
    const float* data() const { return &x; }
};

struct alignas(16) Float4 {
    float x, y, z, w;

    Float4() : x(0), y(0), z(0), w(0) {}
    Float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    float* data() { return &x; }
    const float* data() const { return &x; }
};

//------------------------------------------------------------------------------
// Gaussian primitive data structure (shared with shaders)
//------------------------------------------------------------------------------

struct alignas(16) GaussianData {
    Float3 mean;        // Center position
    float density;      // Opacity/density

    Float3 scale;       // Scale factors
    float _pad0;

    Float4 quat;        // Rotation quaternion (x, y, z, w)
};

//------------------------------------------------------------------------------
// AABB for acceleration structure
//------------------------------------------------------------------------------

struct alignas(8) AABB {
    Float3 minBound;
    Float3 maxBound;
};

//------------------------------------------------------------------------------
// Spline state for volume rendering (shared with shaders)
//------------------------------------------------------------------------------

struct alignas(16) SplineState {
    float logT;         // Log transmittance
    Float3 C;           // Accumulated color

    float t;            // Current parameter
    Float4 drgb;        // Dirac pulse [alpha, alpha*R, alpha*G, alpha*B]
};

//------------------------------------------------------------------------------
// Control point for spline machine
//------------------------------------------------------------------------------

struct alignas(16) ControlPoint {
    float t;            // Parameter position
    Float4 dirac;       // Dirac pulse coefficients
};

//------------------------------------------------------------------------------
// Render parameters
//------------------------------------------------------------------------------

struct RenderParams {
    uint32_t width = 0;
    uint32_t height = 0;
    float tmin = 0.01f;
    float tmax = 100.0f;
    uint32_t maxIters = 512;
    uint32_t shDegree = 0;
    float maxPrimSize = 1.0f;
};

//------------------------------------------------------------------------------
// Forward output structure
//------------------------------------------------------------------------------

struct ForwardOutput {
    void* colorBuffer = nullptr;            // RGBA output (float4 * numRays)
    void* stateBuffer = nullptr;            // Final states (SplineState * numRays)
    void* triCollectionBuffer = nullptr;    // Visited primitives (int * numRays * maxIters)
    void* itersBuffer = nullptr;            // Iterations per ray (uint * numRays)
    void* lastDiracBuffer = nullptr;        // Last Dirac pulse (float4 * numRays)

    size_t numRays = 0;
    size_t maxIters = 0;

    // Device pointers for CUDA interop
    void* d_color = nullptr;
    void* d_state = nullptr;
    void* d_triCollection = nullptr;
    void* d_iters = nullptr;
};

//------------------------------------------------------------------------------
// Gradient output structure
//------------------------------------------------------------------------------

struct GradientOutput {
    void* dMeans = nullptr;         // dL/d_means (float3 * numPrims)
    void* dScales = nullptr;        // dL/d_scales (float3 * numPrims)
    void* dQuats = nullptr;         // dL/d_quats (float4 * numPrims)
    void* dDensities = nullptr;     // dL/d_densities (float * numPrims)
    void* dFeatures = nullptr;      // dL/d_features (float * numPrims * featureSize)
    void* dRayOrigins = nullptr;    // dL/d_ray_origins (float3 * numRays)
    void* dRayDirs = nullptr;       // dL/d_ray_dirs (float3 * numRays)

    size_t numPrims = 0;
    size_t numRays = 0;
    size_t featureSize = 0;
};

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

constexpr float LOG_TRANSMITTANCE_CUTOFF = -10.0f;  // exp(-10) ≈ 0
constexpr uint32_t HIT_BUFFER_SIZE = 16;            // Max hits per trace iteration
constexpr uint32_t MAX_SH_DEGREE = 3;               // Maximum spherical harmonics degree

// SH coefficients count for each degree
inline constexpr uint32_t shCoeffsCount(uint32_t degree) {
    return (degree + 1) * (degree + 1);
}

// Feature size for SH (3 channels)
inline constexpr uint32_t shFeatureSize(uint32_t degree) {
    return shCoeffsCount(degree) * 3;
}

//------------------------------------------------------------------------------
// Error codes
//------------------------------------------------------------------------------

enum class Result {
    Success = 0,
    ErrorInvalidArgument,
    ErrorOutOfMemory,
    ErrorDeviceNotInitialized,
    ErrorShaderCompilation,
    ErrorAccelStructBuild,
    ErrorPipelineCreation,
    ErrorKernelLaunch,
    ErrorCUDA,
    ErrorUnknown
};

inline const char* resultToString(Result result) {
    switch (result) {
        case Result::Success: return "Success";
        case Result::ErrorInvalidArgument: return "Invalid argument";
        case Result::ErrorOutOfMemory: return "Out of memory";
        case Result::ErrorDeviceNotInitialized: return "Device not initialized";
        case Result::ErrorShaderCompilation: return "Shader compilation error";
        case Result::ErrorAccelStructBuild: return "Acceleration structure build error";
        case Result::ErrorPipelineCreation: return "Pipeline creation error";
        case Result::ErrorKernelLaunch: return "Kernel launch error";
        case Result::ErrorCUDA: return "CUDA error";
        case Result::ErrorUnknown: return "Unknown error";
        default: return "Unknown result code";
    }
}

} // namespace gaussian_rt
