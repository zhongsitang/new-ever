// Copyright 2024 Google LLC
//
// Example: How to use the refactored OptixPipeline class
//
// This file demonstrates the modern, simplified approach to OptiX ray tracing
// with Slang shaders. Compare with the old Forward class for reference.

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "OptixPipeline.h"
#include "GAS.h"
#include "structs.h"

// =============================================================================
// Example: Minimal OptiX Context Setup
// =============================================================================
class OptixContext {
public:
    static OptixContext& instance() {
        static OptixContext ctx;
        return ctx;
    }

    OptixDeviceContext get() const { return m_context; }
    int deviceId() const { return m_device_id; }

private:
    OptixContext() {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        m_device_id = 0;

        // Initialize OptiX
        OPTIX_CHECK(optixInit());

        // Create context with default options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = nullptr;
        options.logCallbackLevel = 0;

        CUcontext cuCtx = nullptr;
        cuCtxGetCurrent(&cuCtx);
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
    }

    ~OptixContext() {
        if (m_context) {
            optixDeviceContextDestroy(m_context);
        }
    }

    OptixDeviceContext m_context = nullptr;
    int m_device_id = 0;
};

// =============================================================================
// Example: Simple Volume Renderer using OptixPipeline
// =============================================================================
class SimpleVolumeRenderer {
public:
    SimpleVolumeRenderer(bool backward_mode = false)
        : m_backward_mode(backward_mode)
    {
        auto& ctx = OptixContext::instance();
        m_pipeline.init(ctx.get(), ctx.deviceId(), backward_mode);
    }

    // Render a frame
    void render(
        const Primitives& primitives,
        const OptixTraversableHandle& gas_handle,
        float3* ray_origins,
        float3* ray_directions,
        size_t num_rays,
        float4* output_image,
        size_t sh_degree = 0,
        size_t max_iters = 64,
        float tmin = 0.001f,
        float tmax = 1000.0f
    ) {
        // Prepare launch parameters
        LaunchParams params = {};

        // Output buffers
        params.image = {output_image, num_rays};
        // Other outputs can be nullptr for simple forward pass
        params.iters = {nullptr, 0};
        params.last_face = {nullptr, 0};
        params.touch_count = {nullptr, 0};
        params.last_dirac = {nullptr, 0};
        params.last_state = {nullptr, 0};
        params.tri_collection = {nullptr, 0};

        // Input rays
        params.ray_origins = {ray_origins, num_rays};
        params.ray_directions = {ray_directions, num_rays};

        // Primitive data
        params.half_attribs = {primitives.half_attribs, primitives.num_prims};
        params.means = {primitives.means, primitives.num_prims};
        params.scales = {primitives.scales, primitives.num_prims};
        params.quats = {primitives.quats, primitives.num_prims};
        params.densities = {primitives.densities, primitives.num_prims};
        params.features = {primitives.features, primitives.feature_size};

        // Rendering parameters
        params.sh_degree = sh_degree;
        params.max_iters = max_iters;
        params.tmin = tmin;
        params.tmax = tmax;
        params.initial_drgb = {nullptr, 0};  // Zero-initialized by default
        params.max_prim_size = 1.0f;

        // Acceleration structure
        params.handle = gas_handle;

        // Launch
        uint32_t width = static_cast<uint32_t>(num_rays);
        m_pipeline.launch(params, width, 1, nullptr);
    }

private:
    OptixPipeline m_pipeline;
    bool m_backward_mode;
};

// =============================================================================
// Key Differences from Old Forward Class:
//
// OLD (Forward.cpp):
//   - 300+ lines of code
//   - Manual module/pipeline/SBT setup scattered across multiple functions
//   - Complex state management
//   - Hard to understand initialization order
//
// NEW (OptixPipeline.cpp):
//   - ~150 lines of code
//   - Clear separation: createModule() -> createProgramGroups() -> createPipeline() -> createSBT()
//   - RAII resource management
//   - Single launch() function with all parameters
//
// API Changes:
//   - Old: Forward::Forward(ctx, device, backward, fast_build)
//          Forward::forward(primitives, cam, rays, ...)
//   - New: OptixPipeline::init(ctx, device, backward_mode)
//          OptixPipeline::launch(params, width, height, stream)
//
// Benefits:
//   1. Cleaner code that matches OptiX 9.1 best practices
//   2. All parameters in one struct (LaunchParams) - easier to debug
//   3. Better alignment with OptiX_Apps examples
//   4. Simplified shader binding table setup
//   5. Proper stack size calculation using optixUtil functions
//
// =============================================================================

/*
 * SLANG SHADER ENTRY POINT NAMING:
 *
 * Slang automatically generates OptiX entry point names from shader attributes:
 *
 *   [shader("raygeneration")]
 *   void rg_float() { ... }           -> __raygen__rg_float
 *
 *   [shader("miss")]
 *   void ms() { ... }                 -> __miss__ms
 *
 *   [shader("anyhit")]
 *   void ah() { ... }                 -> __anyhit__ah
 *
 *   [shader("intersection")]
 *   void ellipsoid() { ... }          -> __intersection__ellipsoid
 *
 * These names MUST match the entryFunctionName strings in OptixPipeline.cpp
 *
 *
 * SLANG GLOBAL PARAMS:
 *
 * Slang collects all global shader variables into a struct named SLANG_globalParams.
 * This struct is automatically populated with:
 *   - RWStructuredBuffer<T> -> StructuredBuffer<T> with .data and .size
 *   - StructuredBuffer<T>   -> Same
 *   - Scalar types          -> Direct values
 *
 * The ORDER of members in LaunchParams must match the ORDER of declarations
 * in the slang shader file!
 *
 *
 * OPTIX 9.1 API NOTES:
 *
 * 1. optixModuleCreate (not optixModuleCreateFromPTX)
 *    - Unified API for both PTX and OptiX-IR
 *    - Input can be either format based on NVCC compilation flags
 *
 * 2. optixUtilAccumulateStackSizes / optixUtilComputeStackSizes
 *    - Helper functions for correct stack size calculation
 *    - Replaces manual calculation in older code
 *
 * 3. usesPrimitiveTypeFlags
 *    - Must be set for custom primitives (AABB)
 *    - OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM for our ellipsoids
 *
 * 4. numPayloadValues = 32
 *    - Used for sorted hit list in anyhit shader
 *    - Each payload slot is a uint32
 *    - We use 16 pairs of (t, index) for sorting
 */
