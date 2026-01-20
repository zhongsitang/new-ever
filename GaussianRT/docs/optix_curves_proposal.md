# OptiX Built-in Curves Implementation Proposal

## OptiX Supported Curve Types

OptiX 7.x+ supports these built-in curve primitives:

```cpp
// Curve types (from optix_types.h)
OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE  // Quadratic B-spline with round cross-section
OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE      // Cubic B-spline with round cross-section
OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR             // Linear segments (capsules/LSS)
OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM         // Catmull-Rom splines
OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER       // Cubic Bezier curves
OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE   // Ribbon curves (flat)
// ... more flat variants
```

## Option 1: Native OptiX Curve Implementation

### File Structure
```
GaussianRT/
├── src/
│   ├── optix/                    # Native OptiX implementation
│   │   ├── optix_device.cpp      # OptiX context/device
│   │   ├── curve_accel.cpp       # Curve acceleration structure
│   │   ├── curve_pipeline.cpp    # RT pipeline for curves
│   │   └── curve_sbt.cpp         # Shader binding table
│   └── cuda/
│       └── curve_backward.cu     # Backward pass (still CUDA)
├── optix_shaders/                # PTX shaders
│   ├── curve_raygen.cu
│   ├── curve_closesthit.cu
│   └── curve_miss.cu
```

### Curve Acceleration Structure Build

```cpp
// curve_accel.cpp
#include <optix.h>
#include <optix_stubs.h>

class CurveAccelerationStructure {
public:
    void build(
        const float4* control_points,  // (x, y, z, radius)
        uint32_t num_vertices,
        uint32_t num_segments,
        OptixPrimitiveType curve_type
    ) {
        // Configure curve build input
        OptixBuildInput curve_input = {};
        curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

        // Curve array configuration
        curve_input.curveArray.curveType = curve_type;  // e.g., OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE
        curve_input.curveArray.numPrimitives = num_segments;

        // Vertex buffer (x, y, z, w=radius)
        curve_input.curveArray.vertexBuffers = &d_vertices;
        curve_input.curveArray.numVertices = num_vertices;
        curve_input.curveArray.vertexStrideInBytes = sizeof(float4);

        // Width buffer (optional, can use vertex.w instead)
        curve_input.curveArray.widthBuffers = nullptr;  // Use radius from float4.w
        curve_input.curveArray.normalizeWidths = 0;

        // Index buffer (optional for segment indexing)
        curve_input.curveArray.indexBuffer = d_indices;
        curve_input.curveArray.indexStrideInBytes = sizeof(uint32_t);

        // Flags
        curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
        curve_input.curveArray.primitiveIndexOffset = 0;

        // End caps
        curve_input.curveArray.endcapFlags = OPTIX_CURVE_ENDCAP_ON;  // Round end caps

        // Build options
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Query memory requirements
        OptixAccelBufferSizes buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context_, &accel_options, &curve_input, 1, &buffer_sizes
        ));

        // Allocate and build
        // ...
    }
};
```

### Curve Shader (PTX)

```cpp
// curve_closesthit.cu
#include <optix.h>

extern "C" __global__ void __closesthit__curve() {
    // Get curve parameter u ∈ [0, 1] along segment
    const float u = optixGetCurveParameter();

    // Get primitive index
    const uint32_t prim_idx = optixGetPrimitiveIndex();

    // For cubic B-spline: get 4 control points
    float4 cp[4];
    optixGetCubicBSplineVertexData(cp);

    // Compute position on curve using de Boor's algorithm
    // (OptiX has already done intersection, we just need the point)

    // Get hit distance
    const float t_hit = optixGetRayTmax();

    // Volume rendering contribution
    // ...
}
```

### Slang Integration for Backward Pass

Even with native OptiX forward, you can still use Slang autodiff for backward:

```slang
// curves_backward.slang
[Differentiable]
float4 eval_cubic_bspline(
    float4 p0, float4 p1, float4 p2, float4 p3,
    float u
) {
    // de Boor's algorithm (differentiable)
    float u2 = u * u;
    float u3 = u2 * u;
    float one_minus_u = 1.0f - u;
    float one_minus_u2 = one_minus_u * one_minus_u;
    float one_minus_u3 = one_minus_u2 * one_minus_u;

    // B-spline basis functions
    float b0 = one_minus_u3 / 6.0f;
    float b1 = (3.0f * u3 - 6.0f * u2 + 4.0f) / 6.0f;
    float b2 = (-3.0f * u3 + 3.0f * u2 + 3.0f * u + 1.0f) / 6.0f;
    float b3 = u3 / 6.0f;

    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3;
}
```

## Option 2: Hybrid Architecture

Keep slang-rhi for ellipsoids, add native OptiX for curves:

```cpp
class HybridRenderer {
    // slang-rhi for ellipsoid Gaussians
    std::unique_ptr<gaussian_rt::VolumeRenderer> ellipsoid_renderer_;

    // Native OptiX for curves
    std::unique_ptr<CurveRenderer> curve_renderer_;

    void render(/* ... */) {
        // Render ellipsoids with slang-rhi
        ellipsoid_renderer_->forward(/* ... */);

        // Render curves with native OptiX
        curve_renderer_->forward(/* ... */);

        // Composite results
        composite_results(/* ... */);
    }
};
```

## Option 3: Custom Primitives with AABB (Current Approach)

If staying with slang-rhi, implement curve intersection manually:

```slang
// forward.slang - Custom curve intersection shader
[shader("intersection")]
void curve_intersection_shader() {
    // Get curve control points from buffer
    uint prim_id = PrimitiveIndex();
    float4 p0 = control_points[prim_id * 4 + 0];
    float4 p1 = control_points[prim_id * 4 + 1];
    float4 p2 = control_points[prim_id * 4 + 2];
    float4 p3 = control_points[prim_id * 4 + 3];

    // Ray-curve intersection (custom implementation)
    float t_hit, u_hit;
    if (intersect_cubic_bspline(ray_origin, ray_direction, p0, p1, p2, p3, t_hit, u_hit)) {
        // Report hit
        ReportHit(t_hit, 0, /* attributes */);
    }
}
```

## Recommendation

For **production quality curve rendering**, use **Option 1 (Native OptiX)**:
- OptiX's built-in curves are highly optimized with hardware acceleration
- Intersection is exact, not approximated
- Handles edge cases (grazing angles, end caps) correctly

For **research/prototyping**, **Option 3 (Custom Primitives)** works:
- Keep existing slang-rhi architecture
- More control over intersection algorithm
- Easier to experiment with modifications

## Required OptiX Version

- OptiX 7.1+: Basic curve support
- OptiX 7.4+: Improved curve performance
- OptiX 8.0+: Motion blur for curves
- OptiX 9.0+: Additional primitive types (spheres, LSS)
