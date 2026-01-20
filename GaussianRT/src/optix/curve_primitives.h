#pragma once

// Native OptiX curve primitives support
// This bypasses slang-rhi for curve-specific features

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace gaussian_rt {
namespace optix_native {

// Supported curve types mapping to OptiX
enum class CurveType {
    Linear,             // OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR (capsules)
    QuadraticBSpline,   // OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
    CubicBSpline,       // OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE
    CatmullRom,         // OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM
    CubicBezier,        // OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER
    FlatQuadratic,      // OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE (ribbons)
};

// Control points per segment for each curve type
inline uint32_t control_points_per_segment(CurveType type) {
    switch (type) {
        case CurveType::Linear:           return 2;
        case CurveType::QuadraticBSpline: return 3;
        case CurveType::CubicBSpline:     return 4;
        case CurveType::CatmullRom:       return 4;
        case CurveType::CubicBezier:      return 4;
        case CurveType::FlatQuadratic:    return 3;
        default: return 4;
    }
}

inline OptixPrimitiveType to_optix_type(CurveType type) {
    switch (type) {
        case CurveType::Linear:           return OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        case CurveType::QuadraticBSpline: return OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        case CurveType::CubicBSpline:     return OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        case CurveType::CatmullRom:       return OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
        case CurveType::CubicBezier:      return OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER;
        case CurveType::FlatQuadratic:    return OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE;
        default: return OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    }
}

// Curve data structure
struct CurveData {
    // Control points: (x, y, z, radius) per vertex
    std::vector<float4> vertices;

    // Optional: per-segment indices (for non-sequential curves)
    std::vector<uint32_t> indices;

    // Curve type
    CurveType type = CurveType::CubicBSpline;

    // Number of curve segments
    uint32_t num_segments = 0;

    // Per-curve attributes (color, opacity, SH coefficients, etc.)
    std::vector<float> attributes;
    uint32_t attribute_stride = 0;  // floats per curve
};

// OptiX context wrapper for curves
class CurveContext {
public:
    CurveContext();
    ~CurveContext();

    bool initialize(int device_id = 0);
    void shutdown();

    OptixDeviceContext get() const { return context_; }

private:
    OptixDeviceContext context_ = nullptr;
    CUcontext cuda_context_ = nullptr;
};

// Curve acceleration structure
class CurveAccelerationStructure {
public:
    CurveAccelerationStructure(CurveContext& context);
    ~CurveAccelerationStructure();

    // Build from curve data
    void build(const CurveData& curves, bool allow_update = false);

    // Update (refit) existing structure
    void update(const CurveData& curves);

    // Get traversable handle for ray tracing
    OptixTraversableHandle get_handle() const { return gas_handle_; }

private:
    CurveContext& context_;
    OptixTraversableHandle gas_handle_ = 0;

    // Device buffers
    CUdeviceptr d_vertices_ = 0;
    CUdeviceptr d_indices_ = 0;
    CUdeviceptr d_gas_output_ = 0;
    CUdeviceptr d_temp_buffer_ = 0;

    size_t gas_output_size_ = 0;
    size_t temp_buffer_size_ = 0;
    bool built_ = false;
};

// Curve ray tracing pipeline
class CurvePipeline {
public:
    CurvePipeline(CurveContext& context);
    ~CurvePipeline();

    bool create(const std::string& ptx_path);

    OptixPipeline get() const { return pipeline_; }
    OptixShaderBindingTable* get_sbt() { return &sbt_; }

private:
    CurveContext& context_;
    OptixPipeline pipeline_ = nullptr;
    OptixModule module_ = nullptr;
    OptixShaderBindingTable sbt_ = {};

    // Program groups
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    // SBT buffers
    CUdeviceptr d_raygen_record_ = 0;
    CUdeviceptr d_miss_record_ = 0;
    CUdeviceptr d_hitgroup_record_ = 0;
};

// Complete curve renderer
class CurveRenderer {
public:
    CurveRenderer();
    ~CurveRenderer();

    bool initialize(int device_id = 0, const std::string& shader_dir = ".");

    // Build acceleration structure
    void build_accel(const CurveData& curves);

    // Forward pass
    void forward(
        const CurveData& curves,
        const float* ray_origins,      // [num_rays, 3]
        const float* ray_directions,   // [num_rays, 3]
        uint32_t num_rays,
        float t_min, float t_max,
        // Outputs
        float* out_colors,             // [num_rays, 4]
        float* out_depths,             // [num_rays]
        int* out_hit_indices,          // [num_rays]
        float* out_hit_params          // [num_rays] - curve parameter u
    );

    // Backward pass (use Slang-generated CUDA)
    void backward(
        const CurveData& curves,
        const float* ray_origins,
        const float* ray_directions,
        uint32_t num_rays,
        const float* grad_colors,
        const float* grad_depths,
        // Outputs
        float* grad_vertices,          // [num_vertices, 4]
        float* grad_attributes         // [num_curves, attr_stride]
    );

private:
    std::unique_ptr<CurveContext> context_;
    std::unique_ptr<CurveAccelerationStructure> accel_;
    std::unique_ptr<CurvePipeline> pipeline_;

    // Launch parameters buffer
    CUdeviceptr d_params_ = 0;

    bool initialized_ = false;
};

} // namespace optix_native
} // namespace gaussian_rt
