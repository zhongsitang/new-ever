// Common definitions for GaussianRT
// Shared types and utility functions for volume rendering

module common;

// ============================================================================
// Constants
// ============================================================================

static const float PI = 3.14159265358979323846f;
static const float LOG_TRANSMITTANCE_CUTOFF = -5.54f;  // exp(-5.54) ≈ 0.004
static const uint MAX_SH_DEGREE = 3;
static const uint SH_COEFF_COUNT = 16;  // (MAX_SH_DEGREE + 1)^2

// ============================================================================
// Differentiable Types
// ============================================================================

// Render state accumulated along a ray
struct RenderState : IDifferentiable {
    float log_transmittance;    // log(T) where T = exp(-integral of sigma*dt)
    float3 accumulated_color;   // C = integral of T * sigma * c dt
    float accumulated_depth;    // Weighted depth
    float depth_weight;
    float t_current;            // Current ray parameter

    // Distortion loss components
    float2 distortion_accum;
    float2 weight_accum;
};

// Control point: a sample along the ray
struct ControlPoint : IDifferentiable {
    float t;            // Distance along ray
    float sigma;        // Density/opacity at this point
    float3 color;       // Color at this point
};

// ============================================================================
// Spherical Harmonics Coefficients
// ============================================================================

static const float SH_C0 = 0.28209479177387814f;

static const float SH_C1_0 = 0.4886025119029199f;
static const float SH_C1_1 = 0.4886025119029199f;
static const float SH_C1_2 = 0.4886025119029199f;

static const float SH_C2_0 = 1.0925484305920792f;
static const float SH_C2_1 = 1.0925484305920792f;
static const float SH_C2_2 = 0.31539156525252005f;
static const float SH_C2_3 = 1.0925484305920792f;
static const float SH_C2_4 = 0.5462742152960396f;

static const float SH_C3_0 = 0.5900435899266435f;
static const float SH_C3_1 = 2.890611442640554f;
static const float SH_C3_2 = 0.4570457994644658f;
static const float SH_C3_3 = 0.3731763325901154f;
static const float SH_C3_4 = 0.4570457994644658f;
static const float SH_C3_5 = 1.445305721320277f;
static const float SH_C3_6 = 0.5900435899266435f;

// ============================================================================
// Utility Functions
// ============================================================================

[Differentiable]
float3 rotate_by_quaternion(float3 v, float4 q) {
    // q = (w, x, y, z)
    float3 u = float3(q.y, q.z, q.w);
    float s = q.x;
    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

[Differentiable]
float3 rotate_by_quaternion_inverse(float3 v, float4 q) {
    float4 q_conj = float4(q.x, -q.y, -q.z, -q.w);
    return rotate_by_quaternion(v, q_conj);
}

// Safe exponential to avoid overflow
[Differentiable]
float safe_exp(float x) {
    return exp(clamp(x, -80.0f, 80.0f));
}

// Safe log to avoid -inf
[Differentiable]
float safe_log(float x) {
    return log(max(x, 1e-10f));
}

// ============================================================================
// Spherical Harmonics Evaluation
// ============================================================================

// SH basis functions
[Differentiable]
float3 eval_sh_degree0(float3 direction, float3 c0) {
    return SH_C0 * c0;
}

[Differentiable]
float3 eval_sh_degree1(float3 dir, float3 c1, float3 c2, float3 c3) {
    return SH_C1_0 * dir.y * c1 +
           SH_C1_1 * dir.z * c2 +
           SH_C1_2 * dir.x * c3;
}

[Differentiable]
float3 eval_sh_degree2(float3 dir,
                       float3 c4, float3 c5, float3 c6, float3 c7, float3 c8) {
    float x = dir.x, y = dir.y, z = dir.z;
    return SH_C2_0 * x * y * c4 +
           SH_C2_1 * y * z * c5 +
           SH_C2_2 * (2.0f * z * z - x * x - y * y) * c6 +
           SH_C2_3 * x * z * c7 +
           SH_C2_4 * (x * x - y * y) * c8;
}

[Differentiable]
float3 eval_sh_degree3(float3 dir,
                       float3 c9, float3 c10, float3 c11, float3 c12,
                       float3 c13, float3 c14, float3 c15) {
    float x = dir.x, y = dir.y, z = dir.z;
    return SH_C3_0 * y * (3.0f * x * x - y * y) * c9 +
           SH_C3_1 * x * y * z * c10 +
           SH_C3_2 * y * (4.0f * z * z - x * x - y * y) * c11 +
           SH_C3_3 * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y) * c12 +
           SH_C3_4 * x * (4.0f * z * z - x * x - y * y) * c13 +
           SH_C3_5 * z * (x * x - y * y) * c14 +
           SH_C3_6 * x * (x * x - 3.0f * y * y) * c15;
}

// Full SH evaluation with configurable degree
[Differentiable]
float3 eval_sh_color(
    float3 direction,
    float3 coeffs[SH_COEFF_COUNT],
    uint sh_degree
) {
    float3 color = eval_sh_degree0(direction, coeffs[0]);

    if (sh_degree >= 1) {
        color += eval_sh_degree1(direction, coeffs[1], coeffs[2], coeffs[3]);
    }
    if (sh_degree >= 2) {
        color += eval_sh_degree2(direction,
                                 coeffs[4], coeffs[5], coeffs[6], coeffs[7], coeffs[8]);
    }
    if (sh_degree >= 3) {
        color += eval_sh_degree3(direction,
                                 coeffs[9], coeffs[10], coeffs[11], coeffs[12],
                                 coeffs[13], coeffs[14], coeffs[15]);
    }

    // Apply sigmoid activation for valid color range
    return 1.0f / (1.0f + exp(-color));
}

// ============================================================================
// Volume Rendering Integration
// ============================================================================

// Initialize render state at ray origin
RenderState init_render_state(float t_min) {
    RenderState state;
    state.log_transmittance = 0.0f;  // T = 1
    state.accumulated_color = float3(0.0f);
    state.accumulated_depth = 0.0f;
    state.depth_weight = 0.0f;
    state.t_current = t_min;
    state.distortion_accum = float2(0.0f);
    state.weight_accum = float2(0.0f);
    return state;
}

// Update render state with a new control point
// Implements the volume rendering integral:
//   C = integral_0^inf T(t) * sigma(t) * c(t) dt
// where T(t) = exp(-integral_0^t sigma(s) ds)
[Differentiable]
RenderState update_render_state(
    RenderState prev_state,
    ControlPoint point
) {
    RenderState new_state;

    // Distance traveled since last sample
    float dt = point.t - prev_state.t_current;

    // Optical thickness for this segment
    float tau = point.sigma * dt;

    // Alpha (opacity) for this segment: 1 - exp(-tau)
    float alpha = 1.0f - safe_exp(-tau);

    // Transmittance weight: T * alpha = exp(log_T) * alpha
    float weight = alpha * safe_exp(prev_state.log_transmittance);

    // Update accumulated color
    new_state.accumulated_color = prev_state.accumulated_color + weight * point.color;

    // Update log transmittance: log(T') = log(T) - tau
    new_state.log_transmittance = prev_state.log_transmittance - tau;

    // Update depth accumulation
    new_state.accumulated_depth = prev_state.accumulated_depth + weight * point.t;
    new_state.depth_weight = prev_state.depth_weight + weight;

    // Update current position
    new_state.t_current = point.t;

    // Distortion loss accumulation
    // distortion = sum_i sum_j w_i * w_j * |t_i - t_j|
    float prev_cum_weight = prev_state.weight_accum.x;
    float prev_weighted_t = prev_state.weight_accum.y;

    new_state.distortion_accum.x = prev_state.distortion_accum.x +
        weight * weight * dt +
        2.0f * weight * (prev_cum_weight * point.t - prev_weighted_t);
    new_state.distortion_accum.y = prev_state.distortion_accum.y;

    new_state.weight_accum.x = prev_cum_weight + weight;
    new_state.weight_accum.y = prev_weighted_t + weight * point.t;

    return new_state;
}

// Check if ray should terminate early
bool should_terminate(RenderState state) {
    return state.log_transmittance < LOG_TRANSMITTANCE_CUTOFF;
}

// Finalize render output
float4 finalize_color(RenderState state, float3 background_color) {
    float transmittance = safe_exp(state.log_transmittance);
    float3 final_color = state.accumulated_color + transmittance * background_color;
    float alpha = 1.0f - transmittance;
    return float4(final_color, alpha);
}

float compute_depth(RenderState state) {
    if (state.depth_weight > 1e-6f) {
        return state.accumulated_depth / state.depth_weight;
    }
    return 0.0f;
}

float compute_distortion(RenderState state) {
    return state.distortion_accum.x;
}
