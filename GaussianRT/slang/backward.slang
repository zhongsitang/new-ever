// Backward gradient computation for GaussianRT
// Uses Slang automatic differentiation for volume rendering gradients

import common;

// ============================================================================
// Kernel Parameters (passed via uniform buffer)
// ============================================================================

struct BackwardParams {
    uint num_rays;
    uint max_samples;
    uint sh_degree;
    uint feature_dim;
    float t_min;
    float t_max;
};

ConstantBuffer<BackwardParams> params;

// ============================================================================
// Data Buffers
// ============================================================================

// Forward pass outputs (read-only)
StructuredBuffer<float> final_states;          // [num_rays * 11]
StructuredBuffer<float> last_points;           // [num_rays * 5]
StructuredBuffer<int> sample_counts;           // [num_rays]
StructuredBuffer<int> sample_indices;          // [num_rays * max_samples]

// Scene data (read-only)
StructuredBuffer<float> positions;             // [N * 3]
StructuredBuffer<float> scales;                // [N * 3]
StructuredBuffer<float> rotations;             // [N * 4]
StructuredBuffer<float> opacities;             // [N]
StructuredBuffer<float> features;              // [N * feature_dim]

// Ray data (read-only)
StructuredBuffer<float> ray_origins;           // [num_rays * 3]
StructuredBuffer<float> ray_directions;        // [num_rays * 3]

// Upstream gradients (read-only)
StructuredBuffer<float> grad_colors;           // [num_rays * 4]
StructuredBuffer<float> grad_depths;           // [num_rays]
StructuredBuffer<float> grad_distortions;      // [num_rays]

// Gradient outputs (write, atomic add)
RWStructuredBuffer<float> grad_positions;      // [N * 3]
RWStructuredBuffer<float> grad_scales;         // [N * 3]
RWStructuredBuffer<float> grad_rotations;      // [N * 4]
RWStructuredBuffer<float> grad_opacities;      // [N]
RWStructuredBuffer<float> grad_features;       // [N * feature_dim]
RWStructuredBuffer<float> grad_ray_origins;    // [num_rays * 3]
RWStructuredBuffer<float> grad_ray_directions; // [num_rays * 3]

// ============================================================================
// Helper Functions for Data Loading
// ============================================================================

float3 load_float3_buffer(StructuredBuffer<float> buffer, uint idx) {
    uint base = idx * 3;
    return float3(buffer[base], buffer[base + 1], buffer[base + 2]);
}

float4 load_float4_buffer(StructuredBuffer<float> buffer, uint idx) {
    uint base = idx * 4;
    return float4(buffer[base], buffer[base + 1], buffer[base + 2], buffer[base + 3]);
}

void load_sh_coefficients_buffer(uint element_id, out float3 coeffs[SH_COEFF_COUNT]) {
    uint base_offset = element_id * params.feature_dim;
    for (uint i = 0; i < SH_COEFF_COUNT; ++i) {
        uint offset = base_offset + i * 3;
        coeffs[i] = float3(features[offset], features[offset + 1], features[offset + 2]);
    }
}

RenderState load_render_state_buffer(uint ray_idx) {
    uint base = ray_idx * 11;
    RenderState state;
    state.log_transmittance = final_states[base + 0];
    state.accumulated_color = float3(
        final_states[base + 1],
        final_states[base + 2],
        final_states[base + 3]
    );
    state.accumulated_depth = final_states[base + 4];
    state.depth_weight = final_states[base + 5];
    state.t_current = final_states[base + 6];
    state.distortion_accum = float2(final_states[base + 7], final_states[base + 8]);
    state.weight_accum = float2(final_states[base + 9], final_states[base + 10]);
    return state;
}

// ============================================================================
// Differentiable Volume Rendering Functions
// ============================================================================

[Differentiable]
float2 compute_ray_ellipsoid_intersection(
    float3 ray_origin,
    float3 ray_direction,
    float3 center,
    float3 scale,
    float4 rotation
) {
    float3 local_origin = ray_origin - center;
    local_origin = rotate_by_quaternion_inverse(local_origin, rotation);
    float3 local_direction = rotate_by_quaternion_inverse(ray_direction, rotation);

    float3 scaled_origin = local_origin / scale;
    float3 scaled_direction = local_direction / scale;

    float a = dot(scaled_direction, scaled_direction);
    float b = 2.0f * dot(scaled_origin, scaled_direction);
    float c = dot(scaled_origin, scaled_origin) - 1.0f;

    float discriminant = b * b - 4.0f * a * c;
    float sqrt_disc = sqrt(max(discriminant, 1e-8f));

    float t_near = (-b - sqrt_disc) / (2.0f * a);
    float t_far = (-b + sqrt_disc) / (2.0f * a);

    return float2(t_near, t_far);
}

[Differentiable]
float3 compute_sh_color(float3 direction, float3 coeffs[SH_COEFF_COUNT], uint sh_degree) {
    return eval_sh_color(direction, coeffs, sh_degree);
}

// Compute contribution from a single sample to the final color
// This is the core function we need to differentiate
[Differentiable]
float4 compute_sample_contribution(
    float prev_log_transmittance,
    float prev_t,
    float sample_t,
    float sigma,
    float3 color
) {
    float dt = max(sample_t - prev_t, 1e-6f);
    float tau = sigma * dt;  // optical thickness
    float alpha = 1.0f - safe_exp(-tau);
    float transmittance = safe_exp(prev_log_transmittance);
    float weight = alpha * transmittance;

    // Returns: (color_contribution.xyz, new_log_transmittance)
    return float4(weight * color, prev_log_transmittance - tau);
}

// ============================================================================
// Backward Kernel Implementation
// ============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void backward_main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint ray_idx = dispatch_id.x;
    if (ray_idx >= params.num_rays) return;

    // Load upstream gradients
    uint grad_base = ray_idx * 4;
    float4 dL_dcolor = float4(
        grad_colors[grad_base],
        grad_colors[grad_base + 1],
        grad_colors[grad_base + 2],
        grad_colors[grad_base + 3]
    );
    float dL_ddepth = grad_depths[ray_idx];
    float dL_ddistortion = grad_distortions[ray_idx];

    // Load ray data
    float3 ray_origin = load_float3_buffer(ray_origins, ray_idx);
    float3 ray_direction = load_float3_buffer(ray_directions, ray_idx);

    // Load sample count
    int num_samples = sample_counts[ray_idx];
    if (num_samples == 0) return;

    // Load final state for reference
    RenderState final_state = load_render_state_buffer(ray_idx);

    // Gradient accumulators for ray
    float3 dL_dray_origin_accum = float3(0.0f);
    float3 dL_dray_direction_accum = float3(0.0f);

    // For backward pass, we need to reconstruct intermediate states
    // and propagate gradients from output to inputs

    // Gradient w.r.t. final accumulated color
    float3 dL_daccum_color = dL_dcolor.xyz;

    // Gradient w.r.t. depth (depth = accumulated_depth / depth_weight)
    float depth_weight_safe = max(final_state.depth_weight, 1e-6f);
    float dL_daccum_depth = dL_ddepth / depth_weight_safe;
    float dL_ddepth_weight = -dL_ddepth * final_state.accumulated_depth /
                             (depth_weight_safe * depth_weight_safe);

    // Backward through samples in reverse order
    // We need to track log_transmittance as we go backward
    float current_log_T = final_state.log_transmittance;
    float current_t = final_state.t_current;

    // Gradient w.r.t. log_transmittance (from alpha output)
    float dL_dlog_T = dL_dcolor.w * (-safe_exp(final_state.log_transmittance));

    for (int i = num_samples - 1; i >= 0; --i) {
        int element_id = sample_indices[ray_idx * params.max_samples + i];
        if (element_id < 0) continue;

        // Load element data
        float3 center = load_float3_buffer(positions, element_id);
        float3 scale = load_float3_buffer(scales, element_id);
        float4 rotation = load_float4_buffer(rotations, element_id);
        float opacity = opacities[element_id];

        float3 sh_coeffs[SH_COEFF_COUNT];
        load_sh_coefficients_buffer(element_id, sh_coeffs);

        // Recompute intersection and sample point
        float2 t_vals = compute_ray_ellipsoid_intersection(
            ray_origin, ray_direction, center, scale, rotation
        );
        float t_sample = (t_vals.x + t_vals.y) * 0.5f;

        // Recompute color
        float3 sample_color = compute_sh_color(-ray_direction, sh_coeffs, params.sh_degree);

        // Estimate previous state (before this sample)
        // prev_t is the t value before this sample
        float prev_t = (i > 0) ? t_sample - 0.01f : params.t_min;  // Approximation

        // Compute this sample's contribution parameters
        float dt = max(t_sample - prev_t, 1e-6f);
        float tau = opacity * dt;
        float alpha = 1.0f - safe_exp(-tau);
        float prev_log_T = current_log_T + tau;  // Reconstruct previous log_T
        float prev_transmittance = safe_exp(prev_log_T);
        float weight = alpha * prev_transmittance;

        // -------- Compute gradients --------

        // Gradient of color contribution: contrib = weight * sample_color
        // dL/d(sample_color) = weight * dL/d(accum_color)
        float3 dL_dsample_color = weight * dL_daccum_color;

        // dL/d(weight) = dot(sample_color, dL/d(accum_color)) + t_sample * dL/d(depth) + dL/d(depth_weight)
        float dL_dweight = dot(sample_color, dL_daccum_color) +
                           t_sample * dL_daccum_depth +
                           dL_ddepth_weight;

        // weight = alpha * transmittance = alpha * exp(prev_log_T)
        // dL/d(alpha) = transmittance * dL/d(weight)
        float dL_dalpha = prev_transmittance * dL_dweight;

        // dL/d(prev_log_T) from weight = alpha * exp(prev_log_T)
        float dL_dprev_log_T_from_weight = weight * dL_dweight;

        // alpha = 1 - exp(-tau) => d(alpha)/d(tau) = exp(-tau) = 1 - alpha
        // dL/d(tau) = (1 - alpha) * dL/d(alpha)
        float dL_dtau = (1.0f - alpha) * dL_dalpha;

        // tau = opacity * dt
        // dL/d(opacity) = dt * dL/d(tau)
        float dL_dopacity = dt * dL_dtau;

        // dL/d(dt) = opacity * dL/d(tau)
        float dL_ddt = opacity * dL_dtau;

        // dt = t_sample - prev_t
        // dL/d(t_sample) = dL/d(dt) + weight * dL/d(depth)
        float dL_dt_sample = dL_ddt + weight * dL_daccum_depth;

        // -------- Backward through intersection --------

        // Use autodiff for intersection gradients
        var center_dp = diffPair(center, float3(0.0f));
        var scale_dp = diffPair(scale, float3(0.0f));
        var rotation_dp = diffPair(rotation, float4(0.0f));
        var ray_origin_dp = diffPair(ray_origin, float3(0.0f));
        var ray_direction_dp = diffPair(ray_direction, float3(0.0f));

        // t_sample = (t_near + t_far) / 2
        // dL/d(t_near) = dL/d(t_sample) * 0.5
        // dL/d(t_far) = dL/d(t_sample) * 0.5
        float2 dL_dt_vals = float2(dL_dt_sample * 0.5f, dL_dt_sample * 0.5f);

        bwd_diff(compute_ray_ellipsoid_intersection)(
            ray_origin_dp, ray_direction_dp,
            center_dp, scale_dp, rotation_dp,
            dL_dt_vals
        );

        // -------- Backward through SH color --------

        // Simplified SH gradient (DC term dominates)
        // For DC: color = sigmoid(SH_C0 * coeff[0])
        // Full SH backward would require more complex handling
        float3 dL_dcoeff0 = dL_dsample_color * SH_C0;

        // -------- Accumulate gradients --------

        // Element gradients (atomic add)
        InterlockedAdd(grad_positions, element_id * 3 + 0, center_dp.d.x);
        InterlockedAdd(grad_positions, element_id * 3 + 1, center_dp.d.y);
        InterlockedAdd(grad_positions, element_id * 3 + 2, center_dp.d.z);

        InterlockedAdd(grad_scales, element_id * 3 + 0, scale_dp.d.x);
        InterlockedAdd(grad_scales, element_id * 3 + 1, scale_dp.d.y);
        InterlockedAdd(grad_scales, element_id * 3 + 2, scale_dp.d.z);

        InterlockedAdd(grad_rotations, element_id * 4 + 0, rotation_dp.d.x);
        InterlockedAdd(grad_rotations, element_id * 4 + 1, rotation_dp.d.y);
        InterlockedAdd(grad_rotations, element_id * 4 + 2, rotation_dp.d.z);
        InterlockedAdd(grad_rotations, element_id * 4 + 3, rotation_dp.d.w);

        InterlockedAdd(grad_opacities, element_id, dL_dopacity);

        // SH coefficient gradients (simplified - DC term only)
        uint feat_base = element_id * params.feature_dim;
        InterlockedAdd(grad_features, feat_base + 0, dL_dcoeff0.x);
        InterlockedAdd(grad_features, feat_base + 1, dL_dcoeff0.y);
        InterlockedAdd(grad_features, feat_base + 2, dL_dcoeff0.z);

        // Ray gradients
        dL_dray_origin_accum += ray_origin_dp.d;
        dL_dray_direction_accum += ray_direction_dp.d;

        // Update for next iteration (going backward)
        current_log_T = prev_log_T;
        current_t = prev_t;
        dL_dlog_T += dL_dprev_log_T_from_weight;
    }

    // Store ray gradients
    InterlockedAdd(grad_ray_origins, ray_idx * 3 + 0, dL_dray_origin_accum.x);
    InterlockedAdd(grad_ray_origins, ray_idx * 3 + 1, dL_dray_origin_accum.y);
    InterlockedAdd(grad_ray_origins, ray_idx * 3 + 2, dL_dray_origin_accum.z);

    InterlockedAdd(grad_ray_directions, ray_idx * 3 + 0, dL_dray_direction_accum.x);
    InterlockedAdd(grad_ray_directions, ray_idx * 3 + 1, dL_dray_direction_accum.y);
    InterlockedAdd(grad_ray_directions, ray_idx * 3 + 2, dL_dray_direction_accum.z);
}
