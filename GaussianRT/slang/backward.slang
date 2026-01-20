// Backward gradient computation for GaussianRT
// Uses Slang automatic differentiation to generate CUDA backward kernels

import common;

// ============================================================================
// Differentiable Volume Rendering Core
// ============================================================================

// Ray-ellipsoid intersection (differentiable)
[Differentiable]
float2 intersect_ellipsoid(
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

// Compute sample contribution to accumulated color (differentiable)
[Differentiable]
float4 compute_contribution(
    float prev_log_transmittance,
    float prev_t,
    float t_sample,
    float sigma,
    float3 color
) {
    float dt = max(t_sample - prev_t, 1e-6f);
    float tau = sigma * dt;
    float alpha = 1.0f - safe_exp(-tau);
    float transmittance = safe_exp(prev_log_transmittance);
    float weight = alpha * transmittance;

    float3 color_contrib = weight * color;
    float new_log_T = prev_log_transmittance - tau;

    return float4(color_contrib, new_log_T);
}

// Evaluate SH color (differentiable, simplified to DC for now)
[Differentiable]
float3 eval_sh_dc(float3 coeff0) {
    // DC term only: color = sigmoid(SH_C0 * coeff0)
    float3 raw = SH_C0 * coeff0;
    return float3(
        1.0f / (1.0f + exp(-raw.x)),
        1.0f / (1.0f + exp(-raw.y)),
        1.0f / (1.0f + exp(-raw.z))
    );
}

// ============================================================================
// Complete differentiable forward for a single sample
// ============================================================================

[Differentiable]
float4 render_single_sample(
    float3 ray_origin,
    float3 ray_direction,
    float3 center,
    float3 scale,
    float4 rotation,
    float opacity,
    float3 sh_coeff0,  // DC coefficient only for simplicity
    float prev_log_T,
    float prev_t
) {
    // Compute intersection
    float2 t_vals = intersect_ellipsoid(ray_origin, ray_direction, center, scale, rotation);
    float t_sample = (t_vals.x + t_vals.y) * 0.5f;

    // Compute color from SH
    float3 color = eval_sh_dc(sh_coeff0);

    // Compute contribution
    return compute_contribution(prev_log_T, prev_t, t_sample, opacity, color);
}

// ============================================================================
// CUDA Kernel for backward pass
// ============================================================================

// Buffer declarations for CUDA kernel
// Input buffers (read-only)
StructuredBuffer<float> g_final_states;      // [num_rays * 11]
StructuredBuffer<int> g_sample_counts;       // [num_rays]
StructuredBuffer<int> g_sample_indices;      // [num_rays * max_samples]

StructuredBuffer<float> g_positions;         // [N * 3]
StructuredBuffer<float> g_scales;            // [N * 3]
StructuredBuffer<float> g_rotations;         // [N * 4]
StructuredBuffer<float> g_opacities;         // [N]
StructuredBuffer<float> g_features;          // [N * feature_dim]

StructuredBuffer<float> g_ray_origins;       // [num_rays * 3]
StructuredBuffer<float> g_ray_directions;    // [num_rays * 3]

StructuredBuffer<float> g_grad_colors;       // [num_rays * 4]
StructuredBuffer<float> g_grad_depths;       // [num_rays]

// Output buffers (atomic add)
RWStructuredBuffer<float> g_grad_positions;  // [N * 3]
RWStructuredBuffer<float> g_grad_scales;     // [N * 3]
RWStructuredBuffer<float> g_grad_rotations;  // [N * 4]
RWStructuredBuffer<float> g_grad_opacities;  // [N]
RWStructuredBuffer<float> g_grad_features;   // [N * feature_dim]
RWStructuredBuffer<float> g_grad_ray_origins;    // [num_rays * 3]
RWStructuredBuffer<float> g_grad_ray_directions; // [num_rays * 3]

// Parameters
uniform uint g_num_rays;
uniform uint g_max_samples;
uniform uint g_feature_dim;
uniform float g_t_min;

// Helper to load float3 from buffer
float3 load_float3(StructuredBuffer<float> buf, uint idx) {
    uint base = idx * 3;
    return float3(buf[base], buf[base + 1], buf[base + 2]);
}

float4 load_float4(StructuredBuffer<float> buf, uint idx) {
    uint base = idx * 4;
    return float4(buf[base], buf[base + 1], buf[base + 2], buf[base + 3]);
}

// Atomic add helpers
void atomic_add_float3(RWStructuredBuffer<float> buf, uint idx, float3 val) {
    uint base = idx * 3;
    InterlockedAddF32(buf, base + 0, val.x);
    InterlockedAddF32(buf, base + 1, val.y);
    InterlockedAddF32(buf, base + 2, val.z);
}

void atomic_add_float4(RWStructuredBuffer<float> buf, uint idx, float4 val) {
    uint base = idx * 4;
    InterlockedAddF32(buf, base + 0, val.x);
    InterlockedAddF32(buf, base + 1, val.y);
    InterlockedAddF32(buf, base + 2, val.z);
    InterlockedAddF32(buf, base + 3, val.w);
}

// Main backward kernel
[CUDAKernel]
void backward_kernel(uint3 tid : SV_DispatchThreadID) {
    uint ray_idx = tid.x;
    if (ray_idx >= g_num_rays) return;

    // Load upstream gradient
    float4 dL_dcolor = load_float4(g_grad_colors, ray_idx);
    float dL_ddepth = g_grad_depths[ray_idx];

    // Load ray
    float3 ray_origin = load_float3(g_ray_origins, ray_idx);
    float3 ray_direction = load_float3(g_ray_directions, ray_idx);

    // Load sample count
    int num_samples = g_sample_counts[ray_idx];
    if (num_samples == 0) return;

    // Load final state
    uint state_base = ray_idx * 11;
    float final_log_T = g_final_states[state_base + 0];
    float final_depth = g_final_states[state_base + 4];
    float final_depth_weight = max(g_final_states[state_base + 5], 1e-6f);

    // Gradient accumulators
    float3 dL_dray_origin_accum = float3(0.0f);
    float3 dL_dray_direction_accum = float3(0.0f);

    // Gradient w.r.t. accumulated color (from output)
    float3 dL_daccum_color = dL_dcolor.xyz;

    // Gradient w.r.t. alpha (from output)
    float dL_dalpha_out = dL_dcolor.w;

    // Track state for backward iteration
    float current_log_T = final_log_T;

    // Process samples in reverse order
    for (int i = num_samples - 1; i >= 0; --i) {
        int element_id = g_sample_indices[ray_idx * g_max_samples + i];
        if (element_id < 0) continue;

        // Load element data
        float3 center = load_float3(g_positions, element_id);
        float3 scale = load_float3(g_scales, element_id);
        float4 rotation = load_float4(g_rotations, element_id);
        float opacity = g_opacities[element_id];

        // Load DC SH coefficient
        uint feat_base = element_id * g_feature_dim;
        float3 sh_coeff0 = float3(
            g_features[feat_base + 0],
            g_features[feat_base + 1],
            g_features[feat_base + 2]
        );

        // Estimate prev_t (approximation for reconstruction)
        float prev_t = g_t_min;

        // ---- Forward recomputation for this sample ----
        float2 t_vals = intersect_ellipsoid(ray_origin, ray_direction, center, scale, rotation);
        float t_sample = (t_vals.x + t_vals.y) * 0.5f;
        float3 color = eval_sh_dc(sh_coeff0);

        float dt = max(t_sample - prev_t, 1e-6f);
        float tau = opacity * dt;
        float alpha = 1.0f - safe_exp(-tau);
        float prev_log_T = current_log_T + tau;
        float prev_transmittance = safe_exp(prev_log_T);
        float weight = alpha * prev_transmittance;

        // ---- Compute output gradient for this sample ----
        // Output of render_single_sample is float4(weight * color, new_log_T)
        // We need gradient w.r.t. this output
        float4 dL_doutput;
        dL_doutput.xyz = dL_daccum_color;  // gradient flows to color contribution
        dL_doutput.w = 0.0f;  // Will be updated through chain rule

        // ---- Use autodiff to compute parameter gradients ----
        // Create differential pairs for inputs
        var dp_ray_origin = diffPair(ray_origin, float3(0.0f));
        var dp_ray_direction = diffPair(ray_direction, float3(0.0f));
        var dp_center = diffPair(center, float3(0.0f));
        var dp_scale = diffPair(scale, float3(0.0f));
        var dp_rotation = diffPair(rotation, float4(0.0f));
        var dp_opacity = diffPair(opacity, 0.0f);
        var dp_sh_coeff0 = diffPair(sh_coeff0, float3(0.0f));
        var dp_prev_log_T = diffPair(prev_log_T, 0.0f);
        var dp_prev_t = diffPair(prev_t, 0.0f);

        // Call backward differentiation
        bwd_diff(render_single_sample)(
            dp_ray_origin,
            dp_ray_direction,
            dp_center,
            dp_scale,
            dp_rotation,
            dp_opacity,
            dp_sh_coeff0,
            dp_prev_log_T,
            dp_prev_t,
            dL_doutput
        );

        // ---- Accumulate gradients ----
        atomic_add_float3(g_grad_positions, element_id, dp_center.d);
        atomic_add_float3(g_grad_scales, element_id, dp_scale.d);
        atomic_add_float4(g_grad_rotations, element_id, dp_rotation.d);
        InterlockedAddF32(g_grad_opacities, element_id, dp_opacity.d);

        // SH coefficient gradient
        InterlockedAddF32(g_grad_features, feat_base + 0, dp_sh_coeff0.d.x);
        InterlockedAddF32(g_grad_features, feat_base + 1, dp_sh_coeff0.d.y);
        InterlockedAddF32(g_grad_features, feat_base + 2, dp_sh_coeff0.d.z);

        // Ray gradients
        dL_dray_origin_accum += dp_ray_origin.d;
        dL_dray_direction_accum += dp_ray_direction.d;

        // Update for next iteration
        current_log_T = prev_log_T;
    }

    // Store ray gradients
    atomic_add_float3(g_grad_ray_origins, ray_idx, dL_dray_origin_accum);
    atomic_add_float3(g_grad_ray_directions, ray_idx, dL_dray_direction_accum);
}

// Entry point for compilation
[CUDAKernel]
[CudaDeviceExport]
void gaussianrt_backward(uint3 tid : SV_DispatchThreadID) {
    backward_kernel(tid);
}
