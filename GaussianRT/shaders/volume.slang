// Modern Splinetracer - Volume Rendering State Machine
// Core volume rendering logic using dirac deltas
// Apache License 2.0

module volume;

import safe_math;

static const float LOG_CUTOFF = 5.54f;      // Transmittance cutoff
static const float PRE_MULTI = 1000.0f;     // Pre-multiplier for distortion
static const float LADDER_P = -0.1f;        // Tukey ladder parameter

// Volume rendering state - tracks accumulated color and transmittance
public struct VolumeState : IDifferentiable {
    float2 distortion_parts;
    float2 cum_sum;
    float3 padding;
    float t;        // Current ray parameter
    float4 drgb;    // Accumulated density*rgb gradients
    float logT;     // Log transmittance (negative log of T)
    float3 C;       // Accumulated color
};

// Volume sample at a point
public struct VolumeSample : IDifferentiable {
    float t;        // Position along ray
    float4 dirac;   // (density, density*r, density*g, density*b)
};

// Rendered output
public struct VolumeOutput : IDifferentiable {
    float3 C;               // Final color
    float depth;            // Expected depth
    float distortion_loss;  // Distortion regularization
};

// Create empty volume state
public VolumeState make_empty_volume_state() {
    VolumeState state;
    state.distortion_parts = float2(0.0f);
    state.cum_sum = float2(0.0f);
    state.padding = float3(0.0f);
    state.t = 0.0f;
    state.drgb = float4(0.0f);
    state.logT = 0.0f;
    state.C = float3(0.0f);
    return state;
}

// Integrate a volume sample into the state
[Differentiable]
public VolumeState integrate_volume_sample(
    VolumeState state,
    VolumeSample sample,
    float t_min,
    float t_max,
    float max_prim_size) {

    float t = sample.t;
    float dt = max(t - state.t, 0.0f);

    VolumeState new_state;
    new_state.drgb = state.drgb + sample.dirac;
    new_state.t = t;

    float4 drgb = state.drgb;
    float4 avg = drgb;
    float area = max(avg.x * dt, 0.0f);

    float3 rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    new_state.logT = max(area + state.logT, 0.0f);
    float alpha = -safe_expm1(-area);
    float weight = clip(alpha * safe_exp(-state.logT), 0.0f, 1.0f);
    new_state.C = state.C + weight * rgb_norm;

    // Depth computation
    static const float AREA_THRESHOLD = 1e-6f;
    float segment_depth;
    if (avg.x < AREA_THRESHOLD) {
        segment_depth = alpha * t + (1.0f - alpha) * state.t;
    } else {
        segment_depth = safe_div(1.0f, avg.x) * (-safe_expm1(-area))
                      - (t + t_min) * safe_exp(-area)
                      + (state.t + t_min);
    }
    segment_depth = max(segment_depth, 0.0f);
    new_state.padding.x = state.padding.x + safe_exp(-state.logT) * segment_depth;

    // Distortion loss computation
    float m = tukey_power_ladder((new_state.t + state.t) / 2.0f * PRE_MULTI, LADDER_P);
    new_state.distortion_parts.x = state.distortion_parts.x + 2.0f * weight * m * state.cum_sum.x;
    new_state.distortion_parts.y = state.distortion_parts.y + 2.0f * weight * state.cum_sum.y;
    new_state.cum_sum.x = state.cum_sum.x + weight;
    new_state.cum_sum.y = state.cum_sum.y + weight * m;

    return new_state;
}

// Extract final output from volume state
[Differentiable]
public VolumeOutput extract_volume_output(VolumeState state, float tmin) {
    VolumeOutput output;
    output.C = state.C;
    output.depth = state.padding.x;
    output.distortion_loss = state.distortion_parts.x - state.distortion_parts.y;
    return output;
}

// Convert to dual state for backward pass
public VolumeState to_dual_volume_state(VolumeState state, VolumeSample sample) {
    return state;
}
