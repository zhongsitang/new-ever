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

// =============================================================================
// Volume Integration State Machine
// Implements the volume rendering equation with piecewise-constant density
// =============================================================================

#define EPS 1e-18
import safe_math;
#define PRE_MULTI 1000
#define LADDER_P -0.1

// =============================================================================
// IntegrationState - Accumulated state during ray marching
// =============================================================================
struct IntegrationState : IDifferentiable
{
    // Distortion loss components (for regularization)
    float2 distortion_parts;
    float2 cum_sum;

    // Depth accumulator (stored in first component)
    float3 depth_accum;

    // Current ray parameter t
    float t;

    // Accumulated sample: (density, density*R, density*G, density*B)
    float4 sample;

    // Log transmittance: log(T) where T = exp(-integral of density)
    float log_transmittance;

    // Accumulated color from volume rendering equation
    float3 color;
};

// Legacy alias for compatibility
typedef IntegrationState SplineState;

IntegrationState make_empty_state() {
    return {
        float2(0.f),
        float2(0.f),
        float3(0.f),
        0.f,
        float4(0.0f),
        0.0,
        float3(0.0f),
    };
}

// =============================================================================
// Sample - A density/color sample at a ray parameter t
// Represents a Dirac delta contribution to the piecewise-constant density field
// =============================================================================
struct Sample : IDifferentiable
{
    float t;           // Ray parameter where sample is located
    float4 value;      // (density_delta, density*R, density*G, density*B)
}

// Legacy alias for compatibility
typedef Sample ControlPoint;

bool AlmostEqualRelative(float A, float B, float maxRelDiff = 1.2e-7f)
{
    return abs(A - B) <= max(abs(A), abs(B)) * maxRelDiff;
}

IntegrationState to_dual(in IntegrationState state, in Sample sample)
{
    IntegrationState dual_state = state;
    return dual_state;
}

IntegrationState from_dual(in IntegrationState state, in Sample sample)
{
    IntegrationState dual_state = state;
    return dual_state;
}

// =============================================================================
// Inverse update for backward pass
// Computes the previous state given the current state and sample
// =============================================================================
IntegrationState inverse_update_dual(
    in IntegrationState new_state,
    in Sample new_sample,
    in Sample sample,
    in float t_min,
    in float t_max)
{
    const float t = sample.t;
    const float dt = max(new_state.t - t, 0.f);

    IntegrationState state = {};
    state.sample = new_state.sample - new_sample.value;
    state.t = t;

    float4 current_sample = state.sample;
    let avg = current_sample;
    float area = max(avg.x * dt, 0.f);
    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    state.log_transmittance = max(new_state.log_transmittance - area, 0.f);
    const float weight = clip((1 - safe_exp(-area)) * safe_exp(-state.log_transmittance), 0.f, 1.f);

    state.color = new_state.color - weight * rgb_norm;
    float alpha = -safe_expm1(-area);

    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD) {
        segment_depth_val = alpha * t + (1 - alpha) * state.t;
    } else {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area))
                          - (t + t_min) * safe_exp(-area)
                          + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.depth_accum[0] = state.depth_accum[0] - safe_exp(-state.log_transmittance) * segment_depth;

    // Distortion Loss computation
    let m = tukey_power_ladder((new_state.t + state.t) / 2 * PRE_MULTI, LADDER_P);
    state.cum_sum.x = new_state.cum_sum.x - weight;
    state.cum_sum.y = new_state.cum_sum.y - weight * m;
    state.distortion_parts.x = new_state.distortion_parts.x - 2 * weight * m * state.cum_sum.x;
    state.distortion_parts.y = new_state.distortion_parts.y - 2 * weight * state.cum_sum.y;

    return state;
}

// =============================================================================
// Forward update - integrate a new sample into the state
// Implements the volume rendering equation step
// =============================================================================
[Differentiable]
IntegrationState update(
    in IntegrationState state,
    in Sample sample,
    no_diff in float t_min,
    no_diff in float t_max,
    no_diff in float max_prim_size)
{
    const float t = sample.t;
    const float dt = max(t - state.t, 0.f);

    IntegrationState new_state;
    new_state.sample = state.sample + sample.value;
    new_state.t = t;

    float4 current_sample = state.sample;
    let avg = current_sample;
    let area = max(avg.x * dt, 0.f);
    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    // Update transmittance: T_new = T_old * exp(-area)
    new_state.log_transmittance = max(area + state.log_transmittance, 0.f);

    // Compute alpha and weight for this segment
    float alpha = -safe_expm1(-area);
    const float weight = clip(alpha * safe_exp(-state.log_transmittance), 0.f, 1.f);

    // Accumulate color: C_new = C_old + weight * color
    new_state.color = state.color + weight * rgb_norm;

    // Compute depth contribution for this segment
    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD) {
        segment_depth_val = alpha * t + (1 - alpha) * state.t;
    } else {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area))
                          - (t + t_min) * safe_exp(-area)
                          + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.depth_accum[0] = state.depth_accum[0] + safe_exp(-state.log_transmittance) * segment_depth;

    // Distortion Loss computation for regularization
    let m = tukey_power_ladder((new_state.t + state.t) / 2 * PRE_MULTI, LADDER_P);
    new_state.distortion_parts.x = state.distortion_parts.x + 2 * weight * m * state.cum_sum.x;
    new_state.distortion_parts.y = state.distortion_parts.y + 2 * weight * state.cum_sum.y;
    new_state.cum_sum.x = state.cum_sum.x + weight;
    new_state.cum_sum.y = state.cum_sum.y + weight * m;

    return new_state;
}

// =============================================================================
// RenderOutput - Final rendered values extracted from integration state
// =============================================================================
struct RenderOutput : IDifferentiable {
    float3 color;
    float depth;
    float distortion_loss;
};

// Legacy alias for compatibility
typedef RenderOutput SplineOutput;

[Differentiable]
RenderOutput extract_color(in IntegrationState state, in float tmin) {
    return {
        state.color,
        state.depth_accum[0],
        state.distortion_parts.x - state.distortion_parts.y
    };
}
