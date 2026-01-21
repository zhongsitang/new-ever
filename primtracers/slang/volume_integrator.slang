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
// Volume Rendering Integration State Machine
// =============================================================================
// Implements front-to-back volume rendering with:
// - Transmittance tracking (log-space for numerical stability)
// - Color accumulation with proper alpha weighting
// - Depth computation for expected depth maps
// - Distortion loss for regularization

#define EPS 1e-18
import safe_math;
#define PRE_MULTI 1000
#define LADDER_P -0.1

// -----------------------------------------------------------------------------
// Volume Rendering State
// -----------------------------------------------------------------------------

struct VolumeState : IDifferentiable
{
    // Distortion loss computation
    float2 distortion_parts;
    float2 cum_sum;

    // Depth accumulation (stored in padding[0])
    float3 padding;

    // Current ray parameter
    float t;

    // Accumulated density and density-weighted color: (σ, σ*r, σ*g, σ*b)
    float4 accumulated_drgb;

    // Volume rendering state
    float log_transmittance;  // log(T) for numerical stability
    float3 color;             // Accumulated color C = Σ w_i * c_i
};

VolumeState make_initial_state() {
    return {
        float2(0.f),    // distortion_parts
        float2(0.f),    // cum_sum
        float3(0.f),    // padding (depth in [0])
        0.f,            // t
        float4(0.0f),   // accumulated_drgb

        0.0,            // log_transmittance
        float3(0.0f),   // color
    };
}

// -----------------------------------------------------------------------------
// Sample Point: represents a surface intersection contribution
// -----------------------------------------------------------------------------

struct SamplePoint : IDifferentiable
{
    float t;        // Ray parameter at intersection
    float4 delta;   // Delta contribution: (±σ, ±σ*r, ±σ*g, ±σ*b)
                    // Positive for entry, negative for exit
}

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

bool AlmostEqualRelative(float A, float B, float maxRelDiff = 1.2e-7f)
{
    return abs(A - B) <= max(abs(A), abs(B)) * maxRelDiff;
}

VolumeState to_dual(in VolumeState state, in SamplePoint sample)
{
    VolumeState dual_state = state;
    return dual_state;
}

VolumeState from_dual(in VolumeState state, in SamplePoint sample)
{
    VolumeState dual_state = state;
    return dual_state;
}

// -----------------------------------------------------------------------------
// Inverse Update (for backward pass)
// -----------------------------------------------------------------------------

VolumeState inverse_update_dual(
    in VolumeState new_state,
    in SamplePoint new_sample,
    in SamplePoint sample,
    in float t_min,
    in float t_max)
{
    const float t = sample.t;
    const float dt = max(new_state.t - t, 0.f);

    VolumeState state = {};
    state.accumulated_drgb = new_state.accumulated_drgb - new_sample.delta;
    state.t = t;

    float4 drgb = state.accumulated_drgb;
    let avg = drgb;
    float area = max(avg.x * dt, 0.f);
    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    state.log_transmittance = max(new_state.log_transmittance - area, 0.f);
    const float weight = clip((1-safe_exp(-area)) * safe_exp(-state.log_transmittance), 0.f, 1.f);
    state.color = new_state.color - weight * rgb_norm;

    float alpha = -safe_expm1(-area);
    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD) {
        segment_depth_val = alpha * t + (1-alpha) * state.t;
    } else {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area))
                          - (t + t_min) * safe_exp(-area)
                          + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.padding[0] = state.padding[0] - safe_exp(-state.log_transmittance) * segment_depth;

    // Distortion loss
    let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
    state.cum_sum.x = new_state.cum_sum.x - weight;
    state.cum_sum.y = new_state.cum_sum.y - weight * m;
    state.distortion_parts.x = new_state.distortion_parts.x - 2 * weight * m * state.cum_sum.x;
    state.distortion_parts.y = new_state.distortion_parts.y - 2 * weight * state.cum_sum.y;

    return state;
}

// -----------------------------------------------------------------------------
// Forward Update: integrate a sample point into volume state
// -----------------------------------------------------------------------------

[Differentiable]
VolumeState update(
    in VolumeState state,
    in SamplePoint sample,
    no_diff in float t_min,
    no_diff in float t_max,
    no_diff in float max_prim_size)
{
    const float t = sample.t;
    const float dt = max(t - state.t, 0.f);

    VolumeState new_state;
    new_state.accumulated_drgb = state.accumulated_drgb + sample.delta;
    new_state.t = t;

    float4 drgb = state.accumulated_drgb;
    let avg = drgb;
    let area = max(avg.x * dt, 0.f);
    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    new_state.log_transmittance = max(area + state.log_transmittance, 0.f);
    float alpha = -safe_expm1(-area);
    const float weight = clip(alpha * safe_exp(-state.log_transmittance), 0.f, 1.f);
    new_state.color = state.color + weight * rgb_norm;

    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD) {
        segment_depth_val = alpha * t + (1-alpha) * state.t;
    } else {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area))
                          - (t + t_min) * safe_exp(-area)
                          + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.padding[0] = state.padding[0] + safe_exp(-state.log_transmittance) * segment_depth;

    // Distortion loss
    let m = tukey_power_ladder((new_state.t+state.t)/2 * PRE_MULTI, LADDER_P);
    new_state.distortion_parts.x = state.distortion_parts.x + 2 * weight * m * state.cum_sum.x;
    new_state.distortion_parts.y = state.distortion_parts.y + 2 * weight * state.cum_sum.y;
    new_state.cum_sum.x = state.cum_sum.x + weight;
    new_state.cum_sum.y = state.cum_sum.y + weight * m;

    return new_state;
}

// -----------------------------------------------------------------------------
// Output Extraction
// -----------------------------------------------------------------------------

struct RenderOutput: IDifferentiable {
    float3 C;
    float depth;
    float distortion_loss;
};

[Differentiable]
RenderOutput extract_color(in VolumeState state, in float tmin) {
    return {
        state.color,
        state.padding[0],
        state.distortion_parts.x - state.distortion_parts.y
    };
}
