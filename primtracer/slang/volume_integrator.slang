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
// Volume Integrator for Ellipsoid Primitive Ray Tracing
// =============================================================================
//
// Volume rendering following instant-ngp style:
//   alpha = 1 - exp(-sigma * dt)      // Beer-Lambert
//   weight = T * alpha                 // T = 1 - accumulated_alpha
//   color += weight * rgb
//   accumulated_alpha += weight
//
// For ellipsoid primitives, we process boundary events:
//   - Entry: sigma += density, weighted_rgb += density * rgb
//   - Exit:  sigma -= density, weighted_rgb -= density * rgb
// =============================================================================

import math_utils;

// =============================================================================
// Core Data Structures
// =============================================================================

// Clamp sigma*dt upper bound to avoid exp overflow and 1-alpha underflow.
// exp(-16) ≈ 1.1e-7, close to TINY_VAL.
static const float SIGMA_DT_MAX = 16.0f;

/// Volume rendering state per ray (instant-ngp style)
///
/// Memory layout: 48 bytes, 16-byte aligned
struct IntegratorState : IDifferentiable
{
    float4 color;        // (r, g, b, accumulated_alpha)
    float4 contrib;      // (sigma, weighted_r, weighted_g, weighted_b)
    float t;             // current ray parameter
    float depth_num;     // depth numerator for expected depth
    float _pad0;
    float _pad1;
};

/// Sample point at ellipsoid boundary crossing.
///
/// At entry: delta = (+density, +density*r, +density*g, +density*b)
/// At exit:  delta = (-density, -density*r, -density*g, -density*b)
struct SamplePoint : IDifferentiable
{
    float t;             // ray parameter at boundary
    float4 delta;        // (delta_sigma, delta_r, delta_g, delta_b)
}

// =============================================================================
// Volume Integration
// =============================================================================

/// Mean position offset within segment [t, t+dt] under exponential attenuation.
///
/// offset = dt * (1/x - 1/(e^x - 1)),  where x = sigma * dt
///
/// Taylor expansion for |x| < 1e-3:
///   offset = dt * (1/2 - x/12 + x²/720 - ...)
///
/// Asymptotic: x → 0: dt/2,  x → ∞: dt/x = 1/sigma
[Differentiable]
float calc_depth_offset(float sigma, float dt)
{
    float x = min(sigma * dt, SIGMA_DT_MAX);

    if (abs(x) < 1e-3f) {
        return dt * (0.5f - x * (1.0f / 12.0f) + x * x * (1.0f / 720.0f));
    }

    return dt * (1.0f / x - 1.0f / expm1(x));
}

/// Process one volume integration step at a boundary crossing.
///
/// For segment [state.t, sample.t] with constant density sigma:
///   alpha = 1 - exp(-sigma * dt)
///   weight = T * alpha, where T = 1 - accum_alpha
///   color.rgb += weight * rgb
///   color.a += weight
[Differentiable]
IntegratorState integrate_step(
    in IntegratorState state,
    in SamplePoint sample)
{
    IntegratorState new_state;

    // --- Step 1: Integrate segment [state.t, sample.t] ---
    float dt = max(sample.t - state.t, 0.0f);

    if (dt < TINY_VAL) {
        new_state.color = state.color;
        new_state.depth_num = state.depth_num;
    } else {
        // Segment properties
        float sigma = state.contrib.x;
        float3 rgb = safe_div(state.contrib.yzw, sigma);
        float sigma_dt = min(sigma * dt, SIGMA_DT_MAX);
        float alpha = -expm1(-sigma_dt);  // 1 - exp(-sigma * dt)

        // Transmittance T = 1 - accum_alpha
        float T = 1.0f - state.color.a;
        float weight = T * alpha;

        // Accumulate color and alpha
        new_state.color.rgb = state.color.rgb + weight * rgb;
        new_state.color.a = state.color.a + weight;

        // Depth accumulation (exact mean position within segment)
        float t_mean = state.t + calc_depth_offset(sigma, dt);
        new_state.depth_num = state.depth_num + weight * t_mean;
    }

    // --- Step 2: Update density field for next segment ---
    new_state.contrib = state.contrib + sample.delta;
    new_state.t = sample.t;

    return new_state;
}

/// Reconstruct previous state from current state (inverse of integrate_step).
/// Used in backward pass to recover intermediate states without storing them.
///
/// Forward pass:
///   weight = T_prev * alpha
///   T_new = T_prev - weight = T_prev * (1 - alpha)
///
/// Inverse (solve for T_prev):
///   T_prev = T_new / (1 - alpha)
///   weight = T_prev * alpha
IntegratorState invert_step(
    in IntegratorState new_state,
    in SamplePoint new_sample,
    in SamplePoint prev_sample)
{
    IntegratorState prev_state;

    // --- Undo Step 2: Recover previous contrib and t ---
    prev_state.contrib = new_state.contrib - new_sample.delta;
    prev_state.t = prev_sample.t;

    // --- Undo Step 1: Subtract segment's contribution ---
    float dt = max(new_sample.t - prev_sample.t, 0.0f);

    if (dt < TINY_VAL) {
        prev_state.color = new_state.color;
        prev_state.depth_num = new_state.depth_num;
    } else {
        // Segment properties
        float sigma = prev_state.contrib.x;
        float3 rgb = safe_div(prev_state.contrib.yzw, sigma);
        float sigma_dt = min(sigma * dt, SIGMA_DT_MAX);
        float alpha = -expm1(-sigma_dt);

        // Recover weight
        float T_new = 1.0f - new_state.color.a;
        float T_prev = safe_div(T_new, 1.0f - alpha);
        float weight = T_prev * alpha;

        // Subtract segment contribution
        prev_state.color.rgb = new_state.color.rgb - weight * rgb;
        prev_state.color.a = new_state.color.a - weight;

        float t_mean = prev_sample.t + calc_depth_offset(sigma, dt);
        prev_state.depth_num = new_state.depth_num - weight * t_mean;
    }

    return prev_state;
}

// =============================================================================
// Output Extraction
// =============================================================================

/// Final rendered output for a ray.
struct RenderOutput : IDifferentiable
{
    float4 color; // (r, g, b, alpha)
    float depth;  // Expected depth
};

/// Extract final render output from volume integration state.
[Differentiable]
RenderOutput extract_output(in IntegratorState state)
{
    const float ALPHA_THRESHOLD = 1e-6f;

    float alpha = state.color.a;
    float depth = (alpha < ALPHA_THRESHOLD) ? 0.0f : safe_div(state.depth_num, alpha);

    return { state.color, depth };
}
