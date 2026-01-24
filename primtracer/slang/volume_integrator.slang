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
// This module implements volume rendering integration for ellipsoid primitives.
// Each ellipsoid contributes to the integral when a ray passes through it:
// - At entry point: adds +density to the accumulated density field
// - At exit point: adds -density to the accumulated density field
//
// The accumulated density between entry/exit points determines the opacity
// contribution of each ellipsoid segment along the ray.
// =============================================================================

#define EPS 1e-18
import safe_math;

// =============================================================================
// Core Data Structures
// =============================================================================

/// State for volume rendering integration along a ray.
///
/// This tracks both the integration state (accumulated density-weighted values)
/// and the volume rendering state (transmittance and accumulated color).
///
/// Memory layout (12 floats, 48 bytes, 16-byte aligned):
///   [0-3]:  accumulated_contrib (float4) - density, r*density, g*density, b*density
///   [4-7]:  C_logT (float4) - C.x, C.y, C.z, logT
///   [8-11]: scalars (float4) - depth_accum, t, pad, pad
struct IntegratorState : IDifferentiable
{
    // Accumulated density-weighted contributions: (density, r*density, g*density, b*density)
    // This accumulates the "delta" contributions from ellipsoid boundaries.
    float4 accumulated_contrib;

    // Combined: accumulated color (xyz) + log transmittance (w)
    float4 C_logT;

    // Scalars packed into float4: depth_accum (x), t (y), unused (z, w)
    float4 scalars;

    // Accessor properties
    property float3 C { get { return C_logT.xyz; } set { C_logT.xyz = newValue; } }
    property float logT { get { return C_logT.w; } set { C_logT.w = newValue; } }
    property float depth_accum { get { return scalars.x; } set { scalars.x = newValue; } }
    property float t { get { return scalars.y; } set { scalars.y = newValue; } }
};

/// Create initial volume state for a new ray.
IntegratorState make_initial_state()
{
    IntegratorState state;
    state.accumulated_contrib = float4(0.0f);
    state.C_logT = float4(0.0f);
    state.scalars = float4(0.0f);
    return state;
}

/// Sample point along a ray, representing an ellipsoid boundary crossing.
///
/// When a ray crosses an ellipsoid boundary:
/// - Entry (hit_type=1): delta_contrib.x = +density
/// - Exit (hit_type=0): delta_contrib.x = -density
/// The RGB components are density-weighted: delta_contrib.yzw = sign * density * color
struct SamplePoint : IDifferentiable
{
    float t;              // Ray parameter at this sample
    float4 delta_contrib; // (signed_density, r*signed_density, g*signed_density, b*signed_density)
}

// =============================================================================
// Utility Functions
// =============================================================================

bool AlmostEqualRelative(float A, float B,
                         float maxRelDiff = 1.2e-7f)
{
    return abs(A - B) <= max(abs(A), abs(B)) * maxRelDiff;
}

/// Convert state to dual representation (for backward pass).
IntegratorState to_dual(in IntegratorState state, in SamplePoint sample)
{
    IntegratorState dual_state = state;
    return dual_state;
}

/// Convert state from dual representation.
IntegratorState from_dual(in IntegratorState state, in SamplePoint sample)
{
    IntegratorState dual_state = state;
    return dual_state;
}

// =============================================================================
// Volume Integration
// =============================================================================

/// Inverse update for backward pass - computes previous state from current state.
IntegratorState inverse_update_dual(
    in IntegratorState new_state,
    in SamplePoint new_sample,
    in SamplePoint sample,
    in float t_min,
    in float t_max)
{
    const float t = sample.t;
    const float dt = max(new_state.t - t, 0.f);

    IntegratorState state = new_state;  // Copy all fields first
    state.accumulated_contrib = new_state.accumulated_contrib - new_sample.delta_contrib;

    state.t = t;

    float4 contrib = state.accumulated_contrib;

    let avg = contrib;
    float area = max(avg.x * dt, 0.f);
    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    state.logT = max(new_state.logT - area, 0.f);
    const float weight = clip((1 - safe_exp(-area)) * safe_exp(-state.logT), 0.f, 1.f);

    state.C = new_state.C - weight * rgb_norm;
    float alpha = -safe_expm1(-area);

    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD)
    {
        segment_depth_val = alpha * t + (1 - alpha) * state.t;
    }
    else
    {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area)) - (t + t_min) * safe_exp(-area) + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.depth_accum = state.depth_accum - safe_exp(-state.logT) * segment_depth;

    return state;
}

/// Update volume state with a new sample point (ellipsoid boundary crossing).
///
/// This is the core volume rendering integration step:
/// 1. Accumulate the delta contribution from the boundary
/// 2. Compute the opacity contribution from the segment
/// 3. Update accumulated color weighted by transmittance
[Differentiable]
IntegratorState update(
    in IntegratorState state,
    in SamplePoint sample,
    no_diff in float t_min,
    no_diff in float t_max,
    no_diff in float max_prim_size)
{
    const float t = sample.t;
    const float dt = max(t - state.t, 0.f);

    IntegratorState new_state = state;  // Copy all fields first
    new_state.accumulated_contrib = state.accumulated_contrib + sample.delta_contrib;
    new_state.t = t;

    float4 contrib = state.accumulated_contrib;

    let avg = contrib;
    let area = max(avg.x * dt, 0.f);

    let rgb_norm = safe_div(float3(avg.y, avg.z, avg.w), avg.x);

    new_state.logT = max(area + state.logT, 0.f);
    float alpha = -safe_expm1(-area);
    const float weight = clip(alpha * safe_exp(-state.logT), 0.f, 1.f);
    new_state.C = state.C + weight * rgb_norm;

    const float AREA_THRESHOLD = 1e-6f;
    float segment_depth_val;
    if (avg.x < AREA_THRESHOLD)
    {
        segment_depth_val = alpha * t + (1 - alpha) * state.t;
    }
    else
    {
        segment_depth_val = safe_div(1.f, avg.x) * (-safe_expm1(-area)) - (t + t_min) * safe_exp(-area) + (state.t + t_min);
    }

    const float segment_depth = max(segment_depth_val, 0.f);
    new_state.depth_accum = state.depth_accum + safe_exp(-state.logT) * segment_depth;

    return new_state;
}

// =============================================================================
// Output Extraction
// =============================================================================

/// Final rendered output for a ray.
struct RenderOutput : IDifferentiable
{
    float3 C;    // Accumulated color
    float alpha; // Opacity (1 - transmittance)
    float depth; // Expected depth
};

/// Extract final color and depth from volume state.
[Differentiable]
RenderOutput extract_color(in IntegratorState state, in float tmin)
{
    return {
        state.C,
        -safe_expm1(-state.logT),  // alpha = 1 - exp(-logT)
        state.depth_accum
    };
}
