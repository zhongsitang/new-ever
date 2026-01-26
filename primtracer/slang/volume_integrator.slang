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

import safe_math;

// =============================================================================
// Core Data Structures
// =============================================================================

/// Volume rendering state per ray (48 bytes, 16-byte aligned)
///
/// Memory layout must match C++ IntegratorState in types.h
/// Using float4 for C to avoid float3 alignment issues between C++ and Slang
struct IntegratorState : IDifferentiable
{
    float4 contrib;  // density, r*d, g*d, b*d
    float4 C;                    // accumulated color RGB (w unused, for alignment)
    float logT;                  // log(T) = -τ_total, always ≤ 0
    float depth_num;             // depth numerator (divide by alpha for expected depth)
    float t;                     // current t
    float _pad;                  // padding to 48 bytes
};

/// Sample point along a ray, representing an ellipsoid boundary crossing.
///
/// When a ray crosses an ellipsoid boundary:
/// - Entry (hit_type=1): contrib.x = +density
/// - Exit (hit_type=0): contrib.x = -density
/// The RGB components are density-weighted: contrib.yzw = sign * density * color
struct SamplePoint : IDifferentiable
{
    float t;        // Ray parameter at this sample
    float4 contrib; // (signed_density, r*signed_density, g*signed_density, b*signed_density)
}

// =============================================================================
// Volume Integration
// =============================================================================
//
// Volume rendering equation (discrete form):
//   C = Σ T_i × α_i × c_i
//   where T_i = exp(-Σ τ_j) is transmittance, α_i = 1-exp(-τ_i) is opacity
//
// Key insight: We process hits in sorted order. Between consecutive hits,
// the density field is constant (sum of overlapping ellipsoids' densities).
// So we integrate each segment with constant density, then update density
// at the boundary.
//
// =============================================================================

/// Process one volume integration step at a boundary crossing.
///
/// Volume rendering for segment [prev_t, curr_t] with constant density σ:
///   τ = σ × (curr_t - prev_t)           // optical depth
///   T_new = T_old × exp(-τ)             // transmittance update
///   C_new = C_old + T_old × α × color   // color accumulation (α = 1-exp(-τ))
///
/// After processing, update contrib for next segment.
[Differentiable]
IntegratorState integrate_step(
    in IntegratorState state,
    in SamplePoint sample)
{
    IntegratorState new_state = state;

    // --- Step 1: Integrate segment [state.t, sample.t] with current density ---
    const float dt = max(sample.t - state.t, 0.f);
    const float sigma = state.contrib.x;
    const float tau = max(sigma * dt, 0.f);  // optical depth τ = σ × Δt
    const float3 color = safe_div(state.contrib.yzw, sigma);

    // Transmittance: T_new = T_old × exp(-τ), i.e. log(T) -= τ
    new_state.logT = state.logT - tau;

    // Color: C += T × α × color, where α = 1 - exp(-τ)
    const float alpha = -safe_expm1(-tau);
    const float T = safe_exp(state.logT);  // transmittance before this segment
    const float weight = clip(alpha * T, 0.f, 1.f);
    new_state.C.xyz = state.C.xyz + weight * color;

    // Depth numerator: Δnum = T × [t₀ × α + (1 - e^{-τ}(1+τ))/σ]
    // Numerically stable form: 1 - e^{-τ}(1+τ) = α - τ·e^{-τ}
    // This avoids catastrophic cancellation in "1 - (value close to 1)"
    const float depth_term = safe_div(alpha - tau * safe_exp(-tau), sigma);
    new_state.depth_num = state.depth_num + T * (state.t * alpha + depth_term);

    // --- Step 2: Update density field for next segment ---
    new_state.contrib = state.contrib + sample.contrib;
    new_state.t = sample.t;

    return new_state;
}

/// Reconstruct previous state from current state (inverse of integrate_step).
/// Used in backward pass to recover intermediate states without storing them.
///
/// Given: state after processing sample (new_state)
/// Returns: state before processing sample
IntegratorState invert_step(
    in IntegratorState new_state,
    in SamplePoint new_sample,
    in SamplePoint prev_sample)
{
    IntegratorState prev_state = new_state;

    // --- Undo Step 2: Recover previous contrib and t ---
    prev_state.contrib = new_state.contrib - new_sample.contrib;
    prev_state.t = prev_sample.t;

    // --- Undo Step 1: Subtract segment's contribution ---
    // Segment parameters (same computation as forward)
    const float dt = max(new_sample.t - prev_sample.t, 0.f);
    const float sigma = prev_state.contrib.x;
    const float tau = max(sigma * dt, 0.f);  // optical depth τ = σ × Δt
    const float3 color = safe_div(prev_state.contrib.yzw, sigma);

    // Undo transmittance accumulation: logT_prev = logT_new + τ
    prev_state.logT = min(new_state.logT + tau, 0.f);

    // Undo color accumulation: C_prev = C_new - weight × color
    const float alpha = -safe_expm1(-tau);  // α = 1 - exp(-τ)
    const float T = safe_exp(prev_state.logT);
    const float weight = clip(alpha * T, 0.f, 1.f);
    prev_state.C.xyz = new_state.C.xyz - weight * color;

    // Undo depth accumulation (same formula as forward)
    // Numerically stable: 1 - e^{-τ}(1+τ) = α - τ·e^{-τ}
    const float depth_term = safe_div(alpha - tau * safe_exp(-tau), sigma);
    prev_state.depth_num = new_state.depth_num - T * (prev_sample.t * alpha + depth_term);

    return prev_state;
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

/// Extract final render output from volume integration state.
///
/// Computes:
///   - alpha = 1 - T (opacity from transmittance)
///   - depth = E[t] = depth_num / alpha (expected depth, or 0 if transparent)
[Differentiable]
RenderOutput extract_output(in IntegratorState state)
{
    const float ALPHA_THRESHOLD = 1e-6f;

    float alpha = -safe_expm1(state.logT);  // 1 - exp(logT)
    float depth = (alpha < ALPHA_THRESHOLD) ? 0.0f : safe_div(state.depth_num, alpha);

    return { state.C.xyz, alpha, depth };
}
