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
// Math Utilities for Differentiable Volume Rendering
// =============================================================================

// =============================================================================
// Constants
// =============================================================================

// Threshold for "effectively zero" in divisions
// When denominator < TINY_VAL, ratio is undefined → return 0
static const float TINY_VAL = 1e-7f;

// Clamp exp() input to avoid overflow (exp(88) ≈ 1.6e38 ≈ FLT_MAX)
static const float EXP_LIMIT = 87.0f;

// =============================================================================
// Clipping & Safe Division
// =============================================================================

[Differentiable]
float clip(float v, float minv, float maxv) {
    return max(min(v, maxv), minv);
}

[Differentiable]
float3 clip(float3 v, float minv, float maxv) {
    return float3(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    );
}

[Differentiable]
float4 clip(float4 v, float minv, float maxv) {
    return float4(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv),
        max(min(v.w, maxv), minv)
    );
}

// Safe division with correct gradients
// When |b| < TINY_VAL: return 0, gradient = 0 (avoid explosion)
// Otherwise: return a/b with standard gradients
void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
    if (abs(b.p) < TINY_VAL) {
        // Zero output, zero gradient
        a = diffPair(a.p, 0.0f);
        b = diffPair(b.p, 0.0f);
    } else {
        // Standard: d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        a = diffPair(a.p, R / b.p);
        b = diffPair(b.p, -a.p * R / (b.p * b.p));
    }
}

[BackwardDerivative(bw_safe_div)]
float safe_div(float a, float b) {
    return abs(b) < TINY_VAL ? 0.0f : a / b;
}

[Differentiable]
float3 safe_div(in float3 a, in float b) {
    return float3(safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b));
}

[Differentiable]
float4 safe_div(in float4 a, in float b) {
    return float4(safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b), safe_div(a.w, b));
}

[Differentiable]
float3 safe_div(in float3 a, in float3 b) {
    return float3(safe_div(a.x, b.x), safe_div(a.y, b.y), safe_div(a.z, b.z));
}

[Differentiable]
float safe_sqrt(in float a) {
    return a < TINY_VAL ? 0.0f : sqrt(a);
}

// =============================================================================
// Safe Exponential & Logarithm
// =============================================================================

[Differentiable]
float safe_exp(float v) {
    return exp(clamp(v, -EXP_LIMIT, EXP_LIMIT));
}

void bw_log1p(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, safe_div(1, 1 + v.p) * R);
}

[BackwardDerivative(bw_log1p)]
float log1p(float v) {
    __intrinsic_asm "log1p(($0))";
}

void bw_safe_log(inout DifferentialPair<float> v, float.Differential R) {
    if (abs(v.p) < TINY_VAL) {
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        v = DifferentialPair<float>(v.p, R / v.p);
    }
}

[BackwardDerivative(bw_safe_log)]
float safe_log(float v) {
    return log(max(v, TINY_VAL));
}

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float expm1(float v) {
    __intrinsic_asm "expm1($0)";
}

void bw_log1mexp(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, safe_div(R, expm1(v.p)));
}

// log1mexp(x) = log(1 - exp(-x)) for x > 0
// Uses different formulas for numerical stability
[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
    static const float LOG2 = 0.6931471805599453f;
    if (x > LOG2) {
        return log1p(-exp(-x));
    } else {
        float arg = -expm1(-x);
        return log(max(arg, TINY_VAL));
    }
}

// =============================================================================
// Quaternion Operations
// =============================================================================

[Differentiable]
float4 quat_conjugate(float4 q) {
    return float4(q.x, -q.yzw);
}

/// Rotate vector by quaternion q (q = [w, x, y, z]).
/// v' = q * v * conjugate(q). Assumes q is (near) unit.
[Differentiable]
float3 rotate_vector(float3 v, float4 q) {
    float3 u = q.yzw;
    float w = q.x;
    float3 t = 2.0f * cross(u, v);
    return v + w * t + cross(u, t);
}

// =============================================================================
// Ray-Sphere/Ellipsoid Intersection
// =============================================================================

/// Intersect ray with axis-aligned ellipsoid at origin.
/// Ellipsoid: (x/radii.x)² + (y/radii.y)² + (z/radii.z)² = 1
///
/// Returns (t0, t1) unsorted, or (-1, -1) if no hit.
/// Uses closest-point form for discriminant to handle grazing rays.
[Differentiable]
float2 isect_ellipsoid(float3 origin, float3 dir, float3 radii)
{
    const float2 NO_HIT = float2(-1.0f, -1.0f);

    // Reject degenerate ellipsoids
    if (!all(abs(radii) > TINY_VAL)) return NO_HIT;

    // Transform to unit sphere space
    float3 o = origin / radii;
    float3 d = dir / radii;

    // Quadratic: |o + t*d|² = 1  →  a*t² - 2*b*t + c = 0
    float a = dot(d, d);
    if (a < TINY_VAL) return NO_HIT;

    float b = -dot(o, d);  // negated half-b
    float c = dot(o, o) - 1.0f;

    // Discriminant (closest-point form for stability)
    float k = b / a;
    float3 p = o + k * d;
    float disc = a * (1.0f - dot(p, p));
    if (disc < 0.0f) return NO_HIT;

    float s = sqrt(disc);

    // Citardauq formula: q = b + sign(b)*√disc
    float q = b + copysign(s, b);

    // Compute roots with branch for numerical safety
    if (abs(q) < TINY_VAL) {
        float t = b / a;
        return float2(t, t);
    }
    return float2(c / q, q / a);
}

/// Intersect a ray with an arbitrarily transformed ellipsoid.
/// Returns (t_near, t_far) sorted along the ray, or (-1, -1) if no hit.
///
/// Parameters:
///   - ray_origin, ray_dir: ray in world space (ray_dir should be unit length)
///   - center: ellipsoid center in world space
///   - scale: ellipsoid radii along its local axes
///   - quat: rotation quaternion (local → world)
[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 ray_origin,
    in float3 ray_dir,
    in float3 center,
    in float3 scale,
    in float4 quat)
{
    // Normalize quaternion
    float4 q = length(quat) < TINY_VAL ? float4(1, 0, 0, 0) : normalize(quat);
    float4 qinv = quat_conjugate(q);

    // Transform ray into ellipsoid local space
    float3 ro = rotate_vector(ray_origin - center, qinv);
    float3 rd = rotate_vector(ray_dir, qinv);

    float2 t = isect_ellipsoid(ro, rd, scale);

    // Sort entry/exit
    return float2(min(t.x, t.y), max(t.x, t.y));
}
