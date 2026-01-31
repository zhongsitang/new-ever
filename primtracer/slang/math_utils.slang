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

// Safe division: when |b| < TINY_VAL, return 0 (avoids gradient explosion)
void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
    if (abs(b.p) < TINY_VAL) {
        a = DifferentialPair<float>(a.p, 0.0f);
        b = DifferentialPair<float>(b.p, 0.0f);
    } else {
        a = DifferentialPair<float>(a.p, R / b.p);
        b = DifferentialPair<float>(b.p, -a.p / (b.p * b.p) * R);
    }
}

[BackwardDerivative(bw_safe_div)]
float safe_div(in float a, in float b) {
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

// =============================================================================
// Safe Square Root & Normalization
// =============================================================================

[Differentiable]
float safe_sqrt(in float a) {
    return a < TINY_VAL ? 0.0f : sqrt(a);
}

[Differentiable]
float3 l2_normalize(float3 x) {
    return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
}

[Differentiable]
float4 l2_normalize(float4 x) {
    return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
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
float _log1p(float v) {
    __intrinsic_asm "log1p(($0))";
}

void bw_log(inout DifferentialPair<float> v, float.Differential R) {
    if (abs(v.p) < TINY_VAL) {
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        v = DifferentialPair<float>(v.p, R / v.p);
    }
}

[BackwardDerivative(bw_log)]
float safe_log(float v) {
    return log(max(v, TINY_VAL));
}

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float safe_expm1(float v) {
    __intrinsic_asm "expm1($0)";
}

void bw_log1mexp(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, safe_div(R, safe_expm1(v.p)));
}

// log1mexp(x) = log(1 - exp(-x)) for x > 0
// Uses different formulas for numerical stability
[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
    static const float LOG2 = 0.6931471805599453f;
    if (x > LOG2) {
        return _log1p(-exp(-x));
    } else {
        float arg = -safe_expm1(-x);
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
// Ray-Ellipsoid Intersection
// =============================================================================

/// Intersect a ray with an arbitrarily transformed ellipsoid.
/// Returns (t_entry, t_exit) sorted along the ray, or (-1, -1) if no hit.
///
/// Assumptions:
///   - ray_dir is unit length
///   - quat encodes ellipsoid rotation (local → world)
///   - scale gives ellipsoid radii along its local axes
///
/// Method:
///   Transform the ray into the ellipsoid's local space, scale the ellipsoid
///   to a unit sphere, then solve the standard ray–sphere quadratic robustly.
[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 ray_origin,
    in float3 ray_dir, // unit vector
    in float3 center,
    in float3 scale,
    in float4 quat,       // local → world rotation
    in float eps = 1e-6f) // numerical tolerance (float32-safe)
{
    // Normalize quaternion and compute inverse (world → local rotation)
    float4 q = l2_normalize(quat);
    float4 qinv = quat_conjugate(q);

    // Transform ray into ellipsoid local space
    float3 ro = rotate_vector(ray_origin - center, qinv);
    float3 rd = rotate_vector(ray_dir, qinv);

    // Prevent degenerate radii (very small scale causes numeric blow-up)
    float3 s = max(abs(scale), eps);
    float3 inv_s = 1.0f / s;

    // Map ellipsoid to unit sphere
    float3 oc = ro * inv_s;
    float3 d = rd * inv_s;

    // Solve |oc + t*d|^2 = 1  →  a t^2 + 2 b t + c = 0
    float a = dot(d, d);
    float b = dot(oc, d);

    // Degenerate direction after scaling → no reliable intersection
    const float eps_a = 1e-12f;
    if (a < eps_a) return float2(-1.0f, -1.0f);

    // Stable intersection using closest-approach distance (avoids b*b - a*c)
    float t0 = -b / a;
    float3 p = oc + t0 * d;
    float d2 = dot(p, p);

    // Reject if closest point is outside unit sphere (with tolerance)
    if (d2 > 1.0f + eps) return float2(-1.0f, -1.0f);

    float thc = sqrt(max(1.0f - d2, 0.0f));
    float inv_sqrt_a = rsqrt(a);

    float t0_hit = t0 - thc * inv_sqrt_a;
    float t1_hit = t0 + thc * inv_sqrt_a;

    // Sort entry/exit
    return float2(min(t0_hit, t1_hit), max(t0_hit, t1_hit));
}
