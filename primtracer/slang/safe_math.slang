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
// Safe Math Utilities
// =============================================================================

// TINY_VAL: threshold for "effectively zero" in divisions
// When denominator < TINY_VAL, the ratio is undefined and we return 0
// This is semantically correct for volume rendering where color = contrib/sigma
// and when sigma ≈ 0, the contribution should also be ≈ 0
static const float TINY_VAL = 1e-7f;

// EXP_LIMIT: clamp exp() input to avoid overflow (exp(88) ≈ 1.6e38 ≈ FLT_MAX)
static const float EXP_LIMIT = 87.0f;

// =============================================================================
// Clipping
// =============================================================================

[Differentiable]
float clip(float v, float minv, float maxv) {
    return max(min(v, maxv), minv);
}

[Differentiable]
float3 clip(float3 v, float minv, float maxv) {
    return {
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    };
}

[Differentiable]
float4 clip(float4 v, float minv, float maxv) {
    return {
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv),
        max(min(v.w, maxv), minv)
    };
}

// =============================================================================
// Safe Division
// =============================================================================
//
// Semantics: when denominator ≈ 0, the ratio is undefined → return 0
// This is correct for volume rendering where color = contrib/sigma:
// - when sigma ≈ 0, contrib ≈ 0 too, so color contribution is negligible
// - returning 0 instead of a/TINY_VAL avoids gradient explosion in backward pass
//
// Backward: when |b| < TINY_VAL, forward returns 0 (independent of a,b)
// so gradients w.r.t. a and b are both 0

void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
    if (abs(b.p) < TINY_VAL) {
        // Forward returned 0, so no gradient flows
        a = DifferentialPair<float>(a.p, 0.0f);
        b = DifferentialPair<float>(b.p, 0.0f);
    } else {
        // Normal case: d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        a = DifferentialPair<float>(a.p, R / b.p);
        b = DifferentialPair<float>(b.p, -a.p / (b.p * b.p) * R);
    }
}

[BackwardDerivative(bw_safe_div)]
float safe_div(in float a, in float b) {
    if (abs(b) < TINY_VAL) {
        return 0.0f;  // Ratio undefined → return 0
    }
    return a / b;  // No clip: let caller handle extreme values if needed
}

[Differentiable]
float3 safe_div(in float3 a, in float b) {
    return {safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b)};
}

[Differentiable]
float4 safe_div(in float4 a, in float b) {
    return {safe_div(a.x, b), safe_div(a.y, b), safe_div(a.z, b), safe_div(a.w, b)};
}

[Differentiable]
float3 safe_div(in float3 a, in float3 b) {
    return {safe_div(a.x, b.x), safe_div(a.y, b.y), safe_div(a.z, b.z)};
}

// =============================================================================
// Safe Square Root
// =============================================================================

[Differentiable]
float safe_sqrt(in float a) {
    if (a < TINY_VAL) {
        return 0.0;
    } else {
        return sqrt(a);
    }
}

// =============================================================================
// Normalization
// =============================================================================

[Differentiable]
float4 l2_normalize(float4 x) {
    return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
}

[Differentiable]
float3 l2_normalize(float3 x) {
    return safe_div(x, sqrt(max(dot(x, x), TINY_VAL)));
}

// =============================================================================
// Safe Exponential/Logarithm
// =============================================================================

// safe_exp: clamp INPUT only to avoid overflow, don't clamp output
// This preserves gradient flow while preventing inf/nan
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
    // d(log(v))/dv = 1/v, protect against v ≈ 0
    if (abs(v.p) < TINY_VAL) {
        v = DifferentialPair<float>(v.p, 0.0f);
    } else {
        v = DifferentialPair<float>(v.p, R / v.p);
    }
}

[BackwardDerivative(bw_log)]
float safe_log(float v) {
    // Protect against log(≤0), but don't clip output
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
// Uses different formulas for numerical stability:
// - x > log(2): use log1p(-exp(-x)) to avoid 1 - (value close to 1)
// - x ≤ log(2): use log(-expm1(-x)) since expm1 is accurate for small args
[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
    if (x > 0.6931471805599453) {  // log(2)
        // For large x: exp(-x) is small, so 1-exp(-x) ≈ 1, use log1p for accuracy
        return _log1p(-exp(-x));
    } else {
        // For small x: use expm1 which is accurate, then take log
        // expm1(-x) = exp(-x) - 1, so -expm1(-x) = 1 - exp(-x)
        float arg = -safe_expm1(-x);
        return log(max(arg, TINY_VAL));
    }
}

// =============================================================================
// Quaternion Rotation
// =============================================================================

/// Rotate vector by conjugate quaternion (q = [w, x, y, z]).
[Differentiable]
float3 rotate_vector(float3 v, float4 q)
{
    let t = 2 * cross(-q.yzw, v);
    return v + q.x * t + cross(-q.yzw, t);
}

// =============================================================================
// Integer Utilities
// =============================================================================

uint mod(uint a, uint b) {
    __intrinsic_asm "$0 % $1";
}
