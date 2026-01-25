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

static const float TINY_VAL = 1.0754944e-20;
static const float MIN_VAL = -1e+20;
static const float MAX_VAL = 1e+20;

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

void bw_safe_div(inout DifferentialPair<float> a, inout DifferentialPair<float> b, float.Differential R) {
    if (abs(b.p) < TINY_VAL) {
        a = DifferentialPair<float>(a.p, clip(R / TINY_VAL, MIN_VAL, MAX_VAL));
    } else {
        a = DifferentialPair<float>(a.p, clip(R / b.p, MIN_VAL, MAX_VAL));
    }
    let b2 = b.p * b.p;
    if (b2 < TINY_VAL) {
        b = DifferentialPair<float>(b.p, clip(-a.p / TINY_VAL * R, MIN_VAL, MAX_VAL));
    } else {
        b = DifferentialPair<float>(b.p, clip(-a.p / b2 * R, MIN_VAL, MAX_VAL));
    }
}

[BackwardDerivative(bw_safe_div)]
float safe_div(in float a, in float b) {
    if (abs(b) < TINY_VAL) {
        return clip(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return clip(a / b, MIN_VAL, MAX_VAL);
    }
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

[Differentiable]
float safe_exp(float v) {
    return exp(clip(v, MIN_VAL, log(MAX_VAL)));
}

void bw_log1p(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, safe_div(1, 1 + v.p) * R);
}

[BackwardDerivative(bw_log1p)]
float _log1p(float v) {
    __intrinsic_asm "log1p(($0))";
}

void bw_log(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, R / clip(v.p, MIN_VAL, MAX_VAL));
}

[BackwardDerivative(bw_log)]
float safe_log(float v) {
    return log(clip(v, MIN_VAL, MAX_VAL));
}

void bw_expm1(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, exp(v.p) * R);
}

[BackwardDerivative(bw_expm1)]
float safe_expm1(float v) {
    __intrinsic_asm "expm1($0)";
}

void bw_log1mexp(inout DifferentialPair<float> v, float.Differential R) {
    v = DifferentialPair<float>(v.p, safe_div(R, safe_exp(v.p) - 1));
}

[BackwardDerivative(bw_log1mexp)]
float log1mexp(float x) {
    if (x > 0.30102999566) {
        return _log1p(max(-exp(-x), -1 + 1e-5));
    } else {
        return log(max(-safe_expm1(-x), 1e-20));
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
