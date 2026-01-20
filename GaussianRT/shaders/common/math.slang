// GaussianRT - Math utilities
// Safe math operations with gradients

module math;

import types;

//------------------------------------------------------------------------------
// Safe Math Operations
//------------------------------------------------------------------------------

static const float TINY_VAL = 1.0754944e-20f;
static const float MIN_VAL = -1e10f;
static const float MAX_VAL = 1e10f;

// Safe division with custom backward
[BackwardDerivative(bwd_safeDivide)]
float safeDivide(float a, float b) {
    if (abs(b) < TINY_VAL) {
        return clamp(a / TINY_VAL, MIN_VAL, MAX_VAL);
    }
    return clamp(a / b, MIN_VAL, MAX_VAL);
}

void bwd_safeDivide(inout DifferentialPair<float> a,
                    inout DifferentialPair<float> b,
                    float dOut) {
    if (abs(b.p) < TINY_VAL) {
        a = diffPair(a.p, clamp(dOut / TINY_VAL, MIN_VAL, MAX_VAL));
        b = diffPair(b.p, clamp(-a.p / TINY_VAL * dOut, MIN_VAL, MAX_VAL));
    } else {
        a = diffPair(a.p, clamp(dOut / b.p, MIN_VAL, MAX_VAL));
        float b2 = b.p * b.p;
        if (b2 < TINY_VAL) {
            b = diffPair(b.p, clamp(-a.p / TINY_VAL * dOut, MIN_VAL, MAX_VAL));
        } else {
            b = diffPair(b.p, clamp(-a.p / b2 * dOut, MIN_VAL, MAX_VAL));
        }
    }
}

// Safe exponential
[Differentiable]
float safeExp(float x) {
    return exp(clamp(x, -80.0f, 80.0f));
}

// Safe logarithm
[Differentiable]
float safeLog(float x) {
    return log(max(x, TINY_VAL));
}

// Safe square root
[Differentiable]
float safeSqrt(float x) {
    return sqrt(max(x, 0.0f));
}

//------------------------------------------------------------------------------
// Vector Operations
//------------------------------------------------------------------------------

// Dot product (differentiable)
[Differentiable]
float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product (differentiable)
[Differentiable]
float3 cross3(float3 a, float3 b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Normalize vector (differentiable)
[Differentiable]
float3 normalize3(float3 v) {
    float len = safeSqrt(dot3(v, v));
    return v / max(len, TINY_VAL);
}

// Length of vector
[Differentiable]
float length3(float3 v) {
    return safeSqrt(dot3(v, v));
}

//------------------------------------------------------------------------------
// Matrix Operations
//------------------------------------------------------------------------------

// Matrix-vector multiplication
[Differentiable]
float3 mulMat3Vec(float3x3 m, float3 v) {
    return float3(
        m[0].x * v.x + m[0].y * v.y + m[0].z * v.z,
        m[1].x * v.x + m[1].y * v.y + m[1].z * v.z,
        m[2].x * v.x + m[2].y * v.y + m[2].z * v.z
    );
}

// Transpose matrix
[Differentiable]
float3x3 transpose3x3(float3x3 m) {
    float3x3 result;
    result[0] = float3(m[0].x, m[1].x, m[2].x);
    result[1] = float3(m[0].y, m[1].y, m[2].y);
    result[2] = float3(m[0].z, m[1].z, m[2].z);
    return result;
}

//------------------------------------------------------------------------------
// Quaternion Operations
//------------------------------------------------------------------------------

// Quaternion multiplication
[Differentiable]
float4 quatMul(float4 a, float4 b) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// Quaternion conjugate
[Differentiable]
float4 quatConj(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

// Normalize quaternion
[Differentiable]
float4 quatNormalize(float4 q) {
    float len = safeSqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return q / max(len, TINY_VAL);
}

// Rotate vector by quaternion
[Differentiable]
float3 quatRotate(float4 q, float3 v) {
    float3 qv = float3(q.x, q.y, q.z);
    float3 uv = cross3(qv, v);
    float3 uuv = cross3(qv, uv);
    return v + 2.0f * (q.w * uv + uuv);
}

// Inverse rotate vector by quaternion
[Differentiable]
float3 quatRotateInverse(float4 q, float3 v) {
    return quatRotate(quatConj(q), v);
}
