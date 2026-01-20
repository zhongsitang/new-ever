// Modern Splinetracer - Safe Math Utilities
// Numerically stable math operations for volume rendering
// Apache License 2.0

module safe_math;

static const float EPSILON = 1e-18f;
static const float TINY_VAL = 1.0754944e-20f;

// Safe division to avoid divide-by-zero
[Differentiable]
public float safe_div(float a, float b) {
    return a / max(abs(b), EPSILON) * sign(b + EPSILON);
}

[Differentiable]
public float3 safe_div(float3 a, float b) {
    float safe_b = max(abs(b), EPSILON) * sign(b + EPSILON);
    return a / safe_b;
}

[Differentiable]
public float3 safe_div(float3 a, float3 b) {
    return float3(safe_div(a.x, b.x), safe_div(a.y, b.y), safe_div(a.z, b.z));
}

// Safe square root
[Differentiable]
public float safe_sqrt(float x) {
    return sqrt(max(x, 0.0f));
}

// Safe exponential (clamped to avoid overflow)
[Differentiable]
public float safe_exp(float x) {
    return exp(clamp(x, -80.0f, 80.0f));
}

// Safe expm1: exp(x) - 1, numerically stable for small x
[Differentiable]
public float safe_expm1(float x) {
    if (abs(x) < 1e-5f) {
        return x + 0.5f * x * x;
    }
    return exp(clamp(x, -80.0f, 80.0f)) - 1.0f;
}

// L2 normalize with safety
[Differentiable]
public float3 l2_normalize(float3 v) {
    float len_sq = dot(v, v);
    if (len_sq < EPSILON) {
        return float3(0.0f, 0.0f, 1.0f);
    }
    return v * rsqrt(len_sq);
}

// Clamp value to range
[Differentiable]
public float clip(float x, float lo, float hi) {
    return clamp(x, lo, hi);
}

// Half-precision sign function
public half hsign(half x) {
    return x >= 0.0h ? 1.0h : -1.0h;
}

// Half-precision safe sqrt
public half hsqrt(half x) {
    return sqrt(max(x, (half)0.0));
}

// Tukey power ladder transform for distortion loss
[Differentiable]
public float tukey_power_ladder(float x, float p) {
    if (abs(p) < 1e-6f) {
        return log(max(x, EPSILON));
    }
    return (pow(max(x, EPSILON), p) - 1.0f) / p;
}

// Quaternion rotation of a vector
[Differentiable]
public float3 rotate_vector(float3 v, float4 q) {
    float3 u = float3(q.y, q.z, q.w);
    float s = q.x;
    return 2.0f * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0f * s * cross(u, v);
}

// Convert quaternion to rotation matrix
[Differentiable]
public float3x3 quat_to_matrix(float4 q) {
    float r = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z), 2.0f * (x * z + r * y),
        2.0f * (x * y + r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x),
        2.0f * (x * z - r * y), 2.0f * (y * z + r * x), 1.0f - 2.0f * (x * x + y * y)
    );
}
