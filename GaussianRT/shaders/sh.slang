// Modern Splinetracer - Spherical Harmonics Evaluation
// Efficient SH evaluation for color computation
// Apache License 2.0

module sh;

import safe_math;

// Structure for accessing SH features from buffer
public struct SHFeatures {
    uint prim_index;
    uint degree;
    RWStructuredBuffer<float> features;
};

// Color features structure
public struct Features {
    float3 f0;  // Base color (DC term)
};

// Get SH coefficient for a primitive
public float3 get_sh(SHFeatures sh, uint index) {
    uint base = sh.prim_index * (sh.degree + 1) * (sh.degree + 1) * 3;
    uint offset = base + index * 3;
    return float3(
        sh.features[offset + 0],
        sh.features[offset + 1],
        sh.features[offset + 2]
    );
}

// Evaluate SH for view-independent color (degree 0 only)
[Differentiable]
public float3 eval_sh_col0(float3 dir, Features feat) {
    // SH band 0: constant term
    static const float SH_C0 = 0.28209479177387814f;
    return feat.f0 * SH_C0 + 0.5f;
}

// Evaluate SH up to degree 1
[Differentiable]
public float3 eval_sh_col1(float3 dir, Features feat, float3 f1, float3 f2, float3 f3) {
    static const float SH_C0 = 0.28209479177387814f;
    static const float SH_C1 = 0.4886025119029199f;

    float3 result = feat.f0 * SH_C0;
    result += SH_C1 * (-dir.y * f1 + dir.z * f2 - dir.x * f3);
    return result + 0.5f;
}

// Evaluate SH up to degree 2
[Differentiable]
public float3 eval_sh_col2(float3 dir, Features feat,
                           float3 f1, float3 f2, float3 f3,
                           float3 f4, float3 f5, float3 f6, float3 f7, float3 f8) {
    static const float SH_C0 = 0.28209479177387814f;
    static const float SH_C1 = 0.4886025119029199f;
    static const float SH_C2_0 = 1.0925484305920792f;
    static const float SH_C2_1 = -1.0925484305920792f;
    static const float SH_C2_2 = 0.31539156525252005f;
    static const float SH_C2_3 = -1.0925484305920792f;
    static const float SH_C2_4 = 0.5462742152960396f;

    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float3 result = feat.f0 * SH_C0;
    result += SH_C1 * (-y * f1 + z * f2 - x * f3);
    result += SH_C2_0 * xy * f4;
    result += SH_C2_1 * yz * f5;
    result += SH_C2_2 * (2.0f * zz - xx - yy) * f6;
    result += SH_C2_3 * xz * f7;
    result += SH_C2_4 * (xx - yy) * f8;

    return result + 0.5f;
}

// Full SH evaluation dispatcher
[Differentiable]
public float3 eval_sh(float3 dir, uint degree, SHFeatures sh) {
    Features feat;
    feat.f0 = get_sh(sh, 0);

    if (degree == 0) {
        return eval_sh_col0(dir, feat);
    }

    float3 f1 = get_sh(sh, 1);
    float3 f2 = get_sh(sh, 2);
    float3 f3 = get_sh(sh, 3);

    if (degree == 1) {
        return eval_sh_col1(dir, feat, f1, f2, f3);
    }

    float3 f4 = get_sh(sh, 4);
    float3 f5 = get_sh(sh, 5);
    float3 f6 = get_sh(sh, 6);
    float3 f7 = get_sh(sh, 7);
    float3 f8 = get_sh(sh, 8);

    return eval_sh_col2(dir, feat, f1, f2, f3, f4, f5, f6, f7, f8);
}
