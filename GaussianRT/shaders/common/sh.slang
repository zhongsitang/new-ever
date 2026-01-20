// GaussianRT - Spherical Harmonics evaluation
// Supports up to degree 3 (16 coefficients per channel)

module sh;

import types;

//------------------------------------------------------------------------------
// SH Feature Structure
//------------------------------------------------------------------------------

struct SHFeatures : IDifferentiable {
    float3 f0;   // DC
    float3 f1, f2, f3;  // Degree 1
    float3 f4, f5, f6, f7, f8;  // Degree 2
    float3 f9, f10, f11, f12, f13, f14, f15;  // Degree 3
};

// Load SH features from buffer
SHFeatures loadSHFeatures(StructuredBuffer<float> features, uint primIdx, uint featureSize) {
    SHFeatures sh;

    uint base = primIdx * featureSize;

    // Always load DC
    sh.f0 = float3(features[base], features[base + 1], features[base + 2]);

    if (featureSize >= 12) {  // Degree 1
        sh.f1 = float3(features[base + 3], features[base + 4], features[base + 5]);
        sh.f2 = float3(features[base + 6], features[base + 7], features[base + 8]);
        sh.f3 = float3(features[base + 9], features[base + 10], features[base + 11]);
    } else {
        sh.f1 = sh.f2 = sh.f3 = float3(0.0f);
    }

    if (featureSize >= 27) {  // Degree 2
        sh.f4 = float3(features[base + 12], features[base + 13], features[base + 14]);
        sh.f5 = float3(features[base + 15], features[base + 16], features[base + 17]);
        sh.f6 = float3(features[base + 18], features[base + 19], features[base + 20]);
        sh.f7 = float3(features[base + 21], features[base + 22], features[base + 23]);
        sh.f8 = float3(features[base + 24], features[base + 25], features[base + 26]);
    } else {
        sh.f4 = sh.f5 = sh.f6 = sh.f7 = sh.f8 = float3(0.0f);
    }

    if (featureSize >= 48) {  // Degree 3
        sh.f9 = float3(features[base + 27], features[base + 28], features[base + 29]);
        sh.f10 = float3(features[base + 30], features[base + 31], features[base + 32]);
        sh.f11 = float3(features[base + 33], features[base + 34], features[base + 35]);
        sh.f12 = float3(features[base + 36], features[base + 37], features[base + 38]);
        sh.f13 = float3(features[base + 39], features[base + 40], features[base + 41]);
        sh.f14 = float3(features[base + 42], features[base + 43], features[base + 44]);
        sh.f15 = float3(features[base + 45], features[base + 46], features[base + 47]);
    } else {
        sh.f9 = sh.f10 = sh.f11 = sh.f12 = sh.f13 = sh.f14 = sh.f15 = float3(0.0f);
    }

    return sh;
}

//------------------------------------------------------------------------------
// SH Evaluation Functions
//------------------------------------------------------------------------------

// Degree 0 (DC term)
[Differentiable]
float3 evalSHDegree0(SHFeatures sh) {
    return SH_C0 * sh.f0 + 0.5f;
}

// Degree 1
[Differentiable]
float3 evalSHDegree1(float3 dir, SHFeatures sh) {
    float x = dir.x, y = dir.y, z = dir.z;

    return -SH_C1 * y * sh.f1 +
            SH_C1 * z * sh.f2 -
            SH_C1 * x * sh.f3;
}

// Degree 2
[Differentiable]
float3 evalSHDegree2(float3 dir, SHFeatures sh) {
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    return SH_C2_0 * xy * sh.f4 +
           SH_C2_1 * yz * sh.f5 +
           SH_C2_2 * (2.0f * zz - xx - yy) * sh.f6 +
           SH_C2_3 * xz * sh.f7 +
           SH_C2_4 * (xx - yy) * sh.f8;
}

// Degree 3
[Differentiable]
float3 evalSHDegree3(float3 dir, SHFeatures sh) {
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    return SH_C3_0 * y * (3.0f * xx - yy) * sh.f9 +
           SH_C3_1 * xy * z * sh.f10 +
           SH_C3_2 * y * (4.0f * zz - xx - yy) * sh.f11 +
           SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh.f12 +
           SH_C3_4 * x * (4.0f * zz - xx - yy) * sh.f13 +
           SH_C3_5 * z * (xx - yy) * sh.f14 +
           SH_C3_6 * x * (xx - 3.0f * yy) * sh.f15;
}

// Evaluate SH color for given direction and degree
[Differentiable]
float3 evalSHColor(float3 dir, SHFeatures sh, uint shDegree) {
    float3 color = evalSHDegree0(sh);

    if (shDegree >= 1) {
        color += evalSHDegree1(dir, sh);
    }
    if (shDegree >= 2) {
        color += evalSHDegree2(dir, sh);
    }
    if (shDegree >= 3) {
        color += evalSHDegree3(dir, sh);
    }

    // Clamp to valid range
    return clamp(color, float3(0.0f), float3(1.0f));
}

// Simple RGB evaluation (no SH)
[Differentiable]
float3 evalRGBColor(StructuredBuffer<float> features, uint primIdx) {
    uint base = primIdx * 3;
    return float3(
        features[base],
        features[base + 1],
        features[base + 2]
    );
}

// Main color evaluation function
[Differentiable]
float3 evaluateColor(
    float3 viewDir,
    StructuredBuffer<float> features,
    uint primIdx,
    uint featureSize,
    uint shDegree)
{
    if (featureSize == 3) {
        // Direct RGB
        return evalRGBColor(features, primIdx);
    } else {
        // Spherical harmonics
        SHFeatures sh = loadSHFeatures(features, primIdx, featureSize);
        return evalSHColor(viewDir, sh, shDegree);
    }
}
