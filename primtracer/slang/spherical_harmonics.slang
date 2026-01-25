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
// Spherical Harmonics Evaluation
// =============================================================================

// SH basis coefficients
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f
};
static const float SH_C3[] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f
};

// =============================================================================
// SH Features for StructuredBuffer (forward pass)
// =============================================================================

/// SH feature accessor for StructuredBuffer.
struct SHFeatures {
    uint idx;
    uint sh_degree;
    StructuredBuffer<float> shs;
};

/// Load single SH coefficient from StructuredBuffer.
float3 get_sh(in SHFeatures feats, uint coeff_idx) {
    let num_coeff = (feats.sh_degree + 1) * (feats.sh_degree + 1);
    let offset = feats.idx * num_coeff * 3 + coeff_idx * 3;
    return { feats.shs[offset], feats.shs[offset + 1], feats.shs[offset + 2] };
}

/// Evaluate SH color directly from StructuredBuffer (fused load + eval).
float3 eval_color(float3 dir, SHFeatures feats) {
    float3 color = SH_C0 * get_sh(feats, 0) + 0.5f;

    if (feats.sh_degree > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        color += -SH_C1 * y * get_sh(feats, 1)
               +  SH_C1 * z * get_sh(feats, 2)
               -  SH_C1 * x * get_sh(feats, 3);

        if (feats.sh_degree > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            color += SH_C2[0] * xy * get_sh(feats, 4)
                   + SH_C2[1] * yz * get_sh(feats, 5)
                   + SH_C2[2] * (2.0f * zz - xx - yy) * get_sh(feats, 6)
                   + SH_C2[3] * xz * get_sh(feats, 7)
                   + SH_C2[4] * (xx - yy) * get_sh(feats, 8);

            if (feats.sh_degree > 2) {
                color += SH_C3[0] * y * (3.0f * xx - yy) * get_sh(feats, 9)
                       + SH_C3[1] * x * y * z * get_sh(feats, 10)
                       + SH_C3[2] * y * (4.0f * zz - xx - yy) * get_sh(feats, 11)
                       + SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_sh(feats, 12)
                       + SH_C3[4] * x * (4.0f * zz - xx - yy) * get_sh(feats, 13)
                       + SH_C3[5] * z * (xx - yy) * get_sh(feats, 14)
                       + SH_C3[6] * x * (xx - 3.0f * yy) * get_sh(feats, 15);
            }
        }
    }
    return color;
}

// =============================================================================
// SH Features for TensorView (backward pass with autodiff)
// =============================================================================

/// SH feature storage for up to degree 3 (differentiable).
struct Features : IDifferentiable {
    float3 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
};

/// Evaluate SH color from Features struct (differentiable).
[Differentiable]
float3 eval_color(float3 dir, Features feat, no_diff uint sh_degree) {
    float3 color = SH_C0 * feat.f0 + 0.5f;

    if (sh_degree > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        color += -SH_C1 * y * feat.f1 + SH_C1 * z * feat.f2 - SH_C1 * x * feat.f3;

        if (sh_degree > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            color += SH_C2[0] * xy * feat.f4 + SH_C2[1] * yz * feat.f5 +
                     SH_C2[2] * (2.0f * zz - xx - yy) * feat.f6 +
                     SH_C2[3] * xz * feat.f7 + SH_C2[4] * (xx - yy) * feat.f8;

            if (sh_degree > 2) {
                color += SH_C3[0] * y * (3.0f * xx - yy) * feat.f9 +
                         SH_C3[1] * x * y * z * feat.f10 +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * feat.f11 +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * feat.f12 +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * feat.f13 +
                         SH_C3[5] * z * (xx - yy) * feat.f14 +
                         SH_C3[6] * x * (xx - 3.0f * yy) * feat.f15;
            }
        }
    }
    return color;
}
