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
// SH Feature Storage
// =============================================================================

/// SH feature storage for up to degree 3 (16 coefficients, differentiable).
struct Features : IDifferentiable {
    float3 c[16];
    no_diff uint degree;  // SH degree (0-3), excluded from differentiation
};

/// Evaluate SH color from Features struct (differentiable).
[Differentiable]
float3 eval_color(float3 dir, Features feat) {
    float3 color = SH_C0 * feat.c[0] + 0.5f;

    if (feat.degree > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        color += -SH_C1 * y * feat.c[1] + SH_C1 * z * feat.c[2] - SH_C1 * x * feat.c[3];

        if (feat.degree > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            color += SH_C2[0] * xy * feat.c[4] + SH_C2[1] * yz * feat.c[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * feat.c[6] +
                     SH_C2[3] * xz * feat.c[7] + SH_C2[4] * (xx - yy) * feat.c[8];

            if (feat.degree > 2) {
                color += SH_C3[0] * y * (3.0f * xx - yy) * feat.c[9] +
                         SH_C3[1] * x * y * z * feat.c[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * feat.c[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * feat.c[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * feat.c[13] +
                         SH_C3[5] * z * (xx - yy) * feat.c[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * feat.c[15];
            }
        }
    }
    // Clip to non-negative: SH evaluation can produce negative values
    return max(color, 0.0f);
}
