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

import safe_math;

// =============================================================================
// Float precision spherical harmonics
// =============================================================================

// Spherical harmonics coefficients
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
  1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f,
  0.5462742152960396f
};
static const float SH_C3[] = {
  -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
  -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f
};

struct SHFeatures {
  int idx;
  int max_deg;
  StructuredBuffer<float> shs;
};

float3 get_sh(in SHFeatures feats, int deg) {
  let num_coeff = (feats.max_deg+1)*(feats.max_deg+1);
  return {
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 0],
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 1],
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 2],
  };
}

struct Features : IDifferentiable {
  float3 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
};

[Differentiable]
float3 eval_sh_col0(float3 dir, Features feat) {
  return SH_C0 * feat.f0 + 0.5f;
}

[Differentiable]
float3 eval_sh_col1(float3 dir, Features feat) {
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  float3 color = -SH_C1 * y * feat.f1 + SH_C1 * z * feat.f2 - SH_C1 * x * feat.f3;
  return color;
}

[Differentiable]
float3 eval_sh_col2(float3 dir, Features feat) {
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, yz = y * z, xz = x * z;
  return SH_C2[0] * xy * feat.f4 + SH_C2[1] * yz * feat.f5 +
         SH_C2[2] * (2.0f * zz - xx - yy) * feat.f6 +
         SH_C2[3] * xz * feat.f7 + SH_C2[4] * (xx - yy) * feat.f8;
}

[Differentiable]
float3 eval_sh_col3(float3 dir, Features feat) {
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, yz = y * z, xz = x * z;
  return SH_C3[0] * y * (3.0f * xx - yy) * feat.f9 +
         SH_C3[1] * xy * z * feat.f10 +
         SH_C3[2] * y * (4.0f * zz - xx - yy) * feat.f11 +
         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * feat.f12 +
         SH_C3[4] * x * (4.0f * zz - xx - yy) * feat.f13 +
         SH_C3[5] * z * (xx - yy) * feat.f14 +
         SH_C3[6] * x * (xx - 3.0f * yy) * feat.f15;
}

[Differentiable]
float3 eval_color(float3 dir, Features feat, no_diff uint sh_degree) {
  float3 color = eval_sh_col0(dir, feat);
  if (sh_degree > 0) {
    color += eval_sh_col1(dir, feat);
    if (sh_degree > 1) {
      color += eval_sh_col2(dir, feat);
      if (sh_degree > 2) {
        color += eval_sh_col3(dir, feat);
      }
    }
  }
  return color;
}

// =============================================================================
// Half precision spherical harmonics
// =============================================================================

static const half SH_C0_HALF = half(0.28209479177387814);
static const half SH_C1_HALF = half(0.4886025119029199);
static const half SH_C2_HALF[] = {
  half(1.0925484305920792), half(-1.0925484305920792), half(0.31539156525252005), half(-1.0925484305920792),
  half(0.5462742152960396)
};
static const half SH_C3_HALF[] = {
  half(-0.5900435899266435), half(2.890611442640554), half(-0.4570457994644658), half(0.3731763325901154),
  half(-0.4570457994644658), half(1.445305721320277), half(-0.5900435899266435)
};

struct HalfSHFeatures {
  int idx;
  int max_deg;
  StructuredBuffer<half> shs;
};

half3 get_sh(in HalfSHFeatures feats, int deg) {
  let num_coeff = (feats.max_deg+1)*(feats.max_deg+1);
  return {
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 0],
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 1],
    feats.shs[feats.idx * num_coeff * 3 + 3 * deg + 2],
  };
}

struct HalfFeatures : IDifferentiable {
  half3 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
};

[Differentiable]
half3 half_eval_sh_col0(half3 dir, HalfFeatures feat) {
  return SH_C0_HALF * feat.f0 + half(0.5);
}

[Differentiable]
half3 half_eval_sh_col1(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half3 color = -SH_C1_HALF * y * feat.f1 + SH_C1_HALF * z * feat.f2 - SH_C1_HALF * x * feat.f3;
  return color;
}

[Differentiable]
half3 half_eval_sh_col2(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half xx = x * x, yy = y * y, zz = z * z;
  half xy = x * y, yz = y * z, xz = x * z;
  return SH_C2_HALF[0] * xy * feat.f4 + SH_C2_HALF[1] * yz * feat.f5 +
         SH_C2_HALF[2] * (half(2.0) * zz - xx - yy) * feat.f6 +
         SH_C2_HALF[3] * xz * feat.f7 + SH_C2_HALF[4] * (xx - yy) * feat.f8;
}

[Differentiable]
half3 half_eval_sh_col3(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half xx = x * x, yy = y * y, zz = z * z;
  half xy = x * y, yz = y * z, xz = x * z;
  half h2 = half(2.0), h3 = half(3.0), h4 = half(4.0);
  return SH_C3_HALF[0] * y * (h3 * xx - yy) * feat.f9 +
         SH_C3_HALF[1] * xy * z * feat.f10 +
         SH_C3_HALF[2] * y * (h4 * zz - xx - yy) * feat.f11 +
         SH_C3_HALF[3] * z * (h2 * zz - h3 * xx - h3 * yy) * feat.f12 +
         SH_C3_HALF[4] * x * (h4 * zz - xx - yy) * feat.f13 +
         SH_C3_HALF[5] * z * (xx - yy) * feat.f14 +
         SH_C3_HALF[6] * x * (xx - h3 * yy) * feat.f15;
}

[Differentiable]
half3 half_eval_color(half3 dir, HalfFeatures feat, no_diff uint sh_degree) {
  half3 color = half_eval_sh_col0(dir, feat);
  if (sh_degree > 0) {
    color += half_eval_sh_col1(dir, feat);
    if (sh_degree > 1) {
      color += half_eval_sh_col2(dir, feat);
      if (sh_degree > 2) {
        color += half_eval_sh_col3(dir, feat);
      }
    }
  }
  return color;
}
