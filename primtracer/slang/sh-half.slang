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

// Spherical harmonics coefficients
static const half SH_C0 = half(0.28209479177387814);
static const half SH_C1 = half(0.4886025119029199);
static const half SH_C2[] = {
  half(1.0925484305920792), half(-1.0925484305920792), half(0.31539156525252005), half(-1.0925484305920792),
  half(0.5462742152960396)
};
static const half SH_C3[] = {
  half(-0.5900435899266435), half(2.890611442640554), half(-0.4570457994644658), half(0.3731763325901154),
  half(-0.4570457994644658), half(1.445305721320277), half(-0.5900435899266435)
};

struct HalfSHFeatures {
  int idx; 
  int max_deg;
  RWStructuredBuffer<half> shs;
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
  return SH_C0 * feat.f0 + half(0.5);
}

[Differentiable]
half3 half_eval_sh_col1(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half3 color = -SH_C1 * y * feat.f1 + SH_C1 * z * feat.f2 - SH_C1 * x * feat.f3;
  return color;
}

[Differentiable]
half3 half_eval_sh_col2(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half xx = x * x, yy = y * y, zz = z * z;
  half xy = x * y, yz = y * z, xz = x * z;
  return SH_C2[0] * xy * feat.f4 + SH_C2[1] * yz * feat.f5 +
         SH_C2[2] * (half(2.0) * zz - xx - yy) * feat.f6 +
         SH_C2[3] * xz * feat.f7 + SH_C2[4] * (xx - yy) * feat.f8;
}

[Differentiable]
half3 half_eval_sh_col3(half3 dir, HalfFeatures feat) {
  half x = dir.x;
  half y = dir.y;
  half z = dir.z;
  half xx = x * x, yy = y * y, zz = z * z;
  half xy = x * y, yz = y * z, xz = x * z;
  half h2 = half(2.0), h3 = half(3.0), h4 = half(4.0);
  return SH_C3[0] * y * (h3 * xx - yy) * feat.f9 +
         SH_C3[1] * xy * z * feat.f10 +
         SH_C3[2] * y * (h4 * zz - xx - yy) * feat.f11 +
         SH_C3[3] * z * (h2 * zz - h3 * xx - h3 * yy) * feat.f12 +
         SH_C3[4] * x * (h4 * zz - xx - yy) * feat.f13 +
         SH_C3[5] * z * (xx - yy) * feat.f14 +
         SH_C3[6] * x * (xx - h3 * yy) * feat.f15;
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

