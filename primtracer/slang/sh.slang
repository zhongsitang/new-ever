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
// Spherical harmonics coefficients
// =============================================================================

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

// =============================================================================
// SH feature structures
// =============================================================================

struct SHFeatures {
  int idx;
  int max_deg;
  RWStructuredBuffer<float> shs;
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

// =============================================================================
// SH evaluation functions
// =============================================================================

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
// Tensor utility functions for SH kernels
// =============================================================================

float3 get_float3(TensorView<float> view, uint ind) {
    return {view[ind, 0], view[ind, 1], view[ind, 2]};
}

float3 get_float3(TensorView<float> view, uint ind, uint feat_ind) {
    return {
        view[ind, feat_ind, 0],
        view[ind, feat_ind, 1],
        view[ind, feat_ind, 2]
    };
}

Features get_feats(in TensorView<float> features, in uint prim_ind, in uint sh_degree) {
    Features feat;
    feat.f0 = get_float3(features, prim_ind, 0);
    if (sh_degree > 0) {
        feat.f1 = get_float3(features, prim_ind, 1);
        feat.f2 = get_float3(features, prim_ind, 2);
        feat.f3 = get_float3(features, prim_ind, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3(features, prim_ind, 4);
            feat.f5 = get_float3(features, prim_ind, 5);
            feat.f6 = get_float3(features, prim_ind, 6);
            feat.f7 = get_float3(features, prim_ind, 7);
            feat.f8 = get_float3(features, prim_ind, 8);
            if (sh_degree > 2) {
                feat.f9 = get_float3(features, prim_ind, 9);
                feat.f10 = get_float3(features, prim_ind, 10);
                feat.f11 = get_float3(features, prim_ind, 11);
                feat.f12 = get_float3(features, prim_ind, 12);
                feat.f13 = get_float3(features, prim_ind, 13);
                feat.f14 = get_float3(features, prim_ind, 14);
                feat.f15 = get_float3(features, prim_ind, 15);
            }
        }
    }
    return feat;
}

void add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    view[ind, feat_ind, 0] += val.x;
    view[ind, feat_ind, 1] += val.y;
    view[ind, feat_ind, 2] += val.z;
}

// =============================================================================
// CUDA kernels for SH evaluation
// =============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void sh_kernel(
    TensorView<float> means,
    TensorView<float> features,
    TensorView<float> ray_origin,
    TensorView<float> colors,
    uint sh_degree)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;
    float3 rayo = {ray_origin[0], ray_origin[1], ray_origin[2]};

    if (prim_ind >= means.size(0)) {
        return;
    }
    float3 mean = get_float3(means, prim_ind);
    // First, check
    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats(features, prim_ind, sh_degree);
    float3 color = eval_color(dir_pp, feat, sh_degree);
    colors[prim_ind, 0] = color.x;
    colors[prim_ind, 1] = color.y;
    colors[prim_ind, 2] = color.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void bw_sh_kernel(
    TensorView<float> means,
    TensorView<float> features,
    TensorView<float> dL_dfeatures,
    TensorView<float> ray_origin,
    TensorView<float> dL_dcolors,
    uint sh_degree)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;
    float3 rayo = {ray_origin[0], ray_origin[1], ray_origin[2]};

    if (prim_ind >= means.size(0)) {
        return;
    }

    float3 deriv_color = get_float3(dL_dcolors, prim_ind);
    // skip if gradient is tiny
    if ((abs(deriv_color.x) < 1e-8) && (abs(deriv_color.y) < 1e-8) && (abs(deriv_color.z) < 1e-8)) {
        return;
    }

    float3 mean = get_float3(means, prim_ind);
    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats(features, prim_ind, sh_degree);

    var deriv_direction = diffPair(dir_pp, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color);

    add_float3(dL_dfeatures, prim_ind, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        add_float3(dL_dfeatures, prim_ind, 1u, d_feat.d.f1);
        add_float3(dL_dfeatures, prim_ind, 2u, d_feat.d.f2);
        add_float3(dL_dfeatures, prim_ind, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            add_float3(dL_dfeatures, prim_ind, 4u, d_feat.d.f4);
            add_float3(dL_dfeatures, prim_ind, 5u, d_feat.d.f5);
            add_float3(dL_dfeatures, prim_ind, 6u, d_feat.d.f6);
            add_float3(dL_dfeatures, prim_ind, 7u, d_feat.d.f7);
            add_float3(dL_dfeatures, prim_ind, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                add_float3(dL_dfeatures, prim_ind, 9u, d_feat.d.f9);
                add_float3(dL_dfeatures, prim_ind, 10u, d_feat.d.f10);
                add_float3(dL_dfeatures, prim_ind, 11u, d_feat.d.f11);
                add_float3(dL_dfeatures, prim_ind, 12u, d_feat.d.f12);
                add_float3(dL_dfeatures, prim_ind, 13u, d_feat.d.f13);
                add_float3(dL_dfeatures, prim_ind, 14u, d_feat.d.f14);
                add_float3(dL_dfeatures, prim_ind, 15u, d_feat.d.f15);
            }
        }
    }
}
