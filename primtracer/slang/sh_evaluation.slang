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
import tensor_utils;
import spherical_harmonics;

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
    float3 color = eval_color(dir_pp, feat);
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
    bwd_diff(eval_color)(deriv_direction, d_feat, deriv_color);

    uint num_coeff = (feat.degree + 1) * (feat.degree + 1);
    for (uint i = 0; i < num_coeff; i++) {
        add_float3(dL_dfeatures, prim_ind, i, d_feat.d.c[i]);
    }
}
