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
import sh;

// ============================================================================
// Pointer-based data accessors
// ============================================================================

float3 get_float3_ptr(float* data, uint ind, uint stride) {
    uint base = ind * stride;
    return { data[base], data[base + 1], data[base + 2] };
}

float3 get_float3_3d_ptr(float* data, uint ind, uint feat_ind, uint feat_stride, uint elem_stride) {
    uint base = ind * feat_stride + feat_ind * elem_stride;
    return { data[base], data[base + 1], data[base + 2] };
}

void add_float3_3d_ptr(float* data, uint ind, uint feat_ind, uint feat_stride, uint elem_stride, float3 val) {
    uint base = ind * feat_stride + feat_ind * elem_stride;
    data[base + 0] += val.x;
    data[base + 1] += val.y;
    data[base + 2] += val.z;
}

Features get_feats_ptr(float* features, uint prim_ind, uint sh_degree, uint feature_size) {
    Features feat;
    feat.f0 = get_float3_3d_ptr(features, prim_ind, 0u, feature_size * 3, 3);
    if (sh_degree > 0) {
        feat.f1 = get_float3_3d_ptr(features, prim_ind, 1u, feature_size * 3, 3);
        feat.f2 = get_float3_3d_ptr(features, prim_ind, 2u, feature_size * 3, 3);
        feat.f3 = get_float3_3d_ptr(features, prim_ind, 3u, feature_size * 3, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3_3d_ptr(features, prim_ind, 4u, feature_size * 3, 3);
            feat.f5 = get_float3_3d_ptr(features, prim_ind, 5u, feature_size * 3, 3);
            feat.f6 = get_float3_3d_ptr(features, prim_ind, 6u, feature_size * 3, 3);
            feat.f7 = get_float3_3d_ptr(features, prim_ind, 7u, feature_size * 3, 3);
            feat.f8 = get_float3_3d_ptr(features, prim_ind, 8u, feature_size * 3, 3);
            if (sh_degree > 2) {
                feat.f9 = get_float3_3d_ptr(features, prim_ind, 9u, feature_size * 3, 3);
                feat.f10 = get_float3_3d_ptr(features, prim_ind, 10u, feature_size * 3, 3);
                feat.f11 = get_float3_3d_ptr(features, prim_ind, 11u, feature_size * 3, 3);
                feat.f12 = get_float3_3d_ptr(features, prim_ind, 12u, feature_size * 3, 3);
                feat.f13 = get_float3_3d_ptr(features, prim_ind, 13u, feature_size * 3, 3);
                feat.f14 = get_float3_3d_ptr(features, prim_ind, 14u, feature_size * 3, 3);
                feat.f15 = get_float3_3d_ptr(features, prim_ind, 15u, feature_size * 3, 3);
            }
        }
    }
    return feat;
}

// ============================================================================
// Kernel parameters
// ============================================================================

struct SHKernelParams {
    float* means;        // [N, 3]
    float* features;     // [N, feature_size, 3]
    float* ray_origin;   // [3]
    float* colors;       // [N, 3]
    uint num_prims;
    uint sh_degree;
    uint feature_size;
};

struct SHBackwardParams {
    float* means;           // [N, 3]
    float* features;        // [N, feature_size, 3]
    float* dL_dfeatures;    // [N, feature_size, 3]
    float* ray_origin;      // [3]
    float* dL_dcolors;      // [N, 3]
    uint num_prims;
    uint sh_degree;
    uint feature_size;
};

// ============================================================================
// Forward kernel
// ============================================================================

[CUDAKernel]
void sh_kernel(uniform SHKernelParams params)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;

    if (prim_ind >= params.num_prims) {
        return;
    }

    float3 rayo = { params.ray_origin[0], params.ray_origin[1], params.ray_origin[2] };
    float3 mean = get_float3_ptr(params.means, prim_ind, 3);

    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats_ptr(params.features, prim_ind, params.sh_degree, params.feature_size);
    float3 color = eval_color(dir_pp, feat, params.sh_degree);

    uint base = prim_ind * 3;
    params.colors[base + 0] = color.x;
    params.colors[base + 1] = color.y;
    params.colors[base + 2] = color.z;
}

// ============================================================================
// Backward kernel
// ============================================================================

[CUDAKernel]
void bw_sh_kernel(uniform SHBackwardParams params)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;

    if (prim_ind >= params.num_prims) {
        return;
    }

    float3 deriv_color = get_float3_ptr(params.dL_dcolors, prim_ind, 3);

    // Skip if gradient is tiny
    if ((abs(deriv_color.x) < 1e-8) && (abs(deriv_color.y) < 1e-8) && (abs(deriv_color.z) < 1e-8)) {
        return;
    }

    float3 rayo = { params.ray_origin[0], params.ray_origin[1], params.ray_origin[2] };
    float3 mean = get_float3_ptr(params.means, prim_ind, 3);
    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats_ptr(params.features, prim_ind, params.sh_degree, params.feature_size);

    var deriv_direction = diffPair(dir_pp, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, params.sh_degree, deriv_color);

    add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 0u, params.feature_size * 3, 3, d_feat.d.f0);
    if (params.sh_degree > 0) {
        add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 1u, params.feature_size * 3, 3, d_feat.d.f1);
        add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 2u, params.feature_size * 3, 3, d_feat.d.f2);
        add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 3u, params.feature_size * 3, 3, d_feat.d.f3);
        if (params.sh_degree > 1) {
            add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 4u, params.feature_size * 3, 3, d_feat.d.f4);
            add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 5u, params.feature_size * 3, 3, d_feat.d.f5);
            add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 6u, params.feature_size * 3, 3, d_feat.d.f6);
            add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 7u, params.feature_size * 3, 3, d_feat.d.f7);
            add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 8u, params.feature_size * 3, 3, d_feat.d.f8);
            if (params.sh_degree > 2) {
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 9u, params.feature_size * 3, 3, d_feat.d.f9);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 10u, params.feature_size * 3, 3, d_feat.d.f10);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 11u, params.feature_size * 3, 3, d_feat.d.f11);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 12u, params.feature_size * 3, 3, d_feat.d.f12);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 13u, params.feature_size * 3, 3, d_feat.d.f13);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 14u, params.feature_size * 3, 3, d_feat.d.f14);
                add_float3_3d_ptr(params.dL_dfeatures, prim_ind, 15u, params.feature_size * 3, 3, d_feat.d.f15);
            }
        }
    }
}
