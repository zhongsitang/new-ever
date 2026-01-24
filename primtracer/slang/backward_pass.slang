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
// Backward Pass for Ellipsoid Volume Rendering
// =============================================================================
//
// This module implements the backward pass (gradient computation) for
// differentiable volume rendering of ellipsoid primitives.
//
// Hit ID encoding (same as forward pass):
//   hit_id = prim_index * HITS_PER_PRIMITIVE + hit_type
//   where hit_type: 1 = entry, 0 = exit
// =============================================================================

#define HITS_PER_PRIMITIVE 2
#define HIT_TYPE_EXIT 0
#define HIT_TYPE_ENTRY 1

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// =============================================================================
// Tensor Utilities
// =============================================================================

float3 get_float3(TensorView<float> view, uint ind) {
    return {
        view[ind, 0],
        view[ind, 1],
        view[ind, 2]
    };
}

float3 get_float3(TensorView<float> view, uint ind, uint feat_ind) {
    return {
        view[ind, feat_ind, 0],
        view[ind, feat_ind, 1],
        view[ind, feat_ind, 2]
    };
}

float4 get_float4(TensorView<float> view, uint ind) {
    return {
        view[ind, 0],
        view[ind, 1],
        view[ind, 2],
        view[ind, 3]
    };
}

Features get_feats(in TensorView<float> features, in uint prim_index, in uint sh_degree) {
    Features feat;
    feat.f0 = get_float3(features, prim_index, 0);
    if (sh_degree > 0) {
        feat.f1 = get_float3(features, prim_index, 1);
        feat.f2 = get_float3(features, prim_index, 2);
        feat.f3 = get_float3(features, prim_index, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3(features, prim_index, 4);
            feat.f5 = get_float3(features, prim_index, 5);
            feat.f6 = get_float3(features, prim_index, 6);
            feat.f7 = get_float3(features, prim_index, 7);
            feat.f8 = get_float3(features, prim_index, 8);
            if (sh_degree > 2) {
                feat.f9 = get_float3(features, prim_index, 9);
                feat.f10 = get_float3(features, prim_index, 10);
                feat.f11 = get_float3(features, prim_index, 11);
                feat.f12 = get_float3(features, prim_index, 12);
                feat.f13 = get_float3(features, prim_index, 13);
                feat.f14 = get_float3(features, prim_index, 14);
                feat.f15 = get_float3(features, prim_index, 15);
            }
        }
    }
    return feat;
}

void atomic_add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(ind, feat_ind, 0u), val.x, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 1u), val.y, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 2u), val.z, temp);
}

void atomic_add_float3(TensorView<float> view, uint ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
}

void atomic_add_float4(TensorView<float> view, uint ind, float4 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
    view.InterlockedAdd(uint2(ind, 3u), val.w, temp);
}

/// Load IntegratorState from tensor (48 bytes = 12 floats).
/// Layout: [0-3] accumulated_contrib, [4-7] C (w unused), [8] logT, [9] depth_accum, [10] t, [11] pad
IntegratorState get_state(TensorView<float> view, uint ind) {
    IntegratorState state;
    state.accumulated_contrib = float4(view[ind, 0], view[ind, 1], view[ind, 2], view[ind, 3]);
    state.C = float4(view[ind, 4], view[ind, 5], view[ind, 6], view[ind, 7]);
    state.logT = view[ind, 8];
    state.depth_accum = view[ind, 9];
    state.t = view[ind, 10];
    return state;
}

// =============================================================================
// Gradient Model
// =============================================================================

/// Container for model parameters and their gradients.
struct DualModel {
    // Forward parameters
    TensorView<float> means;
    TensorView<float> scales;
    TensorView<float> quats;
    TensorView<float> densities;
    TensorView<float> features;

    // Gradient outputs
    TensorView<float> dL_dmeans;
    TensorView<float> dL_dscales;
    TensorView<float> dL_dquats;
    TensorView<float> dL_ddensities;
    TensorView<float> dL_dfeatures;
    TensorView<float> dL_drayos;
    TensorView<float> dL_drayds;
};

// =============================================================================
// Backward Pass Core Functions
// =============================================================================

/// Run backward pass for a single volume update step.
DifferentialPair<IntegratorState>
run_update(in IntegratorState old_dual_state,
           in SamplePoint old_sample,
           in SamplePoint sample,
           in uint prim_index,
           in uint hit_type,
           in uint ray_ind,
           in DifferentialPair<IntegratorState> deriv_state,
           in float3 origin,
           in float3 direction,
           in float tmin,
           in float tmax,
           in uint sh_degree,
           in float max_prim_size,
           inout DualModel model)
{
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_sample), {});
    var deriv_sample = diffPair(sample, {});
    bool skip_close = false;

    bwd_diff(update)(old_deriv_state, deriv_sample, tmin, tmax, max_prim_size, deriv_state.d);

    // Propagate gradients to primitive parameters
    let mean = get_float3(model.means, prim_index);
    let scale = get_float3(model.scales, prim_index);
    let quat = get_float4(model.quats, prim_index);
    let density = model.densities[prim_index];
    Features feat = get_feats(model.features, prim_index, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_scales = diffPair(scale, {});
    var deriv_mean = diffPair(mean, {});
    var deriv_quat = diffPair(quat, {});
    var deriv_color = diffPair(color, {});
    var deriv_density = diffPair(density, {});

    bwd_diff(safe_intersect)(deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density, hit_type, skip_close, deriv_sample.d);

    atomic_add_float3(model.dL_dmeans, prim_index, deriv_mean.d);
    atomic_add_float3(model.dL_dscales, prim_index, deriv_scales.d);
    atomic_add_float4(model.dL_dquats, prim_index, deriv_quat.d);
    float temp;
    model.dL_ddensities.InterlockedAdd(prim_index, deriv_density.d, temp);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3(model.dL_dfeatures, prim_index, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3(model.dL_dfeatures, prim_index, 1u, d_feat.d.f1);
        atomic_add_float3(model.dL_dfeatures, prim_index, 2u, d_feat.d.f2);
        atomic_add_float3(model.dL_dfeatures, prim_index, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3(model.dL_dfeatures, prim_index, 4u, d_feat.d.f4);
            atomic_add_float3(model.dL_dfeatures, prim_index, 5u, d_feat.d.f5);
            atomic_add_float3(model.dL_dfeatures, prim_index, 6u, d_feat.d.f6);
            atomic_add_float3(model.dL_dfeatures, prim_index, 7u, d_feat.d.f7);
            atomic_add_float3(model.dL_dfeatures, prim_index, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3(model.dL_dfeatures, prim_index, 9u, d_feat.d.f9);
                atomic_add_float3(model.dL_dfeatures, prim_index, 10u, d_feat.d.f10);
                atomic_add_float3(model.dL_dfeatures, prim_index, 11u, d_feat.d.f11);
                atomic_add_float3(model.dL_dfeatures, prim_index, 12u, d_feat.d.f12);
                atomic_add_float3(model.dL_dfeatures, prim_index, 13u, d_feat.d.f13);
                atomic_add_float3(model.dL_dfeatures, prim_index, 14u, d_feat.d.f14);
                atomic_add_float3(model.dL_dfeatures, prim_index, 15u, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(model.dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_ind, d_rayd);

    return old_deriv_state;
}

/// Load sample point from a hit ID.
SamplePoint load_sample(in uint hit_id, in DualModel model, in float3 origin, float3 direction, uint sh_degree, bool skip_close)
{
    let prim_index = (uint)floor(hit_id / HITS_PER_PRIMITIVE);
    let hit_type = mod(hit_id, HITS_PER_PRIMITIVE);

    let mean = get_float3(model.means, prim_index);
    let scale = get_float3(model.scales, prim_index);
    let quat = get_float4(model.quats, prim_index);
    let density = model.densities[prim_index];

    Features feat = get_feats(model.features, prim_index, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    return safe_intersect(origin, direction,
        scale, mean, quat, color, density, hit_type, skip_close);
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/// Main backward pass kernel.
[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_delta_contrib,
    TensorView<int> iters,
    TensorView<int> hit_collection,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_contrib,
    TensorView<float> dL_dinitial_contrib,
    TensorView<int32_t> prim_hits,

    TensorView<float> dL_doutputs,  // (R, G, B, A, depth)

    float tmin,
    TensorView<float> tmax,  // Per-ray tmax
    float max_prim_size,
    uint max_iters)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= ray_origins.size(0)) {
        return;
    }
    var dual_state = get_state(last_state, ray_ind);
    let direction = get_float3(ray_directions, ray_ind);
    let origin = get_float3(ray_origins, ray_ind) + tmin*direction;
    let ray_tmax = tmax[ray_ind];  // Per-ray tmax
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});
    float start_t = dual_state.t;
    let bw_origin = origin + dual_state.t * direction;

    // dL_doutputs format: (R, G, B, A, depth)
    RenderOutput.Differential dL_doutput;
    dL_doutput.C = {dL_doutputs[ray_ind, 0], dL_doutputs[ray_ind, 1], dL_doutputs[ray_ind, 2]};
    dL_doutput.alpha = dL_doutputs[ray_ind, 3];
    dL_doutput.depth = dL_doutputs[ray_ind, 4];

    uint num_iters = max(min(iters[ray_ind], max_iters), 0);
    if (iters[ray_ind] >= max_iters-1 || iters[ray_ind] <= 0) return;
    var dtmin = diffPair(tmin, {});
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let feature_size = model.features.size(1);
    let sh_degree = int(sqrt(feature_size))-1;

    uint hit_id = hit_collection[ray_ind + max(num_iters-1, 0) * ray_origins.size(0)];
    SamplePoint sample = load_sample(hit_id, model, origin, direction, sh_degree, skip_close);

    // Iterate backward through hits
    for (int i=num_iters; i-->0; )
    {
        uint old_hit_id;
        SamplePoint old_sample;
        if (i-1 >= 0) {
            old_hit_id = hit_collection[ray_ind + (i-1) * ray_origins.size(0)];
            old_sample = load_sample(old_hit_id, model, origin, direction, sh_degree, skip_close);
        } else {
            old_sample.t = 0;
            old_sample.delta_contrib = {0.f, 0.f, 0.f, 0.f};
        }

        IntegratorState old_dual_state = inverse_update_dual(dual_state, sample, old_sample, tmin, ray_tmax);

        let old_deriv_state = run_update(
                old_dual_state,
                old_sample,
                sample,
                (uint)floor(hit_id / HITS_PER_PRIMITIVE),
                mod(hit_id, HITS_PER_PRIMITIVE),
                ray_ind,
                deriv_state,
                origin, direction, tmin, ray_tmax, sh_degree, max_prim_size,
                model);
        int itemp;
        prim_hits.InterlockedAdd((uint)floor(hit_id / HITS_PER_PRIMITIVE), 1, itemp);

        hit_id = old_hit_id;
        dual_state = old_dual_state;
        sample = old_sample;

        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }
    float temp;
    dL_dinitial_contrib[ray_ind, 0u] = deriv_state.d.accumulated_contrib.x;
    dL_dinitial_contrib[ray_ind, 1u] = deriv_state.d.accumulated_contrib.y;
    dL_dinitial_contrib[ray_ind, 2u] = deriv_state.d.accumulated_contrib.z;
    dL_dinitial_contrib[ray_ind, 3u] = deriv_state.d.accumulated_contrib.w;
}

/// Mix density and color into contribution vector.
[Differentiable]
float4 mix_contrib(float density, float3 color) {
    return {density, density*color.x, density*color.y, density*color.z};
}

/// Backward pass for initial contribution (rays starting inside primitives).
[AutoPyBindCUDA]
[CUDAKernel]
void backwards_initial_contrib_kernel(
    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_contrib,
    TensorView<int32_t> initial_prim_indices,
    TensorView<float> dL_dinitial_contrib,
    TensorView<int32_t> prim_hits,
    float tmin,
)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= initial_prim_indices.size(0) || thread_j >= ray_directions.size(0))
        return;

    uint prim_index = initial_prim_indices[thread_i];
    uint ray_ind = thread_j;

    float3 mean = get_float3(model.means, prim_index);
    float4 quat = get_float4(model.quats, prim_index);
    float3 scales = get_float3(model.scales, prim_index);
    float3 rayd = get_float3(ray_directions, ray_ind);
    float3 rayo = get_float3(ray_origins, 0) + tmin * rayd;
    float3 clip_scale = max(scales, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));

    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);
    if (length(Trayo) <= 1) {
        float temp;
        let density = model.densities[prim_index];
        int sh_degree = 0;
        Features feat = get_feats(model.features, prim_index, sh_degree);
        float3 color = eval_color(rayd, feat, sh_degree);
        var deriv_color = diffPair(color, {});
        var deriv_density = diffPair(density, {});
        float4 vdL_dinitial_contrib = get_float4(dL_dinitial_contrib, ray_ind);
        bwd_diff(mix_contrib)(deriv_density, deriv_color, vdL_dinitial_contrib);

        model.dL_ddensities.InterlockedAdd(prim_index, deriv_density.d, temp);

        var deriv_direction = diffPair(rayd, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        float3 dfeat = {
            d_feat.d.f0.x,
            d_feat.d.f0.y,
            d_feat.d.f0.z
        };
        atomic_add_float3(model.dL_dfeatures, prim_index, 0, dfeat);
    }
}
