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
// Backward Pass Kernels for Differentiable Volume Rendering
// =============================================================================

#define SURFACES_PER_PRIM 2

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// -----------------------------------------------------------------------------
// Tensor Access Utilities
// -----------------------------------------------------------------------------

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

Features get_feats(in TensorView<float> features, in uint prim_idx, in uint sh_degree) {
    Features feat;
    feat.f0 = get_float3(features, prim_idx, 0);
    if (sh_degree > 0) {
        feat.f1 = get_float3(features, prim_idx, 1);
        feat.f2 = get_float3(features, prim_idx, 2);
        feat.f3 = get_float3(features, prim_idx, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3(features, prim_idx, 4);
            feat.f5 = get_float3(features, prim_idx, 5);
            feat.f6 = get_float3(features, prim_idx, 6);
            feat.f7 = get_float3(features, prim_idx, 7);
            feat.f8 = get_float3(features, prim_idx, 8);
            if (sh_degree > 2) {
                feat.f9 = get_float3(features, prim_idx, 9);
                feat.f10 = get_float3(features, prim_idx, 10);
                feat.f11 = get_float3(features, prim_idx, 11);
                feat.f12 = get_float3(features, prim_idx, 12);
                feat.f13 = get_float3(features, prim_idx, 13);
                feat.f14 = get_float3(features, prim_idx, 14);
                feat.f15 = get_float3(features, prim_idx, 15);
            }
        }
    }
    return feat;
}

// -----------------------------------------------------------------------------
// Atomic Accumulation Utilities
// -----------------------------------------------------------------------------

void atomic_add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(ind, feat_ind, 0u), val.x, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 1u), val.y, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 2u), val.z, temp);
}

void atomic_add_float2(TensorView<float> view, uint ind, float2 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
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

// -----------------------------------------------------------------------------
// Volume State Serialization
// -----------------------------------------------------------------------------

VolumeState get_state(TensorView<float> view, uint ind) {
    return {
        float2(view[ind, 0], view[ind, 1]),     // distortion_parts
        float2(view[ind, 2], view[ind, 3]),     // cum_sum
        float3(view[ind, 4], view[ind, 5], view[ind, 6]),  // padding
        view[ind, 7],                           // t
        float4(view[ind, 8], view[ind, 9], view[ind, 10], view[ind, 11]),  // accumulated_drgb

        view[ind, 12],                          // log_transmittance
        float3(view[ind, 13], view[ind, 14], view[ind, 15]),  // color
    };
}

// -----------------------------------------------------------------------------
// Dual Model: primitive parameters and their gradients
// -----------------------------------------------------------------------------

struct DualModel {
    TensorView<float> means;
    TensorView<float> scales;
    TensorView<float> quats;
    TensorView<float> densities;
    TensorView<float> features;

    TensorView<float> dL_dmeans;
    TensorView<float> dL_dscales;
    TensorView<float> dL_dquats;
    TensorView<float> dL_ddensities;
    TensorView<float> dL_dfeatures;
    TensorView<float> dL_drayos;
    TensorView<float> dL_drayds;
    TensorView<float> dL_dmeans2D;
};

// -----------------------------------------------------------------------------
// Backward Update Step
// -----------------------------------------------------------------------------

DifferentialPair<VolumeState>
run_update(in VolumeState old_dual_state,
           in SamplePoint old_sample,
           in SamplePoint sample,
           in uint prim_idx,
           in uint surface_type,
           in uint ray_idx,
           in DifferentialPair<VolumeState> deriv_state,
           in float3 origin,
           in float3 direction,
           in float tmin,
           in float tmax,
           in uint sh_degree,
           in float max_prim_size,
           in float4x4 wct,
           in float4x4 inv_wct,
           inout DualModel model)
{
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_sample), {});
    var deriv_sample = diffPair(sample, {});
    bool skip_close = false;

    bwd_diff(update)(old_deriv_state, deriv_sample, tmin, tmax, max_prim_size, deriv_state.d);

    // Propagate gradients to primitive parameters
    let mean = get_float3(model.means, prim_idx);
    let scale = get_float3(model.scales, prim_idx);
    let quat = get_float4(model.quats, prim_idx);
    let density = model.densities[prim_idx];
    Features feat = get_feats(model.features, prim_idx, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_scales = diffPair(scale, {});
    var deriv_mean = diffPair(mean, {});
    var deriv_quat = diffPair(quat, {});
    var deriv_color = diffPair(color, {});
    var deriv_density = diffPair(density, {});

    bwd_diff(safe_intersect)(deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density, surface_type, skip_close, deriv_sample.d);

    atomic_add_float3(model.dL_dmeans, prim_idx, deriv_mean.d);
    atomic_add_float3(model.dL_dscales, prim_idx, deriv_scales.d);
    atomic_add_float4(model.dL_dquats, prim_idx, deriv_quat.d);
    float temp;
    model.dL_ddensities.InterlockedAdd(prim_idx, deriv_density.d, temp);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3(model.dL_dfeatures, prim_idx, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3(model.dL_dfeatures, prim_idx, 1u, d_feat.d.f1);
        atomic_add_float3(model.dL_dfeatures, prim_idx, 2u, d_feat.d.f2);
        atomic_add_float3(model.dL_dfeatures, prim_idx, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3(model.dL_dfeatures, prim_idx, 4u, d_feat.d.f4);
            atomic_add_float3(model.dL_dfeatures, prim_idx, 5u, d_feat.d.f5);
            atomic_add_float3(model.dL_dfeatures, prim_idx, 6u, d_feat.d.f6);
            atomic_add_float3(model.dL_dfeatures, prim_idx, 7u, d_feat.d.f7);
            atomic_add_float3(model.dL_dfeatures, prim_idx, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3(model.dL_dfeatures, prim_idx, 9u, d_feat.d.f9);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 10u, d_feat.d.f10);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 11u, d_feat.d.f11);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 12u, d_feat.d.f12);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 13u, d_feat.d.f13);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 14u, d_feat.d.f14);
                atomic_add_float3(model.dL_dfeatures, prim_idx, 15u, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(model.dL_drayos, ray_idx, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_idx, d_rayd);

    float3 xyd = project(mean, wct);

    var d_xy = diffPair(float2(xyd.x, xyd.y), {});
    var d_dist = diffPair(xyd.z, {});
    var d_inv_wct = diffPair(inv_wct, {});

    bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
    atomic_add_float2(model.dL_dmeans2D, prim_idx, d_xy.d);
    return old_deriv_state;
}

// -----------------------------------------------------------------------------
// Load Sample from Surface ID
// -----------------------------------------------------------------------------

SamplePoint load_sample(in uint surface_id, in DualModel model, in float3 origin, float3 direction, uint sh_degree, bool skip_close)
{
    let prim_idx = (uint)floor(surface_id / SURFACES_PER_PRIM);
    let surface_type = mod(surface_id, SURFACES_PER_PRIM);

    let mean = get_float3(model.means, prim_idx);
    let scale = get_float3(model.scales, prim_idx);
    let quat = get_float4(model.quats, prim_idx);
    let density = model.densities[prim_idx];

    Features feat = get_feats(model.features, prim_idx, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    return safe_intersect(origin, direction,
        scale, mean, quat, color, density, surface_type, skip_close);
}

// -----------------------------------------------------------------------------
// Projection Utilities
// -----------------------------------------------------------------------------

[Differentiable]
float3 project(in float3 xyz, in float4x4 wct) {
    float4 xyzw = float4(xyz, 1.f);
    let p_view = mul(xyzw, wct);
    float2 pix2d = {safe_div(p_view.x, p_view.z), safe_div(p_view.y, p_view.z)};
    return {pix2d.x, pix2d.y, p_view.z};
}

[Differentiable]
float3 inv_project(in float2 xy, in float dist, float4x4 inv_wvt) {
    let p_hom = float4(xy * dist, dist, 1.f);
    let out = mul(p_hom, inv_wvt);
    return {out.x, out.y, out.z};
}

// -----------------------------------------------------------------------------
// Main Backward Kernel
// -----------------------------------------------------------------------------

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_delta,
    TensorView<int> iters,
    TensorView<int> hit_log,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_accumulated_drgb,
    TensorView<float> dL_d_initial_accumulated_drgb,
    TensorView<int32_t> prim_hit_count,

    TensorView<float> dL_doutputs,
    TensorView<float> wcts,

    float tmin,
    float tmax,
    float max_prim_size,
    uint max_iters)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_idx = dispatchIdx.x;
    if (ray_idx >= ray_origins.size(0)) {
        return;
    }

    var dual_state = get_state(last_state, ray_idx);
    let direction = get_float3(ray_directions, ray_idx);
    let origin = get_float3(ray_origins, ray_idx) + tmin * direction;
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});
    float start_t = dual_state.t;
    let bw_origin = origin + dual_state.t * direction;

    let dL_dC = get_float4(dL_doutputs, ray_idx);
    let dL_ddistortion_loss = dL_doutputs[ray_idx, 4];
    RenderOutput.Differential dL_doutput;
    dL_doutput.C = {dL_dC.x, dL_dC.y, dL_dC.z};
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    uint num_iters = max(min(iters[ray_idx], max_iters), 0);
    if (iters[ray_idx] >= max_iters-1 || iters[ray_idx] <= 0) return;

    var dtmin = diffPair(tmin, {});
    var dtmax = diffPair(tmax, {});
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let feature_size = model.features.size(1);
    let sh_degree = int(sqrt(feature_size)) - 1;

    uint start_id = ray_idx * max_iters;
    let wct_ind = (ray_idx < wcts.size(0)) ? ray_idx : 0;
    float4x4 wct = {
        wcts[wct_ind, 0, 0], wcts[wct_ind, 0, 1], wcts[wct_ind, 0, 2], wcts[wct_ind, 0, 3],
        wcts[wct_ind, 1, 0], wcts[wct_ind, 1, 1], wcts[wct_ind, 1, 2], wcts[wct_ind, 1, 3],
        wcts[wct_ind, 2, 0], wcts[wct_ind, 2, 1], wcts[wct_ind, 2, 2], wcts[wct_ind, 2, 3],
        wcts[wct_ind, 3, 0], wcts[wct_ind, 3, 1], wcts[wct_ind, 3, 2], wcts[wct_ind, 3, 3],
    };

    float4x4 inv_wct = inverse(wct);

    uint surface_id = hit_log[ray_idx + max(num_iters-1, 0) * ray_origins.size(0)];
    SamplePoint sample = load_sample(surface_id, model, origin, direction, sh_degree, skip_close);

    // Backward iterate through all samples
    for (int i = num_iters; i-- > 0; )
    {
        uint old_surface_id;
        SamplePoint old_sample;
        if (i - 1 >= 0) {
            old_surface_id = hit_log[ray_idx + (i-1) * ray_origins.size(0)];
            old_sample = load_sample(old_surface_id, model, origin, direction, sh_degree, skip_close);
        } else {
            old_sample.t = 0;
            old_sample.delta = {0.f, 0.f, 0.f, 0.f};
        }

        VolumeState old_dual_state = inverse_update_dual(dual_state, sample, old_sample, tmin, tmax);

        let old_deriv_state = run_update(
                old_dual_state,
                old_sample,
                sample,
                (uint)floor(surface_id / SURFACES_PER_PRIM),
                mod(surface_id, SURFACES_PER_PRIM),
                ray_idx,
                deriv_state,
                origin, direction, tmin, tmax, sh_degree, max_prim_size,
                wct, inv_wct, model);

        int itemp;
        prim_hit_count.InterlockedAdd((uint)floor(surface_id / SURFACES_PER_PRIM), 1, itemp);

        surface_id = old_surface_id;
        dual_state = old_dual_state;
        sample = old_sample;

        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }

    // Write gradient for initial accumulated drgb
    dL_d_initial_accumulated_drgb[ray_idx, 0u] = deriv_state.d.accumulated_drgb.x;
    dL_d_initial_accumulated_drgb[ray_idx, 1u] = deriv_state.d.accumulated_drgb.y;
    dL_d_initial_accumulated_drgb[ray_idx, 2u] = deriv_state.d.accumulated_drgb.z;
    dL_d_initial_accumulated_drgb[ray_idx, 3u] = deriv_state.d.accumulated_drgb.w;
}

// -----------------------------------------------------------------------------
// Helper for mixing density and color
// -----------------------------------------------------------------------------

[Differentiable]
float4 mix_drgb(float density, float3 color) {
    return {density, density*color.x, density*color.y, density*color.z};
}

// -----------------------------------------------------------------------------
// Backward Kernel for Initial DRGB (rays starting inside primitives)
// -----------------------------------------------------------------------------

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_initial_drgb_kernel(
    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_accumulated_drgb,
    TensorView<int32_t> initial_prim_indices,
    TensorView<float> dL_d_initial_accumulated_drgb,
    TensorView<int32_t> prim_hit_count,
    float tmin,
)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= initial_prim_indices.size(0) || thread_j >= ray_directions.size(0))
        return;

    uint prim_idx = initial_prim_indices[thread_i];
    uint ray_idx = thread_j;

    float3 mean = get_float3(model.means, prim_idx);
    float4 quat = get_float4(model.quats, prim_idx);
    float3 scales = get_float3(model.scales, prim_idx);
    float3 rayd = get_float3(ray_directions, ray_idx);
    float3 rayo = get_float3(ray_origins, 0) + tmin * rayd;
    float3 clip_scale = max(scales, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));
    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);

    if (length(Trayo) <= 1) {
        float temp;
        let density = model.densities[prim_idx];
        int sh_degree = 0;
        Features feat = get_feats(model.features, prim_idx, sh_degree);
        float3 color = eval_color(rayd, feat, sh_degree);

        var deriv_color = diffPair(color, {});
        var deriv_density = diffPair(density, {});
        float4 grad_initial_drgb = get_float4(dL_d_initial_accumulated_drgb, ray_idx);
        bwd_diff(mix_drgb)(deriv_density, deriv_color, grad_initial_drgb);

        model.dL_ddensities.InterlockedAdd(prim_idx, deriv_density.d, temp);

        var deriv_direction = diffPair(rayd, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        float3 dfeat = {
            d_feat.d.f0.x,
            d_feat.d.f0.y,
            d_feat.d.f0.z
        };
        atomic_add_float3(model.dL_dfeatures, prim_idx, 0, dfeat);
    }
}
