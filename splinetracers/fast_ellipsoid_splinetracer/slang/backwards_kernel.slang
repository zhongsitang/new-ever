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

#define tri_per_g 2
import spline_machine;
import optix;
import tri_intersect;
import safe_math;
import sh;

// ============================================================================
// Pointer-based data accessors (replaces TensorView)
// ============================================================================

float3 get_float3(float* data, uint ind, uint stride) {
    uint base = ind * stride;
    return { data[base], data[base + 1], data[base + 2] };
}

float3 get_float3_3d(float* data, uint ind, uint feat_ind, uint feat_stride, uint elem_stride) {
    uint base = ind * feat_stride + feat_ind * elem_stride;
    return { data[base], data[base + 1], data[base + 2] };
}

float4 get_float4(float* data, uint ind, uint stride) {
    uint base = ind * stride;
    return { data[base], data[base + 1], data[base + 2], data[base + 3] };
}

Features get_feats(float* features, uint prim_ind, uint sh_degree, uint feature_size) {
    Features feat;
    feat.f0 = get_float3_3d(features, prim_ind, 0u, feature_size * 3, 3);
    if (sh_degree > 0) {
        feat.f1 = get_float3_3d(features, prim_ind, 1u, feature_size * 3, 3);
        feat.f2 = get_float3_3d(features, prim_ind, 2u, feature_size * 3, 3);
        feat.f3 = get_float3_3d(features, prim_ind, 3u, feature_size * 3, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3_3d(features, prim_ind, 4u, feature_size * 3, 3);
            feat.f5 = get_float3_3d(features, prim_ind, 5u, feature_size * 3, 3);
            feat.f6 = get_float3_3d(features, prim_ind, 6u, feature_size * 3, 3);
            feat.f7 = get_float3_3d(features, prim_ind, 7u, feature_size * 3, 3);
            feat.f8 = get_float3_3d(features, prim_ind, 8u, feature_size * 3, 3);
            if (sh_degree > 2) {
                feat.f9 = get_float3_3d(features, prim_ind, 9u, feature_size * 3, 3);
                feat.f10 = get_float3_3d(features, prim_ind, 10u, feature_size * 3, 3);
                feat.f11 = get_float3_3d(features, prim_ind, 11u, feature_size * 3, 3);
                feat.f12 = get_float3_3d(features, prim_ind, 12u, feature_size * 3, 3);
                feat.f13 = get_float3_3d(features, prim_ind, 13u, feature_size * 3, 3);
                feat.f14 = get_float3_3d(features, prim_ind, 14u, feature_size * 3, 3);
                feat.f15 = get_float3_3d(features, prim_ind, 15u, feature_size * 3, 3);
            }
        }
    }
    return feat;
}

void atomic_add_float3_3d(float* data, uint ind, uint feat_ind, uint feat_stride, uint elem_stride, float3 val) {
    uint base = ind * feat_stride + feat_ind * elem_stride;
    __atomicAdd(&data[base + 0], val.x);
    __atomicAdd(&data[base + 1], val.y);
    __atomicAdd(&data[base + 2], val.z);
}

void atomic_add_float2(float* data, uint ind, uint stride, float2 val) {
    uint base = ind * stride;
    __atomicAdd(&data[base + 0], val.x);
    __atomicAdd(&data[base + 1], val.y);
}

void atomic_add_float3(float* data, uint ind, uint stride, float3 val) {
    uint base = ind * stride;
    __atomicAdd(&data[base + 0], val.x);
    __atomicAdd(&data[base + 1], val.y);
    __atomicAdd(&data[base + 2], val.z);
}

void atomic_add_float4(float* data, uint ind, uint stride, float4 val) {
    uint base = ind * stride;
    __atomicAdd(&data[base + 0], val.x);
    __atomicAdd(&data[base + 1], val.y);
    __atomicAdd(&data[base + 2], val.z);
    __atomicAdd(&data[base + 3], val.w);
}

SplineState get_state(float* states, uint ind, uint stride) {
    uint base = ind * stride;
    return {
        float2(states[base + 0], states[base + 1]),
        float2(states[base + 2], states[base + 3]),
        float3(states[base + 4], states[base + 5], states[base + 6]),
        states[base + 7],
        float4(states[base + 8], states[base + 9], states[base + 10], states[base + 11]),
        states[base + 12],
        float3(states[base + 13], states[base + 14], states[base + 15]),
    };
}

// ============================================================================
// Model structure with raw pointers
// ============================================================================

struct DualModel {
    // Inputs
    float* means;          // [N, 3]
    float* scales;         // [N, 3]
    float* quats;          // [N, 4]
    float* densities;      // [N]
    float* features;       // [N, feature_size, 3]

    // Gradients (outputs)
    float* dL_dmeans;      // [N, 3]
    float* dL_dscales;     // [N, 3]
    float* dL_dquats;      // [N, 4]
    float* dL_ddensities;  // [N]
    float* dL_dfeatures;   // [N, feature_size, 3]
    float* dL_drayos;      // [num_rays, 3]
    float* dL_drayds;      // [num_rays, 3]
    float* dL_dmeans2D;    // [N, 2]

    // Dimensions
    uint num_prims;
    uint feature_size;
};

// ============================================================================
// Backward pass core logic
// ============================================================================

DifferentialPair<SplineState>
run_update(in SplineState old_dual_state,
           in ControlPoint old_ctrl_pt,
           in ControlPoint ctrl_pt,
           in uint prim_ind,
           in uint face_id,
           in uint ray_ind,
           in DifferentialPair<SplineState> deriv_state,
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
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_ctrl_pt), {});
    var deriv_ctrl_pt = diffPair(ctrl_pt, {});
    bool skip_close = false;

    bwd_diff(update)(old_deriv_state, deriv_ctrl_pt, tmin, tmax, max_prim_size, deriv_state.d);

    // Propagate to ctrl pt
    let mean = get_float3(model.means, prim_ind, 3);
    let scale = get_float3(model.scales, prim_ind, 3);
    let quat = get_float4(model.quats, prim_ind, 4);
    let density = model.densities[prim_ind];
    Features feat = get_feats(model.features, prim_ind, sh_degree, model.feature_size);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_scales = diffPair(scale, {});
    var deriv_mean = diffPair(mean, {});
    var deriv_quat = diffPair(quat, {});
    var deriv_color = diffPair(color, {});
    var deriv_density = diffPair(density, {});

    bwd_diff(safe_intersect)(deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density, face_id, skip_close, deriv_ctrl_pt.d);

    atomic_add_float3(model.dL_dmeans, prim_ind, 3, deriv_mean.d);
    atomic_add_float3(model.dL_dscales, prim_ind, 3, deriv_scales.d);
    atomic_add_float4(model.dL_dquats, prim_ind, 4, deriv_quat.d);
    __atomicAdd(&model.dL_ddensities[prim_ind], deriv_density.d);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 0u, model.feature_size * 3, 3, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 1u, model.feature_size * 3, 3, d_feat.d.f1);
        atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 2u, model.feature_size * 3, 3, d_feat.d.f2);
        atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 3u, model.feature_size * 3, 3, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 4u, model.feature_size * 3, 3, d_feat.d.f4);
            atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 5u, model.feature_size * 3, 3, d_feat.d.f5);
            atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 6u, model.feature_size * 3, 3, d_feat.d.f6);
            atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 7u, model.feature_size * 3, 3, d_feat.d.f7);
            atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 8u, model.feature_size * 3, 3, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 9u, model.feature_size * 3, 3, d_feat.d.f9);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 10u, model.feature_size * 3, 3, d_feat.d.f10);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 11u, model.feature_size * 3, 3, d_feat.d.f11);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 12u, model.feature_size * 3, 3, d_feat.d.f12);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 13u, model.feature_size * 3, 3, d_feat.d.f13);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 14u, model.feature_size * 3, 3, d_feat.d.f14);
                atomic_add_float3_3d(model.dL_dfeatures, prim_ind, 15u, model.feature_size * 3, 3, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(model.dL_drayos, ray_ind, 3, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_ind, 3, d_rayd);

    float3 xyd = project(mean, wct);

    var d_xy = diffPair(float2(xyd.x, xyd.y), {});
    var d_dist = diffPair(xyd.z, {});
    var d_inv_wct = diffPair(inv_wct, {});

    bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
    atomic_add_float2(model.dL_dmeans2D, prim_ind, 2, d_xy.d);
    return old_deriv_state;
}

ControlPoint load_ctrl_pt(in uint older_tri_ind, in DualModel model, in float3 origin, float3 direction, uint sh_degree, bool skip_close)
{
    let older_prim_ind = (uint)floor(older_tri_ind / tri_per_g);
    let older_face_id = mod(older_tri_ind, tri_per_g);

    let older_mean = get_float3(model.means, older_prim_ind, 3);
    let older_scale = get_float3(model.scales, older_prim_ind, 3);
    let older_quat = get_float4(model.quats, older_prim_ind, 4);
    let older_density = model.densities[older_prim_ind];

    Features older_feat = get_feats(model.features, older_prim_ind, sh_degree, model.feature_size);
    float3 older_color = eval_color(direction, older_feat, sh_degree);

    return safe_intersect(origin, direction,
        older_scale, older_mean, older_quat, older_color, older_density, older_face_id, skip_close);
}

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

// ============================================================================
// Backward kernel parameters (passed from C++)
// ============================================================================

struct BackwardParams {
    // Saved tensors from forward
    float* last_state;        // [num_rays, 16]
    float* last_dirac;        // [num_rays, 4]
    int* iters;               // [num_rays]
    int* tri_collection;      // [num_rays * max_iters]

    // Ray data
    float* ray_origins;       // [num_rays, 3]
    float* ray_directions;    // [num_rays, 3]

    // Initial touch data
    float* initial_drgb;      // [num_rays, 4]
    float* dL_dinital_drgb;   // [num_rays, 4]
    int* touch_count;         // [num_prims]
    int* initial_inds;        // [num_initial]

    // Gradient input
    float* dL_doutputs;       // [num_rays, 5]

    // World-camera transforms
    float* wcts;              // [num_wcts, 4, 4]

    // Model data
    DualModel model;

    // Scalar params
    float tmin;
    float tmax;
    float max_prim_size;
    uint max_iters;
    uint num_rays;
    uint num_wcts;
    uint num_initial;
};

// ============================================================================
// Main backward kernel
// ============================================================================

[CUDAKernel]
void backwards_kernel(uniform BackwardParams params)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= params.num_rays) {
        return;
    }

    uint state_stride = 16;  // sizeof(SplineState) / sizeof(float)
    var dual_state = get_state(params.last_state, ray_ind, state_stride);
    let direction = get_float3(params.ray_directions, ray_ind, 3);
    let origin = get_float3(params.ray_origins, ray_ind, 3) + params.tmin * direction;
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});
    float start_t = dual_state.t;
    let bw_origin = origin + dual_state.t * direction;

    // Load gradient
    uint grad_base = ray_ind * 5;
    float4 dL_dC = {
        params.dL_doutputs[grad_base],
        params.dL_doutputs[grad_base + 1],
        params.dL_doutputs[grad_base + 2],
        params.dL_doutputs[grad_base + 3]
    };
    let dL_ddistortion_loss = params.dL_doutputs[grad_base + 4];

    SplineOutput.Differential dL_doutput;
    dL_doutput.C = {dL_dC.x, dL_dC.y, dL_dC.z};
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    uint num_iters = max(min(params.iters[ray_ind], params.max_iters), 0);
    if (params.iters[ray_ind] >= params.max_iters - 1 || params.iters[ray_ind] <= 0) return;

    var dtmin = diffPair(params.tmin, {});
    var dtmax = diffPair(params.tmax, {});
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let sh_degree = int(sqrt(float(params.model.feature_size))) - 1;

    uint start_id = ray_ind * params.max_iters;
    let wct_ind = (ray_ind < params.num_wcts) ? ray_ind : 0;

    // Load world-camera transform
    uint wct_base = wct_ind * 16;
    float4x4 wct = {
        params.wcts[wct_base + 0], params.wcts[wct_base + 1], params.wcts[wct_base + 2], params.wcts[wct_base + 3],
        params.wcts[wct_base + 4], params.wcts[wct_base + 5], params.wcts[wct_base + 6], params.wcts[wct_base + 7],
        params.wcts[wct_base + 8], params.wcts[wct_base + 9], params.wcts[wct_base + 10], params.wcts[wct_base + 11],
        params.wcts[wct_base + 12], params.wcts[wct_base + 13], params.wcts[wct_base + 14], params.wcts[wct_base + 15],
    };

    float4x4 inv_wct = inverse(wct);

    uint tri_ind = params.tri_collection[ray_ind + max(num_iters - 1, 0) * params.num_rays];
    ControlPoint ctrl_pt = load_ctrl_pt(tri_ind, params.model, origin, direction, sh_degree, skip_close);

    // Main backward loop
    for (int i = num_iters; i-- > 0; )
    {
        uint old_tri_ind;
        ControlPoint old_ctrl_pt;
        if (i - 1 >= 0) {
            old_tri_ind = params.tri_collection[ray_ind + (i - 1) * params.num_rays];
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, params.model, origin, direction, sh_degree, skip_close);
        } else {
            old_ctrl_pt.t = 0;
            old_ctrl_pt.dirac = {0.f, 0.f, 0.f, 0.f};
        }

        SplineState old_dual_state = inverse_update_dual(dual_state, ctrl_pt, old_ctrl_pt, params.tmin, params.tmax);

        let old_deriv_state = run_update(
                old_dual_state,
                old_ctrl_pt,
                ctrl_pt,
                (uint)floor(tri_ind / tri_per_g),
                mod(tri_ind, tri_per_g),
                ray_ind,
                deriv_state,
                origin, direction, params.tmin, params.tmax, sh_degree, params.max_prim_size,
                wct, inv_wct, params.model);

        __atomicAdd(&params.touch_count[(uint)floor(tri_ind / tri_per_g)], 1);

        tri_ind = old_tri_ind;
        dual_state = old_dual_state;
        ctrl_pt = old_ctrl_pt;

        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }

    // Write dL_dinital_drgb
    uint drgb_base = ray_ind * 4;
    params.dL_dinital_drgb[drgb_base + 0] = deriv_state.d.drgb.x;
    params.dL_dinital_drgb[drgb_base + 1] = deriv_state.d.drgb.y;
    params.dL_dinital_drgb[drgb_base + 2] = deriv_state.d.drgb.z;
    params.dL_dinital_drgb[drgb_base + 3] = deriv_state.d.drgb.w;
}

[Differentiable]
float4 mix_drgb(float density, float3 color) {
    return {density, density*color.x, density*color.y, density*color.z};
}

// ============================================================================
// Initial DRGB backward kernel
// ============================================================================

struct InitialDrgbParams {
    float* ray_origins;
    float* ray_directions;
    DualModel model;
    float* initial_drgb;
    int* initial_inds;
    float* dL_dinital_drgb;
    int* touch_count;
    float tmin;
    uint num_rays;
    uint num_initial;
};

[CUDAKernel]
void backwards_initial_drgb_kernel(uniform InitialDrgbParams params)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= params.num_initial || thread_j >= params.num_rays)
        return;

    uint prim_ind = params.initial_inds[thread_i];
    uint ray_ind = thread_j;

    float3 mean = get_float3(params.model.means, prim_ind, 3);
    float4 quat = get_float4(params.model.quats, prim_ind, 4);
    float3 scales = get_float3(params.model.scales, prim_ind, 3);
    float3 rayd = get_float3(params.ray_directions, ray_ind, 3);
    float3 rayo = get_float3(params.ray_origins, 0, 3) + params.tmin * rayd;
    float3 clip_scale = max(scales, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));

    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);
    if (length(Trayo) <= 1) {
        let density = params.model.densities[prim_ind];
        int sh_degree = 0;
        Features feat = get_feats(params.model.features, prim_ind, sh_degree, params.model.feature_size);
        float3 color = eval_color(rayd, feat, sh_degree);
        var deriv_color = diffPair(color, {});
        var deriv_density = diffPair(density, {});
        float4 vdL_dinital_drgb = get_float4(params.dL_dinital_drgb, ray_ind, 4);
        bwd_diff(mix_drgb)(deriv_density, deriv_color, vdL_dinital_drgb);

        __atomicAdd(&params.model.dL_ddensities[prim_ind], deriv_density.d);

        var deriv_direction = diffPair(rayd, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        float3 dfeat = {
            d_feat.d.f0.x,
            d_feat.d.f0.y,
            d_feat.d.f0.z
        };
        atomic_add_float3_3d(params.model.dL_dfeatures, prim_ind, 0, params.model.feature_size * 3, 3, dfeat);
    }
}
