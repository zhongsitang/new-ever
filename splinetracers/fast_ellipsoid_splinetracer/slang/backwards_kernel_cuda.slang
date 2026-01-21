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

// Backwards kernel for direct CUDA compilation (no slangtorch dependency)
// We define TensorView here matching slang-cuda-prelude.h layout, because:
// 1. Slang compiler needs type info at compile time
// 2. Including slang-cuda-prelude.h in C++ would cause duplicate definition
// This is simpler than trying to share the type definition between Slang and C++.

#define tri_per_g 2
import spline_machine;
import optix;
import tri_intersect;
import safe_math;
import sh;

// TensorView - matches slang-cuda-prelude.h layout
static const int kSlangTorchTensorMaxDim = 5;

struct TensorView {
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;
};

// ============================================================================
// TensorView accessors - matching slang-cuda-prelude.h load/store semantics
// ============================================================================

// Load float from 1D index
float tv_load_f(TensorView t, uint i) {
    return *reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i);
}

// Load float from 2D index
float tv_load_f(TensorView t, uint i, uint j) {
    return *reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i + (uint64_t)t.strides[1] * j);
}

// Load float from 3D index
float tv_load_f(TensorView t, uint i, uint j, uint k) {
    return *reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i + (uint64_t)t.strides[1] * j + (uint64_t)t.strides[2] * k);
}

// Load int from 1D index
int tv_load_i(TensorView t, uint i) {
    return *reinterpret_cast<int*>(t.data + (uint64_t)t.strides[0] * i);
}

// Store float to 2D index
void tv_store_f(TensorView t, uint i, uint j, float val) {
    *reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i + (uint64_t)t.strides[1] * j) = val;
}

// Atomic add for float at 1D index
void tv_atomic_add_f(TensorView t, uint i, float val) {
    float* p = reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i);
    __target_switch {
    case cuda: __intrinsic_asm "atomicAdd($0, $1)" : : "l"(p), "f"(val);
    }
}

// Atomic add for float at 2D index
void tv_atomic_add_f(TensorView t, uint i, uint j, float val) {
    float* p = reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i + (uint64_t)t.strides[1] * j);
    __target_switch {
    case cuda: __intrinsic_asm "atomicAdd($0, $1)" : : "l"(p), "f"(val);
    }
}

// Atomic add for float at 3D index
void tv_atomic_add_f(TensorView t, uint i, uint j, uint k, float val) {
    float* p = reinterpret_cast<float*>(t.data + (uint64_t)t.strides[0] * i + (uint64_t)t.strides[1] * j + (uint64_t)t.strides[2] * k);
    __target_switch {
    case cuda: __intrinsic_asm "atomicAdd($0, $1)" : : "l"(p), "f"(val);
    }
}

// Atomic add for int at 1D index
void tv_atomic_add_i(TensorView t, uint i, int val) {
    int* p = reinterpret_cast<int*>(t.data + (uint64_t)t.strides[0] * i);
    __target_switch {
    case cuda: __intrinsic_asm "atomicAdd($0, $1)" : : "l"(p), "r"(val);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

float3 get_float3(TensorView view, uint ind) {
    return float3(
        tv_load_f(view, ind, 0u),
        tv_load_f(view, ind, 1u),
        tv_load_f(view, ind, 2u)
    );
}

float3 get_float3_feat(TensorView view, uint ind, uint feat_ind) {
    return float3(
        tv_load_f(view, ind, feat_ind, 0u),
        tv_load_f(view, ind, feat_ind, 1u),
        tv_load_f(view, ind, feat_ind, 2u)
    );
}

float4 get_float4(TensorView view, uint ind) {
    return float4(
        tv_load_f(view, ind, 0u),
        tv_load_f(view, ind, 1u),
        tv_load_f(view, ind, 2u),
        tv_load_f(view, ind, 3u)
    );
}

Features get_feats(TensorView features, uint prim_ind, uint sh_degree) {
    Features feat;
    feat.f0 = get_float3_feat(features, prim_ind, 0u);
    if (sh_degree > 0) {
        feat.f1 = get_float3_feat(features, prim_ind, 1u);
        feat.f2 = get_float3_feat(features, prim_ind, 2u);
        feat.f3 = get_float3_feat(features, prim_ind, 3u);
        if (sh_degree > 1) {
            feat.f4 = get_float3_feat(features, prim_ind, 4u);
            feat.f5 = get_float3_feat(features, prim_ind, 5u);
            feat.f6 = get_float3_feat(features, prim_ind, 6u);
            feat.f7 = get_float3_feat(features, prim_ind, 7u);
            feat.f8 = get_float3_feat(features, prim_ind, 8u);
            if (sh_degree > 2) {
                feat.f9 = get_float3_feat(features, prim_ind, 9u);
                feat.f10 = get_float3_feat(features, prim_ind, 10u);
                feat.f11 = get_float3_feat(features, prim_ind, 11u);
                feat.f12 = get_float3_feat(features, prim_ind, 12u);
                feat.f13 = get_float3_feat(features, prim_ind, 13u);
                feat.f14 = get_float3_feat(features, prim_ind, 14u);
                feat.f15 = get_float3_feat(features, prim_ind, 15u);
            }
        }
    }
    return feat;
}

void atomic_add_float3_feat(TensorView view, uint ind, uint feat_ind, float3 val) {
    tv_atomic_add_f(view, ind, feat_ind, 0u, val.x);
    tv_atomic_add_f(view, ind, feat_ind, 1u, val.y);
    tv_atomic_add_f(view, ind, feat_ind, 2u, val.z);
}

void atomic_add_float2(TensorView view, uint ind, float2 val) {
    tv_atomic_add_f(view, ind, 0u, val.x);
    tv_atomic_add_f(view, ind, 1u, val.y);
}

void atomic_add_float3(TensorView view, uint ind, float3 val) {
    tv_atomic_add_f(view, ind, 0u, val.x);
    tv_atomic_add_f(view, ind, 1u, val.y);
    tv_atomic_add_f(view, ind, 2u, val.z);
}

void atomic_add_float4(TensorView view, uint ind, float4 val) {
    tv_atomic_add_f(view, ind, 0u, val.x);
    tv_atomic_add_f(view, ind, 1u, val.y);
    tv_atomic_add_f(view, ind, 2u, val.z);
    tv_atomic_add_f(view, ind, 3u, val.w);
}

SplineState get_state(TensorView view, uint ind) {
    return {
        float2(tv_load_f(view, ind, 0u), tv_load_f(view, ind, 1u)),
        float2(tv_load_f(view, ind, 2u), tv_load_f(view, ind, 3u)),
        float3(tv_load_f(view, ind, 4u), tv_load_f(view, ind, 5u), tv_load_f(view, ind, 6u)),
        tv_load_f(view, ind, 7u),
        float4(tv_load_f(view, ind, 8u), tv_load_f(view, ind, 9u), tv_load_f(view, ind, 10u), tv_load_f(view, ind, 11u)),
        tv_load_f(view, ind, 12u),
        float3(tv_load_f(view, ind, 13u), tv_load_f(view, ind, 14u), tv_load_f(view, ind, 15u)),
    };
}

// ============================================================================
// DualModel - holds model parameters and gradient outputs
// ============================================================================

struct DualModel {
    TensorView means;
    TensorView scales;
    TensorView quats;
    TensorView densities;
    TensorView features;

    TensorView dL_dmeans;
    TensorView dL_dscales;
    TensorView dL_dquats;
    TensorView dL_ddensities;
    TensorView dL_dfeatures;
    TensorView dL_drayos;
    TensorView dL_drayds;
    TensorView dL_dmeans2D;
};

// ============================================================================
// Core backward computation
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

    let mean = get_float3(model.means, prim_ind);
    let scale = get_float3(model.scales, prim_ind);
    let quat = get_float4(model.quats, prim_ind);
    let density = tv_load_f(model.densities, prim_ind);
    Features feat = get_feats(model.features, prim_ind, sh_degree);
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

    atomic_add_float3(model.dL_dmeans, prim_ind, deriv_mean.d);
    atomic_add_float3(model.dL_dscales, prim_ind, deriv_scales.d);
    atomic_add_float4(model.dL_dquats, prim_ind, deriv_quat.d);
    tv_atomic_add_f(model.dL_ddensities, prim_ind, deriv_density.d);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 1u, d_feat.d.f1);
        atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 2u, d_feat.d.f2);
        atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 4u, d_feat.d.f4);
            atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 5u, d_feat.d.f5);
            atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 6u, d_feat.d.f6);
            atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 7u, d_feat.d.f7);
            atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 9u, d_feat.d.f9);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 10u, d_feat.d.f10);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 11u, d_feat.d.f11);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 12u, d_feat.d.f12);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 13u, d_feat.d.f13);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 14u, d_feat.d.f14);
                atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 15u, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(model.dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(model.dL_drayds, ray_ind, d_rayd);

    float3 xyd = project(mean, wct);
    var d_xy = diffPair(float2(xyd.x, xyd.y), {});
    var d_dist = diffPair(xyd.z, {});
    var d_inv_wct = diffPair(inv_wct, {});
    bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
    atomic_add_float2(model.dL_dmeans2D, prim_ind, d_xy.d);
    return old_deriv_state;
}

ControlPoint load_ctrl_pt(uint older_tri_ind, DualModel model, float3 origin, float3 direction, uint sh_degree, bool skip_close)
{
    let older_prim_ind = (uint)floor(older_tri_ind / tri_per_g);
    let older_face_id = mod(older_tri_ind, tri_per_g);

    let older_mean = get_float3(model.means, older_prim_ind);
    let older_scale = get_float3(model.scales, older_prim_ind);
    let older_quat = get_float4(model.quats, older_prim_ind);
    let older_density = tv_load_f(model.densities, older_prim_ind);

    Features older_feat = get_feats(model.features, older_prim_ind, sh_degree);
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

[CUDAKernel]
void backwards_kernel(
    TensorView last_state,
    TensorView last_dirac,
    TensorView iters,
    TensorView tri_collection,
    TensorView ray_origins,
    TensorView ray_directions,
    DualModel model,
    TensorView initial_drgb,
    TensorView dL_dinital_drgb,
    TensorView touch_count,
    TensorView dL_doutputs,
    TensorView wcts,
    float tmin,
    float tmax,
    float max_prim_size,
    uint max_iters)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= ray_origins.sizes[0]) {
        return;
    }
    var dual_state = get_state(last_state, ray_ind);
    let direction = get_float3(ray_directions, ray_ind);
    let origin = get_float3(ray_origins, ray_ind) + tmin*direction;
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});
    float start_t = dual_state.t;
    let bw_origin = origin + dual_state.t * direction;

    let dL_dC = get_float4(dL_doutputs, ray_ind);
    let dL_ddistortion_loss = tv_load_f(dL_doutputs, ray_ind, 4u);
    SplineOutput.Differential dL_doutput;
    dL_doutput.C = {dL_dC.x, dL_dC.y, dL_dC.z};
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    uint iters_val = tv_load_i(iters, ray_ind);
    uint num_iters = max(min(iters_val, max_iters), 0u);
    if (iters_val >= max_iters-1 || iters_val <= 0) return;
    var dtmin = diffPair(tmin, {});
    var dtmax = diffPair(tmax, {});
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let feature_size = model.features.sizes[1];
    let sh_degree = uint(sqrt(float(feature_size)))-1;

    uint start_id = ray_ind * max_iters;
    let wct_ind = (ray_ind < wcts.sizes[0]) ? ray_ind : 0u;
    float4x4 wct = {
        tv_load_f(wcts, wct_ind, 0u, 0u), tv_load_f(wcts, wct_ind, 0u, 1u), tv_load_f(wcts, wct_ind, 0u, 2u), tv_load_f(wcts, wct_ind, 0u, 3u),
        tv_load_f(wcts, wct_ind, 1u, 0u), tv_load_f(wcts, wct_ind, 1u, 1u), tv_load_f(wcts, wct_ind, 1u, 2u), tv_load_f(wcts, wct_ind, 1u, 3u),
        tv_load_f(wcts, wct_ind, 2u, 0u), tv_load_f(wcts, wct_ind, 2u, 1u), tv_load_f(wcts, wct_ind, 2u, 2u), tv_load_f(wcts, wct_ind, 2u, 3u),
        tv_load_f(wcts, wct_ind, 3u, 0u), tv_load_f(wcts, wct_ind, 3u, 1u), tv_load_f(wcts, wct_ind, 3u, 2u), tv_load_f(wcts, wct_ind, 3u, 3u),
    };

    float4x4 inv_wct = inverse(wct);

    uint tri_ind = tv_load_i(tri_collection, ray_ind + max(num_iters-1, 0u) * ray_origins.sizes[0]);
    ControlPoint ctrl_pt = load_ctrl_pt(tri_ind, model, origin, direction, sh_degree, skip_close);

    for (int i=num_iters; i-->0; )
    {
        uint old_tri_ind;
        ControlPoint old_ctrl_pt;
        if (i-1 >= 0) {
            old_tri_ind = tv_load_i(tri_collection, ray_ind + uint(i-1) * ray_origins.sizes[0]);
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, model, origin, direction, sh_degree, skip_close);
        } else {
            old_ctrl_pt.t = 0;
            old_ctrl_pt.dirac = float4(0.f, 0.f, 0.f, 0.f);
        }

        SplineState old_dual_state = inverse_update_dual(dual_state, ctrl_pt, old_ctrl_pt, tmin, tmax);

        let old_deriv_state = run_update(
                old_dual_state,
                old_ctrl_pt,
                ctrl_pt,
                (uint)floor(tri_ind / tri_per_g),
                mod(tri_ind, tri_per_g),
                ray_ind,
                deriv_state,
                origin, direction, tmin, tmax, sh_degree, max_prim_size,
                wct, inv_wct, model);
        tv_atomic_add_i(touch_count, (uint)floor(tri_ind / tri_per_g), 1);

        tri_ind = old_tri_ind;
        dual_state = old_dual_state;
        ctrl_pt = old_ctrl_pt;
        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }
    tv_store_f(dL_dinital_drgb, ray_ind, 0u, deriv_state.d.drgb.x);
    tv_store_f(dL_dinital_drgb, ray_ind, 1u, deriv_state.d.drgb.y);
    tv_store_f(dL_dinital_drgb, ray_ind, 2u, deriv_state.d.drgb.z);
    tv_store_f(dL_dinital_drgb, ray_ind, 3u, deriv_state.d.drgb.w);
}

[Differentiable]
float4 mix_drgb(float density, float3 color) {
    return {density, density*color.x, density*color.y, density*color.z};
}

[CUDAKernel]
void backwards_initial_drgb_kernel(
    TensorView ray_origins,
    TensorView ray_directions,
    DualModel model,
    TensorView initial_drgb,
    TensorView initial_inds,
    TensorView dL_dinital_drgb,
    TensorView touch_count,
    float tmin)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= initial_inds.sizes[0] || thread_j >= ray_directions.sizes[0])
        return;

    uint prim_ind = tv_load_i(initial_inds, thread_i);
    uint ray_ind = thread_j;

    float3 mean = get_float3(model.means, prim_ind);
    float4 quat = get_float4(model.quats, prim_ind);
    float3 scales = get_float3(model.scales, prim_ind);
    float3 rayd = get_float3(ray_directions, ray_ind);
    float3 rayo = get_float3(ray_origins, 0u) + tmin * rayd;
    float3 clip_scale = max(scales, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));

    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);
    if (length(Trayo) <= 1) {
        let density = tv_load_f(model.densities, prim_ind);
        uint sh_degree = 0;
        Features feat = get_feats(model.features, prim_ind, sh_degree);
        float3 color = eval_color(rayd, feat, sh_degree);
        var deriv_color = diffPair(color, {});
        var deriv_density = diffPair(density, {});
        float4 vdL_dinital_drgb = get_float4(dL_dinital_drgb, ray_ind);
        bwd_diff(mix_drgb)(deriv_density, deriv_color, vdL_dinital_drgb);

        tv_atomic_add_f(model.dL_ddensities, prim_ind, deriv_density.d);

        var deriv_direction = diffPair(rayd, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        float3 dfeat = float3(
            d_feat.d.f0.x,
            d_feat.d.f0.y,
            d_feat.d.f0.z
        );
        atomic_add_float3_feat(model.dL_dfeatures, prim_ind, 0u, dfeat);
    }
}
