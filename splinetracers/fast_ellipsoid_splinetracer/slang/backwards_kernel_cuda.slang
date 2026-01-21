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

// Backward kernel for spline tracer - Pure CUDA version (no slangtorch/TensorView)
// Uses simple pointer types for all buffer access

#define tri_per_g 2
import spline_machine;
import tri_intersect;
import safe_math;
import sh;

// ============================================================================
// Simple helper functions for raw pointer access
// ============================================================================
float3 load_float3(float* buf, uint ind) {
    return float3(buf[ind * 3], buf[ind * 3 + 1], buf[ind * 3 + 2]);
}

float4 load_float4(float* buf, uint ind) {
    return float4(buf[ind * 4], buf[ind * 4 + 1], buf[ind * 4 + 2], buf[ind * 4 + 3]);
}

float3 load_float3_feat(float* buf, uint ind, uint feat_ind, uint feat_stride) {
    uint base = ind * feat_stride * 3 + feat_ind * 3;
    return float3(buf[base], buf[base + 1], buf[base + 2]);
}

void store_float3(float* buf, uint ind, float3 val) {
    buf[ind * 3] = val.x;
    buf[ind * 3 + 1] = val.y;
    buf[ind * 3 + 2] = val.z;
}

void store_float4(float* buf, uint ind, float4 val) {
    buf[ind * 4] = val.x;
    buf[ind * 4 + 1] = val.y;
    buf[ind * 4 + 2] = val.z;
    buf[ind * 4 + 3] = val.w;
}

// CUDA atomic add for floats
void atomicAddFloat(float* addr, float val) {
    __intrinsic_asm "atomicAdd($0, $1)";
}

void atomic_add_float3(float* buf, uint ind, float3 val) {
    atomicAddFloat(buf + ind * 3, val.x);
    atomicAddFloat(buf + ind * 3 + 1, val.y);
    atomicAddFloat(buf + ind * 3 + 2, val.z);
}

void atomic_add_float4(float* buf, uint ind, float4 val) {
    atomicAddFloat(buf + ind * 4, val.x);
    atomicAddFloat(buf + ind * 4 + 1, val.y);
    atomicAddFloat(buf + ind * 4 + 2, val.z);
    atomicAddFloat(buf + ind * 4 + 3, val.w);
}

void atomic_add_float2(float* buf, uint ind, float2 val) {
    atomicAddFloat(buf + ind * 2, val.x);
    atomicAddFloat(buf + ind * 2 + 1, val.y);
}

void atomic_add_float3_feat(float* buf, uint ind, uint feat_ind, uint feat_stride, float3 val) {
    uint base = ind * feat_stride * 3 + feat_ind * 3;
    atomicAddFloat(buf + base, val.x);
    atomicAddFloat(buf + base + 1, val.y);
    atomicAddFloat(buf + base + 2, val.z);
}

void atomicAddInt(int* addr, int val) {
    __intrinsic_asm "atomicAdd($0, $1)";
}

// ============================================================================
// Load features for SH evaluation
// ============================================================================
Features load_feats(float* features, uint prim_ind, uint sh_degree, uint feat_stride) {
    Features feat;
    feat.f0 = load_float3_feat(features, prim_ind, 0, feat_stride);
    if (sh_degree > 0) {
        feat.f1 = load_float3_feat(features, prim_ind, 1, feat_stride);
        feat.f2 = load_float3_feat(features, prim_ind, 2, feat_stride);
        feat.f3 = load_float3_feat(features, prim_ind, 3, feat_stride);
        if (sh_degree > 1) {
            feat.f4 = load_float3_feat(features, prim_ind, 4, feat_stride);
            feat.f5 = load_float3_feat(features, prim_ind, 5, feat_stride);
            feat.f6 = load_float3_feat(features, prim_ind, 6, feat_stride);
            feat.f7 = load_float3_feat(features, prim_ind, 7, feat_stride);
            feat.f8 = load_float3_feat(features, prim_ind, 8, feat_stride);
            if (sh_degree > 2) {
                feat.f9 = load_float3_feat(features, prim_ind, 9, feat_stride);
                feat.f10 = load_float3_feat(features, prim_ind, 10, feat_stride);
                feat.f11 = load_float3_feat(features, prim_ind, 11, feat_stride);
                feat.f12 = load_float3_feat(features, prim_ind, 12, feat_stride);
                feat.f13 = load_float3_feat(features, prim_ind, 13, feat_stride);
                feat.f14 = load_float3_feat(features, prim_ind, 14, feat_stride);
                feat.f15 = load_float3_feat(features, prim_ind, 15, feat_stride);
            }
        }
    }
    return feat;
}

// ============================================================================
// Load SplineState from flat buffer
// ============================================================================
SplineState load_state(float* buf, uint ind) {
    uint base = ind * 16;
    return SplineState(
        float2(buf[base + 0], buf[base + 1]),     // distortion_parts
        float2(buf[base + 2], buf[base + 3]),     // cum_sum
        float3(buf[base + 4], buf[base + 5], buf[base + 6]),  // padding
        buf[base + 7],                             // t
        float4(buf[base + 8], buf[base + 9], buf[base + 10], buf[base + 11]),  // drgb
        buf[base + 12],                            // logT
        float3(buf[base + 13], buf[base + 14], buf[base + 15])  // C
    );
}

// ============================================================================
// Load control point from model buffers
// ============================================================================
ControlPoint load_ctrl_pt(
    uint tri_ind,
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    float3 origin,
    float3 direction,
    uint sh_degree,
    uint feat_stride
) {
    let prim_ind = (uint)floor(tri_ind / tri_per_g);
    let face_id = tri_ind % tri_per_g;

    let mean = load_float3(means, prim_ind);
    let scale = load_float3(scales, prim_ind);
    let quat = load_float4(quats, prim_ind);
    let density = densities[prim_ind];

    Features feat = load_feats(features, prim_ind, sh_degree, feat_stride);
    float3 color = eval_color(direction, feat, sh_degree);

    bool skip_close = false;
    return safe_intersect(origin, direction, scale, mean, quat, color, density, face_id, skip_close);
}

// ============================================================================
// Projection helpers
// ============================================================================
[Differentiable]
float3 project(in float3 xyz, in float4x4 wct) {
    float4 xyzw = float4(xyz, 1.f);
    let p_view = mul(xyzw, wct);
    float2 pix2d = float2(safe_div(p_view.x, p_view.z), safe_div(p_view.y, p_view.z));
    return float3(pix2d.x, pix2d.y, p_view.z);
}

[Differentiable]
float3 inv_project(in float2 xy, in float dist, float4x4 inv_wvt) {
    let p_hom = float4(xy * dist, dist, 1.f);
    let out = mul(p_hom, inv_wvt);
    return float3(out.x, out.y, out.z);
}

// ============================================================================
// Run one backward update step
// ============================================================================
DifferentialPair<SplineState> run_update(
    in SplineState old_dual_state,
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
    in uint feat_stride,
    in float max_prim_size,
    in float4x4 wct,
    in float4x4 inv_wct,
    // Model buffers (read)
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    // Gradient buffers (write)
    float* dL_dmeans,
    float* dL_dscales,
    float* dL_dquats,
    float* dL_ddensities,
    float* dL_dfeatures,
    float* dL_drayos,
    float* dL_drayds,
    float* dL_dmeans2D
) {
    var old_deriv_state = diffPair(from_dual(old_dual_state, old_ctrl_pt), SplineState.Differential());
    var deriv_ctrl_pt = diffPair(ctrl_pt, ControlPoint.Differential());
    bool skip_close = false;

    bwd_diff(update)(old_deriv_state, deriv_ctrl_pt, tmin, tmax, max_prim_size, deriv_state.d);

    // Load current primitive data
    let mean = load_float3(means, prim_ind);
    let scale = load_float3(scales, prim_ind);
    let quat = load_float4(quats, prim_ind);
    let density = densities[prim_ind];
    Features feat = load_feats(features, prim_ind, sh_degree, feat_stride);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin = diffPair(origin, float3(0));
    var deriv_direction = diffPair(direction, float3(0));
    var deriv_scales = diffPair(scale, float3(0));
    var deriv_mean = diffPair(mean, float3(0));
    var deriv_quat = diffPair(quat, float4(0));
    var deriv_color = diffPair(color, float3(0));
    var deriv_density = diffPair(density, 0.f);

    bwd_diff(safe_intersect)(
        deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density,
        face_id, skip_close, deriv_ctrl_pt.d
    );

    // Accumulate gradients atomically
    atomic_add_float3(dL_dmeans, prim_ind, deriv_mean.d);
    atomic_add_float3(dL_dscales, prim_ind, deriv_scales.d);
    atomic_add_float4(dL_dquats, prim_ind, deriv_quat.d);
    atomicAddFloat(dL_ddensities + prim_ind, deriv_density.d);

    // Backward through color evaluation
    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, float3(0));
    var d_feat = diffPair(feat, Features.Differential());
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    // Accumulate feature gradients
    atomic_add_float3_feat(dL_dfeatures, prim_ind, 0u, feat_stride, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3_feat(dL_dfeatures, prim_ind, 1u, feat_stride, d_feat.d.f1);
        atomic_add_float3_feat(dL_dfeatures, prim_ind, 2u, feat_stride, d_feat.d.f2);
        atomic_add_float3_feat(dL_dfeatures, prim_ind, 3u, feat_stride, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3_feat(dL_dfeatures, prim_ind, 4u, feat_stride, d_feat.d.f4);
            atomic_add_float3_feat(dL_dfeatures, prim_ind, 5u, feat_stride, d_feat.d.f5);
            atomic_add_float3_feat(dL_dfeatures, prim_ind, 6u, feat_stride, d_feat.d.f6);
            atomic_add_float3_feat(dL_dfeatures, prim_ind, 7u, feat_stride, d_feat.d.f7);
            atomic_add_float3_feat(dL_dfeatures, prim_ind, 8u, feat_stride, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 9u, feat_stride, d_feat.d.f9);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 10u, feat_stride, d_feat.d.f10);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 11u, feat_stride, d_feat.d.f11);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 12u, feat_stride, d_feat.d.f12);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 13u, feat_stride, d_feat.d.f13);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 14u, feat_stride, d_feat.d.f14);
                atomic_add_float3_feat(dL_dfeatures, prim_ind, 15u, feat_stride, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(dL_drayds, ray_ind, d_rayd);

    // 2D mean gradient
    float3 xyd = project(mean, wct);
    var d_xy = diffPair(float2(xyd.x, xyd.y), float2(0));
    var d_dist = diffPair(xyd.z, 0.f);
    var d_inv_wct = diffPair(inv_wct, float4x4(0));
    bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
    atomic_add_float2(dL_dmeans2D, prim_ind, d_xy.d);

    return old_deriv_state;
}

[Differentiable]
float4 mix_drgb(float density, float3 color) {
    return float4(density, density * color.x, density * color.y, density * color.z);
}

// ============================================================================
// Main backward kernel - CUDA entry point
// ============================================================================
[CudaKernel]
void backwards_kernel(
    // Forward saved data
    float* last_state,          // [num_rays, 16]
    int* iters,                 // [num_rays]
    int* tri_collection,        // [num_rays * max_iters]
    // Ray data
    float* ray_origins,         // [num_rays, 3]
    float* ray_directions,      // [num_rays, 3]
    // Model (read-only)
    float* means,               // [num_prims, 3]
    float* scales,              // [num_prims, 3]
    float* quats,               // [num_prims, 4]
    float* densities,           // [num_prims]
    float* features,            // [num_prims, feat_stride, 3]
    // Gradients (output)
    float* dL_dmeans,
    float* dL_dscales,
    float* dL_dquats,
    float* dL_ddensities,
    float* dL_dfeatures,
    float* dL_drayos,
    float* dL_drayds,
    float* dL_dmeans2D,
    float* dL_dinitial_drgb,
    int* touch_count,
    // Gradient input
    float* dL_doutputs,         // [num_rays, 5]
    // WCT matrices
    float* wcts,                // [num_wcts, 4, 4]
    // Parameters
    float tmin,
    float tmax,
    float max_prim_size,
    uint max_iters,
    uint num_rays,
    uint num_prims,
    uint feature_size,
    uint num_wcts
) {
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;

    if (ray_ind >= num_rays) return;

    int num_iters_ray = iters[ray_ind];
    if (num_iters_ray >= (int)max_iters - 1 || num_iters_ray <= 0) return;

    let sh_degree = (uint)sqrt((float)feature_size) - 1;
    let feat_stride = feature_size;

    // Load ray
    let direction = load_float3(ray_directions, ray_ind);
    let origin = load_float3(ray_origins, ray_ind) + tmin * direction;

    // Load last state
    var dual_state = load_state(last_state, ray_ind);

    // Initialize gradient state
    var deriv_state = diffPair(dual_state, SplineState.Differential());

    // Load gradient input
    let dL_dC = load_float4(dL_doutputs, ray_ind);
    let dL_ddistortion_loss = dL_doutputs[ray_ind * 5 + 4];

    SplineOutput.Differential dL_doutput;
    dL_doutput.C = float3(dL_dC.x, dL_dC.y, dL_dC.z);
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    var dtmin = diffPair(tmin, 0.f);
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    // Load WCT matrix
    let wct_ind = (ray_ind < num_wcts) ? ray_ind : 0;
    float4x4 wct = float4x4(
        wcts[wct_ind * 16 + 0], wcts[wct_ind * 16 + 1], wcts[wct_ind * 16 + 2], wcts[wct_ind * 16 + 3],
        wcts[wct_ind * 16 + 4], wcts[wct_ind * 16 + 5], wcts[wct_ind * 16 + 6], wcts[wct_ind * 16 + 7],
        wcts[wct_ind * 16 + 8], wcts[wct_ind * 16 + 9], wcts[wct_ind * 16 + 10], wcts[wct_ind * 16 + 11],
        wcts[wct_ind * 16 + 12], wcts[wct_ind * 16 + 13], wcts[wct_ind * 16 + 14], wcts[wct_ind * 16 + 15]
    );
    float4x4 inv_wct = inverse(wct);

    // Load last control point
    uint tri_ind = tri_collection[ray_ind + max(num_iters_ray - 1, 0) * num_rays];
    ControlPoint ctrl_pt = load_ctrl_pt(tri_ind, means, scales, quats, densities, features,
                                        origin, direction, sh_degree, feat_stride);

    // Backward loop
    for (int i = num_iters_ray; i-- > 0; ) {
        uint old_tri_ind;
        ControlPoint old_ctrl_pt;

        if (i - 1 >= 0) {
            old_tri_ind = tri_collection[ray_ind + (i - 1) * num_rays];
            old_ctrl_pt = load_ctrl_pt(old_tri_ind, means, scales, quats, densities, features,
                                       origin, direction, sh_degree, feat_stride);
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
            tri_ind % tri_per_g,
            ray_ind,
            deriv_state,
            origin, direction, tmin, tmax, sh_degree, feat_stride, max_prim_size,
            wct, inv_wct,
            means, scales, quats, densities, features,
            dL_dmeans, dL_dscales, dL_dquats, dL_ddensities, dL_dfeatures,
            dL_drayos, dL_drayds, dL_dmeans2D
        );

        atomicAddInt(touch_count + (uint)floor(tri_ind / tri_per_g), 1);

        tri_ind = old_tri_ind;
        dual_state = old_dual_state;
        ctrl_pt = old_ctrl_pt;
        deriv_state = diffPair(old_dual_state, old_deriv_state.d);
    }

    // Store dL_dinitial_drgb
    store_float4(dL_dinitial_drgb, ray_ind, deriv_state.d.drgb);
}

// ============================================================================
// Backward kernel for initial drgb (primitives containing ray origin)
// ============================================================================
[CudaKernel]
void backwards_initial_drgb_kernel(
    float* ray_origins,
    float* ray_directions,
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    float* dL_ddensities,
    float* dL_dfeatures,
    int* initial_inds,
    float* dL_dinitial_drgb,
    int* touch_count,
    float tmin,
    uint num_rays,
    uint num_initial_inds,
    uint feature_size
) {
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;

    if (thread_i >= num_initial_inds || thread_j >= num_rays) return;

    uint prim_ind = initial_inds[thread_i];
    uint ray_ind = thread_j;

    float3 mean = load_float3(means, prim_ind);
    float4 quat = load_float4(quats, prim_ind);
    float3 scale = load_float3(scales, prim_ind);
    float3 rayd = load_float3(ray_directions, ray_ind);
    float3 rayo = load_float3(ray_origins, 0) + tmin * rayd;
    float3 clip_scale = max(scale, float3(1e-8));

    let R = quat2mat(safe_div(quat, length(quat)));
    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);

    if (length(Trayo) <= 1) {
        let density = densities[prim_ind];
        uint sh_degree = 0;
        uint feat_stride = feature_size;
        Features feat = load_feats(features, prim_ind, sh_degree, feat_stride);
        float3 color = eval_color(rayd, feat, sh_degree);

        var deriv_color = diffPair(color, float3(0));
        var deriv_density = diffPair(density, 0.f);
        float4 vdL_dinitial_drgb = load_float4(dL_dinitial_drgb, ray_ind);

        bwd_diff(mix_drgb)(deriv_density, deriv_color, vdL_dinitial_drgb);

        atomicAddFloat(dL_ddensities + prim_ind, deriv_density.d);

        var deriv_direction = diffPair(rayd, float3(0));
        var d_feat = diffPair(feat, Features.Differential());
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);

        atomic_add_float3_feat(dL_dfeatures, prim_ind, 0, feat_stride, d_feat.d.f0);
    }
}
