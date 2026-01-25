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
//   hit_id = (hit_type << HIT_ID_TYPE_SHIFT) | prim_index
//   where hit_type: 1 = entry, 0 = exit
// This encoding ensures exit sorts before entry at same t
// =============================================================================

#define HIT_TYPE_EXIT 0
#define HIT_TYPE_ENTRY 1

static const uint HIT_ID_TYPE_SHIFT = 28;
static const uint HIT_ID_PRIM_MASK = (1u << HIT_ID_TYPE_SHIFT) - 1;

uint get_prim_idx(uint hit_id) { return hit_id & HIT_ID_PRIM_MASK; }
uint get_hit_type(uint hit_id) { return hit_id >> HIT_ID_TYPE_SHIFT; }

import tensor_utils;
import ellipsoid_intersect;
import volume_integrator;
import spherical_harmonics;

// =============================================================================
// Parameters
// =============================================================================

/// Container for primitive parameters and their gradient outputs.
struct Params {
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
///
/// Uses intersect_clamped with t_clamp = tmin to handle rays starting inside primitives.
/// The clamping ensures forward/backward consistency and naturally zeros geometry gradients
/// for virtual entry hits.
DifferentialPair<IntegratorState>
run_update(in IntegratorState prev_state,
           in SamplePoint prev_sample,
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
           inout Params params)
{
    var prev_deriv_state = diffPair(prev_state, {});
    var deriv_sample = diffPair(sample, {});

    bwd_diff(integrate_step)(prev_deriv_state, deriv_sample, tmin, tmax, max_prim_size, deriv_state.d);

    // Propagate gradients to primitive parameters
    let mean = get_float3(params.means, prim_index);
    let scale = get_float3(params.scales, prim_index);
    let quat = get_float4(params.quats, prim_index);
    let density = params.densities[prim_index];
    Features feat = get_feats(params.features, prim_index, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    var deriv_origin = diffPair(origin, {});
    var deriv_direction = diffPair(direction, {});
    var deriv_scales = diffPair(scale, {});
    var deriv_mean = diffPair(mean, {});
    var deriv_quat = diffPair(quat, {});
    var deriv_color = diffPair(color, {});
    var deriv_density = diffPair(density, {});

    // Use t_clamp = tmin to match forward pass behavior for rays starting inside primitives
    float t_clamp = tmin;
    bwd_diff(intersect_clamped)(deriv_origin, deriv_direction,
        deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density, hit_type, t_clamp, deriv_sample.d);

    atomic_add_float3(params.dL_dmeans, prim_index, deriv_mean.d);
    atomic_add_float3(params.dL_dscales, prim_index, deriv_scales.d);
    atomic_add_float4(params.dL_dquats, prim_index, deriv_quat.d);
    float temp;
    params.dL_ddensities.InterlockedAdd(prim_index, deriv_density.d, temp);

    float3 d_rayd = deriv_direction.d;
    deriv_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
    d_rayd += deriv_direction.d;

    atomic_add_float3(params.dL_dfeatures, prim_index, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        atomic_add_float3(params.dL_dfeatures, prim_index, 1u, d_feat.d.f1);
        atomic_add_float3(params.dL_dfeatures, prim_index, 2u, d_feat.d.f2);
        atomic_add_float3(params.dL_dfeatures, prim_index, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            atomic_add_float3(params.dL_dfeatures, prim_index, 4u, d_feat.d.f4);
            atomic_add_float3(params.dL_dfeatures, prim_index, 5u, d_feat.d.f5);
            atomic_add_float3(params.dL_dfeatures, prim_index, 6u, d_feat.d.f6);
            atomic_add_float3(params.dL_dfeatures, prim_index, 7u, d_feat.d.f7);
            atomic_add_float3(params.dL_dfeatures, prim_index, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                atomic_add_float3(params.dL_dfeatures, prim_index, 9u, d_feat.d.f9);
                atomic_add_float3(params.dL_dfeatures, prim_index, 10u, d_feat.d.f10);
                atomic_add_float3(params.dL_dfeatures, prim_index, 11u, d_feat.d.f11);
                atomic_add_float3(params.dL_dfeatures, prim_index, 12u, d_feat.d.f12);
                atomic_add_float3(params.dL_dfeatures, prim_index, 13u, d_feat.d.f13);
                atomic_add_float3(params.dL_dfeatures, prim_index, 14u, d_feat.d.f14);
                atomic_add_float3(params.dL_dfeatures, prim_index, 15u, d_feat.d.f15);
            }
        }
    }

    atomic_add_float3(params.dL_drayos, ray_ind, deriv_origin.d);
    atomic_add_float3(params.dL_drayds, ray_ind, d_rayd);

    return prev_deriv_state;
}

/// Load sample point from a hit ID with t clamping for rays starting inside primitives.
///
/// The t_clamp parameter ensures consistency between forward (OptiX) and backward passes:
/// - Forward: OptiX reports virtual entry hits at t = tmin for rays starting inside
/// - Backward: intersect_clamped uses max(t_entry, tmin) to match this behavior
///
/// This unified approach handles both normal hits and "inside" hits correctly,
/// with the max operation naturally zeroing gradients for geometry parameters
/// when the ray starts inside a primitive.
SamplePoint load_sample_clamped(in uint hit_id, in Params params, in float3 origin, float3 direction, uint sh_degree, float t_clamp)
{
    let prim_index = get_prim_idx(hit_id);
    let hit_type = get_hit_type(hit_id);

    let mean = get_float3(params.means, prim_index);
    let scale = get_float3(params.scales, prim_index);
    let quat = get_float4(params.quats, prim_index);
    let density = params.densities[prim_index];

    Features feat = get_feats(params.features, prim_index, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    return intersect_clamped(origin, direction,
        scale, mean, quat, color, density, hit_type, t_clamp);
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/// Main backward pass kernel.
///
/// This unified kernel handles all hits including rays starting inside primitives.
/// Virtual entry hits (rays starting inside) are processed the same as normal hits,
/// with intersect_clamped ensuring correct gradient behavior.
[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_delta_contrib,
    TensorView<int> iters,
    TensorView<int> hit_collection,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    Params params,
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
    var state = get_state(last_state, ray_ind);
    let direction = get_float3(ray_directions, ray_ind);
    let origin = get_float3(ray_origins, ray_ind);
    let ray_tmax = tmax[ray_ind];

    var deriv_state = diffPair(state, {});

    // dL_doutputs format: (R, G, B, A, depth)
    RenderOutput.Differential dL_doutput;
    dL_doutput.C = {dL_doutputs[ray_ind, 0], dL_doutputs[ray_ind, 1], dL_doutputs[ray_ind, 2]};
    dL_doutput.alpha = dL_doutputs[ray_ind, 3];
    dL_doutput.depth = dL_doutputs[ray_ind, 4];

    uint num_iters = max(min(iters[ray_ind], max_iters), 0);
    if (iters[ray_ind] >= max_iters-1 || iters[ray_ind] <= 0) return;
    var dtmin = diffPair(tmin, {});
    bwd_diff(extract_color)(deriv_state, dtmin, dL_doutput);

    let feature_size = params.features.size(1);
    let sh_degree = int(sqrt(feature_size))-1;

    // t_clamp = tmin to match forward pass behavior for rays starting inside primitives
    float t_clamp = tmin;

    uint hit_id = hit_collection[ray_ind + max(num_iters-1, 0) * ray_origins.size(0)];
    SamplePoint sample = load_sample_clamped(hit_id, params, origin, direction, sh_degree, t_clamp);

    // Iterate backward through hits
    for (int i=num_iters; i-->0; )
    {
        uint prev_hit_id;
        SamplePoint prev_sample;
        if (i-1 >= 0) {
            prev_hit_id = hit_collection[ray_ind + (i-1) * ray_origins.size(0)];
            prev_sample = load_sample_clamped(prev_hit_id, params, origin, direction, sh_degree, t_clamp);
        } else {
            prev_sample.t = tmin;
            prev_sample.delta_contrib = {0.f, 0.f, 0.f, 0.f};
        }

        IntegratorState prev_state = inverse_update(state, sample, prev_sample, tmin, ray_tmax);

        let prev_deriv_state = run_update(
                prev_state,
                prev_sample,
                sample,
                get_prim_idx(hit_id),
                get_hit_type(hit_id),
                ray_ind,
                deriv_state,
                origin, direction, tmin, ray_tmax, sh_degree, max_prim_size,
                params);
        int itemp;
        prim_hits.InterlockedAdd(get_prim_idx(hit_id), 1, itemp);

        hit_id = prev_hit_id;
        state = prev_state;
        sample = prev_sample;

        deriv_state = diffPair(prev_state, prev_deriv_state.d);
    }
}
