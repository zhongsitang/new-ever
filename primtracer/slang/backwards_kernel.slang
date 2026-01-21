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
//
// This module implements the backward pass for computing gradients through
// the volume rendering pipeline. It processes the recorded hit sequence in
// reverse order to propagate gradients to primitive parameters.

import spline_machine;
import ellipsoid;
import safe_math;
import sh;

// =============================================================================
// Tensor Access Utilities
// =============================================================================

float4 get_float4(TensorView<float> view, uint idx) {
    return {view[idx, 0], view[idx, 1], view[idx, 2], view[idx, 3]};
}

void atomic_add_float3(TensorView<float> view, uint idx, uint feat_idx, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(idx, feat_idx, 0u), val.x, temp);
    view.InterlockedAdd(uint3(idx, feat_idx, 1u), val.y, temp);
    view.InterlockedAdd(uint3(idx, feat_idx, 2u), val.z, temp);
}

void atomic_add_float3(TensorView<float> view, uint idx, float3 val) {
    float temp;
    view.InterlockedAdd(uint2(idx, 0u), val.x, temp);
    view.InterlockedAdd(uint2(idx, 1u), val.y, temp);
    view.InterlockedAdd(uint2(idx, 2u), val.z, temp);
}

void atomic_add_float4(TensorView<float> view, uint idx, float4 val) {
    float temp;
    view.InterlockedAdd(uint2(idx, 0u), val.x, temp);
    view.InterlockedAdd(uint2(idx, 1u), val.y, temp);
    view.InterlockedAdd(uint2(idx, 2u), val.z, temp);
    view.InterlockedAdd(uint2(idx, 3u), val.w, temp);
}

// =============================================================================
// Integration State Serialization
// =============================================================================

IntegrationState get_state(TensorView<float> view, uint idx) {
    return {
        float2(view[idx, 0], view[idx, 1]),   // distortion_parts
        float2(view[idx, 2], view[idx, 3]),   // cum_sum
        float3(view[idx, 4], view[idx, 5], view[idx, 6]),  // depth_accum
        view[idx, 7],                          // t
        float4(view[idx, 8], view[idx, 9], view[idx, 10], view[idx, 11]),  // sample
        view[idx, 12],                         // log_transmittance
        float3(view[idx, 13], view[idx, 14], view[idx, 15]),  // color
    };
}

// =============================================================================
// Dual Model - Forward values and gradient accumulators
// =============================================================================

struct DualModel {
    // Forward values (read-only)
    TensorView<float> means;
    TensorView<float> scales;
    TensorView<float> quats;
    TensorView<float> densities;
    TensorView<float> features;

    // Gradient accumulators (write via atomic add)
    TensorView<float> dL_dmeans;
    TensorView<float> dL_dscales;
    TensorView<float> dL_dquats;
    TensorView<float> dL_ddensities;
    TensorView<float> dL_dfeatures;
    TensorView<float> dL_dray_origins;
    TensorView<float> dL_dray_directions;
};

// =============================================================================
// Backward Update Step
// =============================================================================
// Computes gradients for a single integration step and propagates to primitive parameters

DifferentialPair<IntegrationState>
backward_update_step(
    in IntegrationState prev_dual_state,
    in Sample prev_sample,
    in Sample current_sample,
    in uint prim_idx,
    in uint hit_type,
    in uint ray_idx,
    in DifferentialPair<IntegrationState> d_state,
    in float3 origin,
    in float3 direction,
    in float t_near,
    in float t_far,
    in uint sh_degree,
    in float max_prim_size,
    inout DualModel model)
{
    // Backward through update() to get gradient w.r.t. previous state and sample
    var d_prev_state = diffPair(from_dual(prev_dual_state, prev_sample), {});
    var d_sample = diffPair(current_sample, {});
    bool skip_close = false;

    bwd_diff(update)(d_prev_state, d_sample, t_near, t_far, max_prim_size, d_state.d);

    // Load primitive parameters
    let mean = get_float3(model.means, prim_idx);
    let scale = get_float3(model.scales, prim_idx);
    let quat = get_float4(model.quats, prim_idx);
    let density = model.densities[prim_idx];
    Features feat = get_feats(model.features, prim_idx, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    // Setup differential pairs for primitive parameters
    var d_origin = diffPair(origin, {});
    var d_direction = diffPair(direction, {});
    var d_scale = diffPair(scale, {});
    var d_mean = diffPair(mean, {});
    var d_quat = diffPair(quat, {});
    var d_color = diffPair(color, {});
    var d_density = diffPair(density, {});

    // Backward through intersection computation
    bwd_diff(safe_intersect)(d_origin, d_direction,
        d_scale, d_mean, d_quat, d_color, d_density, hit_type, skip_close, d_sample.d);

    // Accumulate gradients to primitive parameters
    atomic_add_float3(model.dL_dmeans, prim_idx, d_mean.d);
    atomic_add_float3(model.dL_dscales, prim_idx, d_scale.d);
    atomic_add_float4(model.dL_dquats, prim_idx, d_quat.d);
    float temp;
    model.dL_ddensities.InterlockedAdd(prim_idx, d_density.d, temp);

    // Backward through SH color evaluation
    float3 d_ray_direction = d_direction.d;
    d_direction = diffPair(direction, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(d_direction, d_feat, sh_degree, d_color.d);
    d_ray_direction += d_direction.d;

    // Accumulate SH coefficient gradients
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

    // Accumulate ray gradients
    atomic_add_float3(model.dL_dray_origins, ray_idx, d_origin.d);
    atomic_add_float3(model.dL_dray_directions, ray_idx, d_ray_direction);

    return d_prev_state;
}

// =============================================================================
// Sample Loading from Hit Record
// =============================================================================

Sample load_sample_from_hit(
    in uint tri_idx,
    in DualModel model,
    in float3 origin,
    in float3 direction,
    in uint sh_degree,
    in bool skip_close)
{
    let prim_idx = (uint)floor(tri_idx / tri_per_g);
    let hit_type = mod(tri_idx, tri_per_g);

    let mean = get_float3(model.means, prim_idx);
    let scale = get_float3(model.scales, prim_idx);
    let quat = get_float4(model.quats, prim_idx);
    let density = model.densities[prim_idx];

    Features feat = get_feats(model.features, prim_idx, sh_degree);
    float3 color = eval_color(direction, feat, sh_degree);

    return safe_intersect(origin, direction, scale, mean, quat, color, density, hit_type, skip_close);
}

// =============================================================================
// Main Backward Kernel
// =============================================================================

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_kernel(
    TensorView<float> last_state,
    TensorView<float> last_sample,
    TensorView<int> iterations,
    TensorView<int> hit_sequence,

    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_sample,
    TensorView<float> dL_d_initial_sample,
    TensorView<int32_t> touch_count,

    TensorView<float> dL_doutputs,

    float t_near,
    float t_far,
    float max_prim_size,
    uint max_iterations)
{
    uint3 dispatch_idx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_idx = dispatch_idx.x;
    if (ray_idx >= ray_origins.size(0)) {
        return;
    }

    var dual_state = get_state(last_state, ray_idx);
    let direction = get_float3(ray_directions, ray_idx);
    let origin = get_float3(ray_origins, ray_idx) + t_near * direction;
    bool skip_close = false;

    var d_state = diffPair(dual_state, {});

    // Setup output gradient
    let dL_dcolor = get_float4(dL_doutputs, ray_idx);
    let dL_d_distortion = dL_doutputs[ray_idx, 4];
    RenderOutput.Differential dL_doutput;
    dL_doutput.color = {dL_dcolor.x, dL_dcolor.y, dL_dcolor.z};
    dL_doutput.depth = dL_dcolor.w;
    dL_doutput.distortion_loss = dL_d_distortion;

    // Early exit for boundary cases
    uint num_iterations = max(min(iterations[ray_idx], max_iterations), 0);
    if (iterations[ray_idx] >= max_iterations - 1 || iterations[ray_idx] <= 0) {
        return;
    }

    // Backward through color extraction
    var d_t_near = diffPair(t_near, {});
    var d_t_far = diffPair(t_far, {});
    bwd_diff(extract_color)(d_state, d_t_near, dL_doutput);

    let feature_size = model.features.size(1);
    let sh_degree = int(sqrt(feature_size)) - 1;

    // Load the last sample in the sequence
    uint tri_idx = hit_sequence[ray_idx + max(num_iterations - 1, 0) * ray_origins.size(0)];
    Sample current_sample = load_sample_from_hit(tri_idx, model, origin, direction, sh_degree, skip_close);

    // Process hits in reverse order
    for (int i = num_iterations; i-- > 0; )
    {
        uint prev_tri_idx;
        Sample prev_sample;

        if (i - 1 >= 0) {
            prev_tri_idx = hit_sequence[ray_idx + (i - 1) * ray_origins.size(0)];
            prev_sample = load_sample_from_hit(prev_tri_idx, model, origin, direction, sh_degree, skip_close);
        } else {
            // Initial sample (before any hits)
            prev_sample.t = 0;
            prev_sample.value = {0.f, 0.f, 0.f, 0.f};
        }

        // Compute previous dual state
        IntegrationState prev_dual_state = inverse_update_dual(dual_state, current_sample, prev_sample, t_near, t_far);

        // Run backward update
        let d_prev_state = backward_update_step(
            prev_dual_state,
            prev_sample,
            current_sample,
            (uint)floor(tri_idx / tri_per_g),
            mod(tri_idx, tri_per_g),
            ray_idx,
            d_state,
            origin, direction, t_near, t_far, sh_degree, max_prim_size,
            model);

        // Update touch count
        int itemp;
        touch_count.InterlockedAdd((uint)floor(tri_idx / tri_per_g), 1, itemp);

        // Move to previous step
        tri_idx = prev_tri_idx;
        dual_state = prev_dual_state;
        current_sample = prev_sample;
        d_state = diffPair(prev_dual_state, d_prev_state.d);
    }

    // Output gradient for initial sample
    dL_d_initial_sample[ray_idx, 0u] = d_state.d.sample.x;
    dL_d_initial_sample[ray_idx, 1u] = d_state.d.sample.y;
    dL_d_initial_sample[ray_idx, 2u] = d_state.d.sample.z;
    dL_d_initial_sample[ray_idx, 3u] = d_state.d.sample.w;
}

// =============================================================================
// Initial Sample Gradient Kernel
// =============================================================================

[Differentiable]
float4 compute_density_rgb(float density, float3 color) {
    return {density, density * color.x, density * color.y, density * color.z};
}

[AutoPyBindCUDA]
[CUDAKernel]
void backwards_initial_drgb_kernel(
    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    DualModel model,
    TensorView<float> initial_sample,
    TensorView<int32_t> initial_hit_indices,
    TensorView<float> dL_d_initial_sample,
    TensorView<int32_t> touch_count,
    float t_near)
{
    uint thread_j = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint thread_i = cudaThreadIdx().y + cudaBlockIdx().y * cudaBlockDim().y;
    if (thread_i >= initial_hit_indices.size(0) || thread_j >= ray_directions.size(0)) {
        return;
    }

    uint prim_idx = initial_hit_indices[thread_i];
    uint ray_idx = thread_j;

    float3 mean = get_float3(model.means, prim_idx);
    float4 quat = get_float4(model.quats, prim_idx);
    float3 scale = get_float3(model.scales, prim_idx);
    float3 ray_dir = get_float3(ray_directions, ray_idx);
    float3 ray_origin = get_float3(ray_origins, 0) + t_near * ray_dir;
    float3 safe_scale = max(scale, 1e-8);

    let R = quat2mat(safe_div(quat, length(quat)));
    let transformed_origin = safe_div(mul(ray_origin - mean, R), safe_scale);

    // Check if ray origin is inside ellipsoid
    if (length(transformed_origin) <= 1) {
        float temp;
        let density = model.densities[prim_idx];
        int sh_degree = 0;
        Features feat = get_feats(model.features, prim_idx, sh_degree);
        float3 color = eval_color(ray_dir, feat, sh_degree);

        var d_color = diffPair(color, {});
        var d_density = diffPair(density, {});
        float4 grad_initial = get_float4(dL_d_initial_sample, ray_idx);
        bwd_diff(compute_density_rgb)(d_density, d_color, grad_initial);

        model.dL_ddensities.InterlockedAdd(prim_idx, d_density.d, temp);

        var d_direction = diffPair(ray_dir, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(d_direction, d_feat, sh_degree, d_color.d);

        float3 d_f0 = {d_feat.d.f0.x, d_feat.d.f0.y, d_feat.d.f0.z};
        atomic_add_float3(model.dL_dfeatures, prim_idx, 0, d_f0);
    }
}
