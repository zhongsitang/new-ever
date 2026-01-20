// GaussianRT - Backward Pass Kernel
// Differentiable volume rendering backward propagation
// Apache License 2.0
// Compiled to CUDA via slangc

#define tri_per_g 2
import volume;
import ellipsoid;
import safe_math;
import sh;

// Helper functions for array access
float3 load_float3(float* data, uint idx) {
    uint base = idx * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

float4 load_float4(float* data, uint idx) {
    uint base = idx * 4;
    return float4(data[base], data[base + 1], data[base + 2], data[base + 3]);
}

float3 load_float3_2d(float* data, uint idx, uint feat_idx, uint feat_stride) {
    uint base = idx * feat_stride + feat_idx * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

// Atomic add helpers
void atomic_add_float3(float* data, uint idx, float3 val) {
    uint base = idx * 3;
    atomicAdd(&data[base], val.x);
    atomicAdd(&data[base + 1], val.y);
    atomicAdd(&data[base + 2], val.z);
}

void atomic_add_float4(float* data, uint idx, float4 val) {
    uint base = idx * 4;
    atomicAdd(&data[base], val.x);
    atomicAdd(&data[base + 1], val.y);
    atomicAdd(&data[base + 2], val.z);
    atomicAdd(&data[base + 3], val.w);
}

void atomic_add_float2(float* data, uint idx, float2 val) {
    uint base = idx * 2;
    atomicAdd(&data[base], val.x);
    atomicAdd(&data[base + 1], val.y);
}

void atomic_add_float3_2d(float* data, uint idx, uint feat_idx, uint feat_stride, float3 val) {
    uint base = idx * feat_stride + feat_idx * 3;
    atomicAdd(&data[base], val.x);
    atomicAdd(&data[base + 1], val.y);
    atomicAdd(&data[base + 2], val.z);
}

VolumeState load_state(float* data, uint idx) {
    uint base = idx * 16;
    VolumeState state;
    state.distortion_parts = float2(data[base], data[base + 1]);
    state.cum_sum = float2(data[base + 2], data[base + 3]);
    state.padding = float3(data[base + 4], data[base + 5], data[base + 6]);
    state.t = data[base + 7];
    state.drgb = float4(data[base + 8], data[base + 9], data[base + 10], data[base + 11]);
    state.logT = data[base + 12];
    state.C = float3(data[base + 13], data[base + 14], data[base + 15]);
    return state;
}

Features load_features(float* data, uint prim_idx, uint sh_degree, uint feat_stride) {
    Features feat;
    feat.f0 = load_float3_2d(data, prim_idx, 0, feat_stride);
    if (sh_degree > 0) {
        feat.f1 = load_float3_2d(data, prim_idx, 1, feat_stride);
        feat.f2 = load_float3_2d(data, prim_idx, 2, feat_stride);
        feat.f3 = load_float3_2d(data, prim_idx, 3, feat_stride);
        if (sh_degree > 1) {
            feat.f4 = load_float3_2d(data, prim_idx, 4, feat_stride);
            feat.f5 = load_float3_2d(data, prim_idx, 5, feat_stride);
            feat.f6 = load_float3_2d(data, prim_idx, 6, feat_stride);
            feat.f7 = load_float3_2d(data, prim_idx, 7, feat_stride);
            feat.f8 = load_float3_2d(data, prim_idx, 8, feat_stride);
            if (sh_degree > 2) {
                feat.f9 = load_float3_2d(data, prim_idx, 9, feat_stride);
                feat.f10 = load_float3_2d(data, prim_idx, 10, feat_stride);
                feat.f11 = load_float3_2d(data, prim_idx, 11, feat_stride);
                feat.f12 = load_float3_2d(data, prim_idx, 12, feat_stride);
                feat.f13 = load_float3_2d(data, prim_idx, 13, feat_stride);
                feat.f14 = load_float3_2d(data, prim_idx, 14, feat_stride);
                feat.f15 = load_float3_2d(data, prim_idx, 15, feat_stride);
            }
        }
    }
    return feat;
}

[Differentiable]
float3 project(float3 xyz, float4x4 wct) {
    float4 xyzw = float4(xyz, 1.f);
    let p_view = mul(xyzw, wct);
    float2 pix2d = float2(safe_div(p_view.x, p_view.z), safe_div(p_view.y, p_view.z));
    return float3(pix2d.x, pix2d.y, p_view.z);
}

[Differentiable]
float3 inv_project(float2 xy, float dist, float4x4 inv_wvt) {
    let p_hom = float4(xy * dist, dist, 1.f);
    let out = mul(p_hom, inv_wvt);
    return float3(out.x, out.y, out.z);
}

VolumeSample load_volume_sample(
    uint tri_ind,
    float3 origin,
    float3 direction,
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    uint sh_degree,
    uint feat_stride,
    bool skip_close
) {
    let prim_ind = tri_ind / tri_per_g;
    let face_id = tri_ind % tri_per_g;

    let mean = load_float3(means, prim_ind);
    let scale = load_float3(scales, prim_ind);
    let quat = load_float4(quats, prim_ind);
    let density = densities[prim_ind];

    Features feat = load_features(features, prim_ind, sh_degree, feat_stride);
    float3 color = eval_color(direction, feat, sh_degree);

    return safe_intersect(origin, direction, scale, mean, quat, color, density, face_id, skip_close);
}

// Main backward kernel
[CUDAKernel]
void backwards_kernel(
    float* last_state,
    float* last_dirac,
    int* iters,
    int* tri_collection,
    float* ray_origins,
    float* ray_directions,
    // Model data
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    // Gradient outputs
    float* dL_dmeans,
    float* dL_dscales,
    float* dL_dquats,
    float* dL_ddensities,
    float* dL_dfeatures,
    float* dL_drayos,
    float* dL_drayds,
    float* dL_dmeans2D,
    // Other data
    float* initial_drgb,
    float* dL_dinitial_drgb,
    int* touch_count,
    float* dL_doutputs,
    float* wcts,
    // Parameters
    float tmin,
    float tmax,
    float max_prim_size,
    uint max_iters,
    uint num_rays,
    uint num_prims,
    uint feature_size
) {
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_ind = dispatchIdx.x;
    if (ray_ind >= num_rays) {
        return;
    }

    int num_iters_val = iters[ray_ind];
    if (num_iters_val >= (int)max_iters - 1 || num_iters_val <= 0) {
        return;
    }

    let sh_degree = (uint)(sqrt((float)feature_size) - 1);
    let feat_stride = feature_size * 3;

    var dual_state = load_state(last_state, ray_ind);
    let direction = load_float3(ray_directions, ray_ind);
    let origin = load_float3(ray_origins, ray_ind) + tmin * direction;
    bool skip_close = false;

    var deriv_state = diffPair(dual_state, {});

    // Load output gradients
    uint out_base = ray_ind * 5;
    let dL_dC = float4(dL_doutputs[out_base], dL_doutputs[out_base + 1],
                       dL_doutputs[out_base + 2], dL_doutputs[out_base + 3]);
    let dL_ddistortion_loss = dL_doutputs[out_base + 4];

    VolumeOutput.Differential dL_doutput;
    dL_doutput.C = float3(dL_dC.x, dL_dC.y, dL_dC.z);
    dL_doutput.depth = dL_dC.w;
    dL_doutput.distortion_loss = dL_ddistortion_loss;

    var dtmin = diffPair(tmin, {});
    bwd_diff(extract_volume_output)(deriv_state, dtmin, dL_doutput);

    // Load world-to-clip transform
    let wct_ind = (ray_ind < 1) ? 0 : ray_ind;  // Use first wct if only one provided
    uint wct_base = wct_ind * 16;
    float4x4 wct = float4x4(
        wcts[wct_base + 0], wcts[wct_base + 1], wcts[wct_base + 2], wcts[wct_base + 3],
        wcts[wct_base + 4], wcts[wct_base + 5], wcts[wct_base + 6], wcts[wct_base + 7],
        wcts[wct_base + 8], wcts[wct_base + 9], wcts[wct_base + 10], wcts[wct_base + 11],
        wcts[wct_base + 12], wcts[wct_base + 13], wcts[wct_base + 14], wcts[wct_base + 15]
    );
    float4x4 inv_wct = inverse(wct);

    uint num_iters = (uint)max(min(num_iters_val, (int)max_iters), 0);
    uint tri_ind = tri_collection[ray_ind + max(num_iters - 1, 0u) * num_rays];

    VolumeSample sample = load_volume_sample(tri_ind, origin, direction,
        means, scales, quats, densities, features, sh_degree, feat_stride, skip_close);

    // Backward loop
    for (int i = (int)num_iters; i-- > 0; ) {
        uint old_tri_ind;
        VolumeSample previous_sample;

        if (i - 1 >= 0) {
            old_tri_ind = tri_collection[ray_ind + (i - 1) * num_rays];
            previous_sample = load_volume_sample(old_tri_ind, origin, direction,
                means, scales, quats, densities, features, sh_degree, feat_stride, skip_close);
        } else {
            previous_sample.t = 0;
            previous_sample.dirac = float4(0.f, 0.f, 0.f, 0.f);
        }

        VolumeState previous_dual_state = inverse_integrate_dual(dual_state, sample, previous_sample, tmin, tmax);

        // Run backward through integration
        var old_deriv_state = diffPair(from_dual_volume_state(previous_dual_state, previous_sample), {});
        var deriv_sample = diffPair(sample, {});
        bwd_diff(integrate_volume_sample)(old_deriv_state, deriv_sample, tmin, tmax, max_prim_size, deriv_state.d);

        // Get current primitive
        let prim_ind = tri_ind / tri_per_g;
        let face_id = tri_ind % tri_per_g;

        let mean = load_float3(means, prim_ind);
        let scale = load_float3(scales, prim_ind);
        let quat = load_float4(quats, prim_ind);
        let density = densities[prim_ind];
        Features feat = load_features(features, prim_ind, sh_degree, feat_stride);
        float3 color = eval_color(direction, feat, sh_degree);

        // Backward through intersection
        var deriv_origin = diffPair(origin, {});
        var deriv_direction = diffPair(direction, {});
        var deriv_scales = diffPair(scale, {});
        var deriv_mean = diffPair(mean, {});
        var deriv_quat = diffPair(quat, {});
        var deriv_color = diffPair(color, {});
        var deriv_density = diffPair(density, {});

        bwd_diff(safe_intersect)(deriv_origin, deriv_direction,
            deriv_scales, deriv_mean, deriv_quat, deriv_color, deriv_density, face_id, skip_close, deriv_sample.d);

        // Accumulate gradients
        atomic_add_float3(dL_dmeans, prim_ind, deriv_mean.d);
        atomic_add_float3(dL_dscales, prim_ind, deriv_scales.d);
        atomic_add_float4(dL_dquats, prim_ind, deriv_quat.d);
        atomicAdd(&dL_ddensities[prim_ind], deriv_density.d);

        // Color gradient -> feature gradient
        float3 d_rayd = deriv_direction.d;
        deriv_direction = diffPair(direction, {});
        var d_feat = diffPair(feat, {});
        bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color.d);
        d_rayd += deriv_direction.d;

        // Accumulate feature gradients
        atomic_add_float3_2d(dL_dfeatures, prim_ind, 0u, feat_stride, d_feat.d.f0);
        if (sh_degree > 0) {
            atomic_add_float3_2d(dL_dfeatures, prim_ind, 1u, feat_stride, d_feat.d.f1);
            atomic_add_float3_2d(dL_dfeatures, prim_ind, 2u, feat_stride, d_feat.d.f2);
            atomic_add_float3_2d(dL_dfeatures, prim_ind, 3u, feat_stride, d_feat.d.f3);
            if (sh_degree > 1) {
                atomic_add_float3_2d(dL_dfeatures, prim_ind, 4u, feat_stride, d_feat.d.f4);
                atomic_add_float3_2d(dL_dfeatures, prim_ind, 5u, feat_stride, d_feat.d.f5);
                atomic_add_float3_2d(dL_dfeatures, prim_ind, 6u, feat_stride, d_feat.d.f6);
                atomic_add_float3_2d(dL_dfeatures, prim_ind, 7u, feat_stride, d_feat.d.f7);
                atomic_add_float3_2d(dL_dfeatures, prim_ind, 8u, feat_stride, d_feat.d.f8);
                if (sh_degree > 2) {
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 9u, feat_stride, d_feat.d.f9);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 10u, feat_stride, d_feat.d.f10);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 11u, feat_stride, d_feat.d.f11);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 12u, feat_stride, d_feat.d.f12);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 13u, feat_stride, d_feat.d.f13);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 14u, feat_stride, d_feat.d.f14);
                    atomic_add_float3_2d(dL_dfeatures, prim_ind, 15u, feat_stride, d_feat.d.f15);
                }
            }
        }

        // Ray gradients
        atomic_add_float3(dL_drayos, ray_ind, deriv_origin.d);
        atomic_add_float3(dL_drayds, ray_ind, d_rayd);

        // 2D mean gradient for splatting
        float3 xyd = project(mean, wct);
        var d_xy = diffPair(float2(xyd.x, xyd.y), {});
        var d_dist = diffPair(xyd.z, {});
        var d_inv_wct = diffPair(inv_wct, {});
        bwd_diff(inv_project)(d_xy, d_dist, d_inv_wct, deriv_mean.d);
        atomic_add_float2(dL_dmeans2D, prim_ind, d_xy.d);

        // Touch count
        atomicAdd(&touch_count[prim_ind], 1);

        // Update for next iteration
        tri_ind = old_tri_ind;
        dual_state = previous_dual_state;
        sample = previous_sample;
        deriv_state = diffPair(previous_dual_state, old_deriv_state.d);
    }

    // Store initial drgb gradient
    uint drgb_base = ray_ind * 4;
    dL_dinitial_drgb[drgb_base + 0] = deriv_state.d.drgb.x;
    dL_dinitial_drgb[drgb_base + 1] = deriv_state.d.drgb.y;
    dL_dinitial_drgb[drgb_base + 2] = deriv_state.d.drgb.z;
    dL_dinitial_drgb[drgb_base + 3] = deriv_state.d.drgb.w;
}

// C-callable launch wrapper
extern "C" void backwards_kernel_launch(
    dim3 gridDim,
    dim3 blockDim,
    cudaStream_t stream,
    float* last_state,
    float* last_dirac,
    int* iters,
    int* tri_collection,
    float* ray_origins,
    float* ray_directions,
    float* means,
    float* scales,
    float* quats,
    float* densities,
    float* features,
    float* dL_dmeans,
    float* dL_dscales,
    float* dL_dquats,
    float* dL_ddensities,
    float* dL_dfeatures,
    float* dL_drayos,
    float* dL_drayds,
    float* dL_dmeans2D,
    float* initial_drgb,
    float* dL_dinitial_drgb,
    int* touch_count,
    float* dL_doutputs,
    float* wcts,
    float tmin,
    float tmax,
    float max_prim_size,
    unsigned int max_iters,
    unsigned int num_rays,
    unsigned int num_prims,
    unsigned int feature_size
) {
    backwards_kernel<<<gridDim, blockDim, 0, stream>>>(
        last_state, last_dirac, iters, tri_collection,
        ray_origins, ray_directions,
        means, scales, quats, densities, features,
        dL_dmeans, dL_dscales, dL_dquats, dL_ddensities, dL_dfeatures,
        dL_drayos, dL_drayds, dL_dmeans2D,
        initial_drgb, dL_dinitial_drgb, touch_count,
        dL_doutputs, wcts,
        tmin, tmax, max_prim_size, max_iters, num_rays, num_prims, feature_size
    );
}
