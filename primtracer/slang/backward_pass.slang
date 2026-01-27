// Backward Pass for Ellipsoid Volume Rendering
// Optimizations: per-ray context caching, ping-pong hit buffer, pure backward step

import tensor_utils;
import ellipsoid_intersect;
import volume_integrator;
import spherical_harmonics;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

struct Primitives
{
    TensorView<float> means, scales, quats, densities, features;
    TensorView<float> dL_dmeans, dL_dscales, dL_dquats, dL_ddensities, dL_dfeatures;
};

struct Rays
{
    TensorView<float> origins, directions, tmax;
    float tmin;
    TensorView<float> dL_drayos, dL_drayds;
};

struct SavedState
{
    TensorView<float> last_state, last_contrib;
    TensorView<int> ray_hits, hit_collection;
};

struct RayCtx
{
    float3 rayo, rayd;
    float tmin, tmax;
    int sh_deg;
};

struct HitCache
{
    int prim_idx, hit_type;
    float3 mean, scale;
    float4 quat;
    float density;
    Features feat;
};

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

inline int compute_sh_deg(in Primitives prims)
{
    return int(sqrt(float(prims.features.size(1)))) - 1;
}

// -----------------------------------------------------------------------------
// Sample Computation
// -----------------------------------------------------------------------------
// Hit encoding: hit_id = prim_idx * 2 + hit_type (exit=0, entry=1)
// Clamp rule: entry -> max(t, tmin), exit -> min(t, tmax)

[Differentiable]
SamplePoint compute_sample(
    float3 rayo, float3 rayd,
    float3 mean, float3 scale, float4 quat,
    float density, Features feat,
    no_diff int hit_type, no_diff float tmin, no_diff float tmax)
{
    float3 color = eval_color(rayd, feat);
    float2 t_range = ray_intersect_ellipsoid(rayo, rayd, mean, scale, quat);

    bool is_entry = (hit_type == 1);
    float t = is_entry ? max(t_range.x, tmin) : min(t_range.y, tmax);
    float signed_density = is_entry ? density : -density;

    return { t, float4(signed_density, signed_density * color) };
}

// -----------------------------------------------------------------------------
// Hit Cache Loading
// -----------------------------------------------------------------------------

inline void load_hit_cache(
    in Primitives prims, in RayCtx ray, int hit_id,
    out HitCache hit, out SamplePoint sample)
{
    hit.prim_idx = hit_id >> 1;
    hit.hit_type = hit_id & 1;

    hit.mean = get_float3(prims.means, hit.prim_idx);
    hit.scale = get_float3(prims.scales, hit.prim_idx);
    hit.quat = get_float4(prims.quats, hit.prim_idx);
    hit.density = prims.densities[hit.prim_idx];
    hit.feat = get_feats(prims.features, hit.prim_idx, ray.sh_deg);

    sample = compute_sample(
        ray.rayo, ray.rayd,
        hit.mean, hit.scale, hit.quat,
        hit.density, hit.feat,
        hit.hit_type, ray.tmin, ray.tmax);
}

// -----------------------------------------------------------------------------
// Backward Step
// -----------------------------------------------------------------------------

inline IntegratorState.Differential backward_step(
    inout Primitives prims, inout Rays rays,
    IntegratorState prev_state, SamplePoint prev_sample, SamplePoint sample,
    in RayCtx ray, in HitCache hit,
    IntegratorState.Differential d_out, int ray_idx)
{
    // Backward: integrate_step
    var dp_prev = diffPair(prev_state, {});
    var dp_sample = diffPair(sample, {});
    bwd_diff(integrate_step)(dp_prev, dp_sample, d_out);

    // Backward: compute_sample
    var dp_origin = diffPair(ray.rayo, {});
    var dp_dir = diffPair(ray.rayd, {});
    var dp_mean = diffPair(hit.mean, {});
    var dp_scale = diffPair(hit.scale, {});
    var dp_quat = diffPair(hit.quat, {});
    var dp_density = diffPair(hit.density, {});
    var dp_feat = diffPair(hit.feat, {});

    bwd_diff(compute_sample)(
        dp_origin, dp_dir,
        dp_mean, dp_scale, dp_quat,
        dp_density, dp_feat,
        hit.hit_type, ray.tmin, ray.tmax,
        dp_sample.d);

    // Accumulate gradients
    atomic_add_float3(prims.dL_dmeans, hit.prim_idx, dp_mean.d);
    atomic_add_float3(prims.dL_dscales, hit.prim_idx, dp_scale.d);
    atomic_add_float4(prims.dL_dquats, hit.prim_idx, dp_quat.d);
    atomic_add_float(prims.dL_ddensities, hit.prim_idx, dp_density.d);

    uint num_coeff = (hit.feat.degree + 1) * (hit.feat.degree + 1);
    for (uint i = 0; i < num_coeff; i++)
        atomic_add_float3(prims.dL_dfeatures, hit.prim_idx, i, dp_feat.d.c[i]);

    atomic_add_float3(rays.dL_drayos, ray_idx, dp_origin.d);
    atomic_add_float3(rays.dL_drayds, ray_idx, dp_dir.d);

    return dp_prev.d;
}

// -----------------------------------------------------------------------------
// CUDA Kernel
// -----------------------------------------------------------------------------

[AutoPyBindCUDA]
[CUDAKernel]
void bw_trace_rays(
    Primitives prims, Rays rays, SavedState saved,
    TensorView<float> dL_doutputs, int max_hits)
{
    int ray_idx = int(cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x);
    if (ray_idx >= rays.origins.size(0))
        return;

    int num_hits = saved.ray_hits[ray_idx];
    if (num_hits > max_hits - 1 || num_hits <= 0)
        return;

    int hit_stride = rays.origins.size(0);

    // Load per-ray context
    RayCtx ray;
    ray.rayo = get_float3(rays.origins, ray_idx);
    ray.rayd = get_float3(rays.directions, ray_idx);
    ray.tmin = rays.tmin;
    ray.tmax = rays.tmax[ray_idx];
    ray.sh_deg = compute_sh_deg(prims);

    // Initialize state and output gradients
    var state = get_state(saved.last_state, ray_idx);
    var dp_state = diffPair(state, {});

    RenderOutput.Differential dL_out;
    dL_out.C = float3(dL_doutputs[ray_idx, 0], dL_doutputs[ray_idx, 1], dL_doutputs[ray_idx, 2]);
    dL_out.alpha = dL_doutputs[ray_idx, 3];
    dL_out.depth = dL_doutputs[ray_idx, 4];
    bwd_diff(extract_output)(dp_state, dL_out);

    IntegratorState.Differential d_state = dp_state.d;

    // Ping-pong buffers
    HitCache hitBuf[2];
    SamplePoint sampBuf[2];

    // Load last hit into slot 0
    int last_hit_id = saved.hit_collection[ray_idx + (num_hits - 1) * hit_stride];
    load_hit_cache(prims, ray, last_hit_id, hitBuf[0], sampBuf[0]);
    int cur = 0;

    // Backward iteration
    for (int i = num_hits; i-- > 0;)
    {
        int prev = cur ^ 1;

        // Prefetch previous hit into alternate slot
        if (i > 0)
        {
            int prev_hit_id = saved.hit_collection[ray_idx + (i - 1) * hit_stride];
            load_hit_cache(prims, ray, prev_hit_id, hitBuf[prev], sampBuf[prev]);
        }
        else
        {
            sampBuf[prev].t = ray.tmin;
            sampBuf[prev].contrib = float4(0.f);
        }

        // Compute previous state and run backward step
        IntegratorState prev_state = invert_step(state, sampBuf[cur], sampBuf[prev]);

        d_state = backward_step(
            prims, rays,
            prev_state, sampBuf[prev], sampBuf[cur],
            ray, hitBuf[cur],
            d_state, ray_idx);

        state = prev_state;
        if (i > 0)
            cur = prev;
    }
}
