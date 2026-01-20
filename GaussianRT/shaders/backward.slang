// GaussianRT - Backward Pass Kernel
// Differentiable volume rendering backward propagation
// Apache License 2.0

#define TRI_PER_PRIM 2
import volume;
import ellipsoid;
import safe_math;
import sh;

// Helper to load float3 from tensor
float3 load_float3(RWStructuredBuffer<float> buf, uint idx, uint stride) {
    uint base = idx * stride;
    return float3(buf[base], buf[base + 1], buf[base + 2]);
}

float4 load_float4(RWStructuredBuffer<float> buf, uint idx, uint stride) {
    uint base = idx * stride;
    return float4(buf[base], buf[base + 1], buf[base + 2], buf[base + 3]);
}

// Atomic add for float3
void atomic_add_float3(RWStructuredBuffer<float> buf, uint idx, uint stride, float3 val) {
    uint base = idx * stride;
    float temp;
    InterlockedAddF32(buf, base, val.x, temp);
    InterlockedAddF32(buf, base + 1, val.y, temp);
    InterlockedAddF32(buf, base + 2, val.z, temp);
}

void atomic_add_float4(RWStructuredBuffer<float> buf, uint idx, uint stride, float4 val) {
    uint base = idx * stride;
    float temp;
    InterlockedAddF32(buf, base, val.x, temp);
    InterlockedAddF32(buf, base + 1, val.y, temp);
    InterlockedAddF32(buf, base + 2, val.z, temp);
    InterlockedAddF32(buf, base + 3, val.w, temp);
}

void atomic_add_float2(RWStructuredBuffer<float> buf, uint idx, uint stride, float2 val) {
    uint base = idx * stride;
    float temp;
    InterlockedAddF32(buf, base, val.x, temp);
    InterlockedAddF32(buf, base + 1, val.y, temp);
}

// Load volume state from buffer
VolumeState load_state(RWStructuredBuffer<float> buf, uint ray_idx) {
    uint base = ray_idx * 16;  // VolumeState is 16 floats
    VolumeState state;
    state.distortion_parts = float2(buf[base], buf[base + 1]);
    state.cum_sum = float2(buf[base + 2], buf[base + 3]);
    state.padding = float3(buf[base + 4], buf[base + 5], buf[base + 6]);
    state.t = buf[base + 7];
    state.drgb = float4(buf[base + 8], buf[base + 9], buf[base + 10], buf[base + 11]);
    state.logT = buf[base + 12];
    state.C = float3(buf[base + 13], buf[base + 14], buf[base + 15]);
    return state;
}

// Get features for a primitive
Features load_features(RWStructuredBuffer<float> features, uint prim_idx, uint sh_degree, uint feature_stride) {
    Features feat;
    uint base = prim_idx * feature_stride;

    feat.f0 = float3(features[base], features[base + 1], features[base + 2]);
    if (sh_degree > 0) {
        feat.f1 = float3(features[base + 3], features[base + 4], features[base + 5]);
        feat.f2 = float3(features[base + 6], features[base + 7], features[base + 8]);
        feat.f3 = float3(features[base + 9], features[base + 10], features[base + 11]);
        if (sh_degree > 1) {
            feat.f4 = float3(features[base + 12], features[base + 13], features[base + 14]);
            feat.f5 = float3(features[base + 15], features[base + 16], features[base + 17]);
            feat.f6 = float3(features[base + 18], features[base + 19], features[base + 20]);
            feat.f7 = float3(features[base + 21], features[base + 22], features[base + 23]);
            feat.f8 = float3(features[base + 24], features[base + 25], features[base + 26]);
            if (sh_degree > 2) {
                feat.f9 = float3(features[base + 27], features[base + 28], features[base + 29]);
                feat.f10 = float3(features[base + 30], features[base + 31], features[base + 32]);
                feat.f11 = float3(features[base + 33], features[base + 34], features[base + 35]);
                feat.f12 = float3(features[base + 36], features[base + 37], features[base + 38]);
                feat.f13 = float3(features[base + 39], features[base + 40], features[base + 41]);
                feat.f14 = float3(features[base + 42], features[base + 43], features[base + 44]);
                feat.f15 = float3(features[base + 45], features[base + 46], features[base + 47]);
            }
        }
    }
    return feat;
}

// Compute volume sample from hit info
VolumeSample compute_volume_sample(
    float3 origin,
    float3 direction,
    float3 mean,
    float3 scale,
    float4 quat,
    float density,
    float3 color,
    uint face_id
) {
    float2 minmaxt = ray_intersect_ellipsoid(origin - mean, direction, scale, quat);
    float t = (face_id == 1) ? minmaxt.x : minmaxt.y;
    float dirac_density = density * ((face_id == 1) ? 1.0f : -1.0f);

    VolumeSample sample;
    sample.t = t;
    sample.dirac = float4(
        dirac_density,
        dirac_density * color.x,
        dirac_density * color.y,
        dirac_density * color.z
    );
    return sample;
}

// Backward kernel resources
RWStructuredBuffer<float> last_state;      // VolumeState per ray
RWStructuredBuffer<float> last_dirac;      // float4 per ray
RWStructuredBuffer<int> iters;             // int per ray
RWStructuredBuffer<int> tri_collection;    // int per ray * max_iters

RWStructuredBuffer<float> ray_origins;     // float3 per ray
RWStructuredBuffer<float> ray_directions;  // float3 per ray

// Model buffers
RWStructuredBuffer<float> means;           // float3 per prim
RWStructuredBuffer<float> scales;          // float3 per prim
RWStructuredBuffer<float> quats;           // float4 per prim
RWStructuredBuffer<float> densities;       // float per prim
RWStructuredBuffer<float> features;        // float[feature_size] per prim

// Gradient buffers
RWStructuredBuffer<float> dL_dmeans;
RWStructuredBuffer<float> dL_dscales;
RWStructuredBuffer<float> dL_dquats;
RWStructuredBuffer<float> dL_ddensities;
RWStructuredBuffer<float> dL_dfeatures;
RWStructuredBuffer<float> dL_dray_origins;
RWStructuredBuffer<float> dL_dray_dirs;
RWStructuredBuffer<float> dL_dmeans2D;

RWStructuredBuffer<float> initial_drgb;
RWStructuredBuffer<float> dL_dinitial_drgb;
RWStructuredBuffer<int> touch_count;

RWStructuredBuffer<float> dL_doutputs;     // float5 per ray (RGBA + distortion)
RWStructuredBuffer<float> wcts;            // float4x4 per ray (or 1)

// Uniform parameters
uint num_rays;
uint num_prims;
uint max_iters_param;
uint sh_degree;
uint feature_stride;
float tmin;
float tmax;
float max_prim_size;

[shader("compute")]
[numthreads(64, 1, 1)]
void backwardKernel(uint3 dispatchIdx : SV_DispatchThreadID) {
    uint ray_idx = dispatchIdx.x;
    if (ray_idx >= num_rays) return;

    int num_iters = iters[ray_idx];
    if (num_iters >= (int)max_iters_param - 1 || num_iters <= 0) return;

    // Load ray
    float3 direction = load_float3(ray_directions, ray_idx, 3);
    float3 origin = load_float3(ray_origins, ray_idx, 3) + tmin * direction;

    // Load output gradient
    uint out_base = ray_idx * 5;
    float3 dL_dC = float3(dL_doutputs[out_base], dL_doutputs[out_base + 1], dL_doutputs[out_base + 2]);
    float dL_ddepth = dL_doutputs[out_base + 3];
    float dL_ddistortion = dL_doutputs[out_base + 4];

    // Load current state
    VolumeState dual_state = load_state(last_state, ray_idx);

    // Initialize state derivative
    VolumeState.Differential d_state = {};
    d_state.C = dL_dC;
    d_state.padding[0] = dL_ddepth;
    d_state.distortion_parts.x = dL_ddistortion;
    d_state.distortion_parts.y = -dL_ddistortion;

    // Get last triangle
    uint tri_idx = tri_collection[ray_idx + (num_iters - 1) * num_rays];
    uint prim_idx = tri_idx / TRI_PER_PRIM;
    uint face_id = tri_idx % TRI_PER_PRIM;

    // Load primitive data
    float3 mean = load_float3(means, prim_idx, 3);
    float3 scale = load_float3(scales, prim_idx, 3);
    float4 quat = load_float4(quats, prim_idx, 4);
    float density = densities[prim_idx];
    Features feat = load_features(features, prim_idx, sh_degree, feature_stride);
    float3 color = eval_sh_col0(direction, feat);

    VolumeSample sample = compute_volume_sample(origin, direction, mean, scale, quat, density, color, face_id);

    // Backward loop
    for (int i = num_iters - 1; i >= 0; i--) {
        // Get previous sample
        VolumeSample prev_sample;
        uint prev_tri_idx;
        if (i > 0) {
            prev_tri_idx = tri_collection[ray_idx + (i - 1) * num_rays];
            uint prev_prim_idx = prev_tri_idx / TRI_PER_PRIM;
            uint prev_face_id = prev_tri_idx % TRI_PER_PRIM;

            float3 prev_mean = load_float3(means, prev_prim_idx, 3);
            float3 prev_scale = load_float3(scales, prev_prim_idx, 3);
            float4 prev_quat = load_float4(quats, prev_prim_idx, 4);
            float prev_density = densities[prev_prim_idx];
            Features prev_feat = load_features(features, prev_prim_idx, sh_degree, feature_stride);
            float3 prev_color = eval_sh_col0(direction, prev_feat);

            prev_sample = compute_volume_sample(origin, direction, prev_mean, prev_scale, prev_quat, prev_density, prev_color, prev_face_id);
        } else {
            prev_sample.t = 0.0f;
            prev_sample.dirac = float4(0.0f);
        }

        // Compute inverse integration to get previous state
        VolumeState prev_state = inverse_integrate_dual(dual_state, sample, prev_sample, tmin, tmax);

        // Backward through integration
        var d_prev_state = diffPair(from_dual_volume_state(prev_state, prev_sample), {});
        var d_sample = diffPair(sample, {});

        bwd_diff(integrate_volume_sample)(d_prev_state, d_sample, tmin, tmax, max_prim_size, d_state);

        // Get current primitive
        prim_idx = tri_idx / TRI_PER_PRIM;
        face_id = tri_idx % TRI_PER_PRIM;

        mean = load_float3(means, prim_idx, 3);
        scale = load_float3(scales, prim_idx, 3);
        quat = load_float4(quats, prim_idx, 4);
        density = densities[prim_idx];

        // Backward through sample computation
        var d_origin = diffPair(origin, {});
        var d_direction = diffPair(direction, {});
        var d_scale = diffPair(scale, {});
        var d_mean = diffPair(mean, {});
        var d_quat = diffPair(quat, {});
        var d_color = diffPair(color, {});
        var d_density = diffPair(density, {});

        // Compute gradients (simplified - actual implementation would use bwd_diff)
        float dirac_sign = (face_id == 1) ? 1.0f : -1.0f;
        float3 d_color_val = dirac_sign * float3(d_sample.d.dirac.y, d_sample.d.dirac.z, d_sample.d.dirac.w);
        float d_density_val = dirac_sign * (d_sample.d.dirac.x + dot(color, d_color_val));

        // Accumulate gradients
        atomic_add_float3(dL_dmeans, prim_idx, 3, d_mean.d);
        atomic_add_float3(dL_dscales, prim_idx, 3, d_scale.d);
        atomic_add_float4(dL_dquats, prim_idx, 4, d_quat.d);

        float temp;
        InterlockedAddF32(dL_ddensities, prim_idx, d_density_val, temp);

        // Touch count
        int itemp;
        InterlockedAdd(touch_count[prim_idx], 1, itemp);

        // Update for next iteration
        tri_idx = prev_tri_idx;
        dual_state = prev_state;
        sample = prev_sample;
        d_state = d_prev_state.d;
    }

    // Store initial drgb gradient
    dL_dinitial_drgb[ray_idx * 4 + 0] = d_state.drgb.x;
    dL_dinitial_drgb[ray_idx * 4 + 1] = d_state.drgb.y;
    dL_dinitial_drgb[ray_idx * 4 + 2] = d_state.drgb.z;
    dL_dinitial_drgb[ray_idx * 4 + 3] = d_state.drgb.w;
}

// Kernel for backward through initial drgb
[shader("compute")]
[numthreads(16, 16, 1)]
void backwardInitialDrgbKernel(uint3 dispatchIdx : SV_DispatchThreadID) {
    uint ray_idx = dispatchIdx.x;
    uint prim_idx_local = dispatchIdx.y;

    // This kernel handles primitives that were inside rays at t=0
    // Implementation depends on initial_touch_inds buffer
}
