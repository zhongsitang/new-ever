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

#define RT_EPS 0
#define tri_per_g 2
#define LOG_CUTOFF 5.54
#define BUFFER_SIZE 16
import spline_machine;
import tri_intersect;
import safe_math;
import sh;

struct Camera {
    float fx, fy;
    int height;
    int width;
    float2 _pad0;
    float3 U, V, W;
    float3 eye;
    float _pad1;
};


// LaunchParams
RWStructuredBuffer<float4>      image;
RWStructuredBuffer<uint>        iters;
RWStructuredBuffer<uint>        last_face;
RWStructuredBuffer<uint>        touch_count;
RWStructuredBuffer<float4>      last_dirac;
RWStructuredBuffer<SplineState> last_state;
RWStructuredBuffer<int> tri_collection;
StructuredBuffer<float3>      ray_origins;
StructuredBuffer<float3>      ray_directions;
Camera camera;

RWStructuredBuffer<float3> means;
RWStructuredBuffer<float3> scales;
RWStructuredBuffer<float4> quats;
RWStructuredBuffer<float> densities;
RWStructuredBuffer<float> features;

uint sh_degree;
uint max_iters;
float tmin;
float tmax;
RWStructuredBuffer<float4> initial_drgb;
float max_prim_size;
RaytracingAccelerationStructure traversable;

struct [raypayload] RayPayload
{
    uint data[2 * BUFFER_SIZE];
};

struct EllipsoidHitAttributes
{
    float2 t_pair;
};


ControlPoint get_ctrl_pt(uint tri, float t) {
    ControlPoint ctrl_pt;
    let prim_ind = tri / 2;
    let hitkind = tri % 2;
    let height = densities[prim_ind];
    let dirac_height = height * ((hitkind == 1) ? 1 : -1);
    Features feat;
    SHFeatures sh_feats = {prim_ind, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    let color = eval_sh_col0(rayd, feat);
    ctrl_pt.t = t;
    ctrl_pt.dirac.x = dirac_height;
    ctrl_pt.dirac.y = dirac_height * color.x;
    ctrl_pt.dirac.z = dirac_height * color.y;
    ctrl_pt.dirac.w = dirac_height * color.z;
    return ctrl_pt;
}

[shader("intersection")]
void intersectionShader() {
    uint prim_ind = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    let mean = means[prim_ind];
    let scale = scales[prim_ind];
    let quat = quats[prim_ind];
    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();
    if ((minmaxt.y < cur_t)) {
        return;
    }

    EllipsoidHitAttributes attrs;
    bool use_min = cur_t < minmaxt.x;
    if (use_min) {
        attrs.t_pair = float2(minmaxt.x, minmaxt.y);
        ReportHit(minmaxt.x, 1u, attrs);
    } else if (cur_t < minmaxt.y) {
        attrs.t_pair = float2(minmaxt.x, minmaxt.y);
        ReportHit(minmaxt.y, 0u, attrs);
    }
}

[shader("anyhit")]
void anyHitShader(inout RayPayload payload, in EllipsoidHitAttributes attrs : SV_IntersectionAttributes)
{
    float t = RayTCurrent();
    uint ind = PrimitiveIndex();
    uint hitkind = HitKind();
    float other_t = (hitkind == 1) ? attrs.t_pair.y : attrs.t_pair.x;
    if (hitkind == 0) {
        float temp_t = other_t;
        other_t = t;
        t = temp_t;
    }

    float cur_t = RayTMin();

    float h_t;
    uint h_i;

    float test_t;
    uint test_i;

    for (int n=0; n<2; n++) {
        if (n == 0) {
            h_t = t;
            h_i = 2 * ind + 1;
        } else {
            h_t = other_t;
            h_i = 2 * ind;
        }
        if (h_t > cur_t) {
            [ForceUnroll]
            for (int i=0; i<BUFFER_SIZE; i++) {
                test_t = asfloat(payload.data[i * 2]);
                if (h_t < test_t) {
                    payload.data[i * 2] = asuint(h_t);
                    test_i = payload.data[i * 2 + 1];
                    payload.data[i * 2 + 1] = h_i;
                    h_i = test_i;
                    h_t = test_t;
                }
            }
        }
    }

    if (t < asfloat(payload.data[2 * (BUFFER_SIZE - 1)])) {
        IgnoreHit();
    }
}

[shader("miss")]
void missShader(inout RayPayload payload)
{
}

[shader("raygeneration")]
void rayGenShader()
{
    let FAST_MODE = false;

    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];
    SplineState state = make_empty_state();
    state.t = 0;
    state.drgb = initial_drgb[idx.x];

    let start_id = idx.x * max_iters;
    
    ControlPoint ctrl_pt = {};
    uint last_tri = -1;
    float prev_t = state.t;
    uint next_tri = -1;
    float next_t = 1e20;

    uint tri;

    int iter = 0;
    while (state.logT < LOG_CUTOFF && iter < max_iters)
    {
        let start_t = abs(state.t);
        RayPayload payload = {};
        for (int i=0; i<BUFFER_SIZE; i++) {
            payload.data[2 * i] = asuint(1e10f);
            payload.data[2 * i + 1] = asuint(0xffffffff);
        }

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = start_t;
        ray.TMax = tmax;
        TraceRay(traversable, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
        bool end = false;
        for (int i=0; i<BUFFER_SIZE; i++) {
            ctrl_pt.t = asfloat(payload.data[2 * i]);
            tri = payload.data[2 * i + 1];
            if (ctrl_pt.t > 1e9) {
                end = true;
                break;
            }
            ctrl_pt = get_ctrl_pt(tri, ctrl_pt.t);
            state = update(state, ctrl_pt, tmin, tmax, max_prim_size);
            uint prev_value = 0;
            InterlockedAdd(touch_count[tri / tri_per_g], 1, prev_value);
            tri_collection[idx.x + iter * dim.x] = tri;
            iter++;
            if (!(state.logT < LOG_CUTOFF && iter < max_iters)) break;
        }
        if (end) break;

    }
    let output = extract_color(state, tmin);
    image[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};
    let dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
