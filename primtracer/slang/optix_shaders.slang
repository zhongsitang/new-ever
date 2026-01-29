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
// OptiX Shaders for Ellipsoid Volume Rendering
// =============================================================================
//
// Volume rendering of ellipsoid primitives using sorted hit collection.
//
// Hit Processing:
//   - Each ellipsoid generates entry (+density) and exit (-density) events
//   - Hits are collected into a fixed-size buffer, sorted by (t, hit_id)
//   - When buffer fills, ray is re-traced from last processed t
//
// Hit ID Encoding:
//   hit_id = prim_idx * 2 + hit_type (exit=0, entry=1)
//   Exit sorts before Entry at same t
//
// =============================================================================

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// =============================================================================
// Constants
// =============================================================================

#define HIT_BUFFER_SIZE 16

static const float T_SENTINEL = 1e10f; // Marks empty buffer slots

// =============================================================================
// Data Structures
// =============================================================================

struct HitBuffer
{
    float t[HIT_BUFFER_SIZE];
    uint hit_id[HIT_BUFFER_SIZE];
    uint next_id; // Next hit_id to process at RayTMin()
};

struct HitAttr
{
    uint entry_t; // Bit-cast float via asuint/asfloat
};

// =============================================================================
// Launch Parameters (must match C++ LaunchParams layout)
// =============================================================================

// Input buffers
StructuredBuffer<float> ray_origins;
StructuredBuffer<float> ray_directions;
StructuredBuffer<float> tmax;
StructuredBuffer<float> means;
StructuredBuffer<float> scales;
StructuredBuffer<float> quats;
StructuredBuffer<float> densities;
StructuredBuffer<float> features;

// Output buffers
RWStructuredBuffer<float> image;
RWStructuredBuffer<float> depth;
RWStructuredBuffer<IntegratorState> last_state;
RWStructuredBuffer<float> last_contrib;
RWStructuredBuffer<int> last_prim;
RWStructuredBuffer<int> prim_hits;
RWStructuredBuffer<int> ray_hits;
RWStructuredBuffer<int> hit_collection;

// Scalars (order must match C++ LaunchParams)
int sh_degree;
int max_hits;
float min_logT;
float tmin;

// Acceleration structure
RaytracingAccelerationStructure traversable;

// =============================================================================
// Helper Functions
// =============================================================================

float3 load_float3(StructuredBuffer<float> buf, uint idx)
{
    return float3(buf[idx * 3], buf[idx * 3 + 1], buf[idx * 3 + 2]);
}

float4 load_float4(StructuredBuffer<float> buf, uint idx)
{
    return float4(buf[idx * 4], buf[idx * 4 + 1], buf[idx * 4 + 2], buf[idx * 4 + 3]);
}

Features get_feats(StructuredBuffer<float> shs, uint prim_idx, uint sh_degree) {
    Features feat;
    feat.degree = sh_degree;
    uint num_coeff = (sh_degree + 1) * (sh_degree + 1);
    uint base = prim_idx * num_coeff * 3;
    for (uint i = 0; i < num_coeff; i++) {
        uint offset = base + i * 3;
        feat.c[i] = float3(shs[offset], shs[offset + 1], shs[offset + 2]);
    }
    return feat;
}

void store_float4(RWStructuredBuffer<float> buf, uint idx, float4 val)
{
    buf[idx * 4 + 0] = val.x;
    buf[idx * 4 + 1] = val.y;
    buf[idx * 4 + 2] = val.z;
    buf[idx * 4 + 3] = val.w;
}

HitBuffer init_hit_buffer(uint next_id)
{
    HitBuffer buf;
    buf.next_id = next_id;
    [ForceUnroll]
    for (int i = 0; i < HIT_BUFFER_SIZE; i++)
    {
        buf.t[i] = T_SENTINEL;
        buf.hit_id[i] = 0;
    }
    return buf;
}

SamplePoint create_sample(uint hit_id, float t, float3 ray_dir)
{
    uint prim_idx = hit_id >> 1;
    float density = densities[prim_idx];
    float signed_density = ((hit_id & 1) != 0) ? density : -density;  // entry=+, exit=-
    float3 color = eval_color(ray_dir, get_feats(features, prim_idx, sh_degree));
    return { t, float4(signed_density, signed_density * color) };
}

/// Insert hit maintaining (t, hit_id) order.
void insert_hit(inout HitBuffer buf, float t, uint hit_id)
{
    [ForceUnroll]
    for (int i = 0; i < HIT_BUFFER_SIZE; i++)
        if (t < buf.t[i] || (t == buf.t[i] && hit_id < buf.hit_id[i]))
        {
            float tmp_t = buf.t[i];
            buf.t[i] = t;
            t = tmp_t;
            uint tmp_id = buf.hit_id[i];
            buf.hit_id[i] = hit_id;
            hit_id = tmp_id;
        }
}

// =============================================================================
// OptiX Shaders
// =============================================================================

[shader("intersection")]
void intersect_ellipsoid()
{
    uint prim_idx = PrimitiveIndex();
    float3 ray_origin = WorldRayOrigin();
    float3 ray_dir = WorldRayDirection();
    float cur_t = RayTMin();

    float3 mean = load_float3(means, prim_idx);
    float3 scale = load_float3(scales, prim_idx);
    float4 quat = load_float4(quats, prim_idx);

    float2 t = ray_intersect_ellipsoid(ray_origin, ray_dir, mean, scale, quat);

    // Clamp entry to global tmin (for rays starting inside ellipsoid)
    float t_entry = max(t.x, max(tmin, 0.0f));
    float t_exit = min(t.y, RayTCurrent());
    
    // Skip if exit behind cur_t
    if (t_exit < cur_t)
        return;

    // Report entry hit
    ReportHitOptix(t_exit, 0, asuint(t_entry));
}

[shader("closesthit")]
void closest_hit(inout HitBuffer buf, in HitAttr attr)
{
}

[shader("anyhit")]
void collect_hits(inout HitBuffer buf, in HitAttr attr)
{
    uint prim_idx = PrimitiveIndex();
    float cur_t = RayTMin();
    float t_exit = RayTCurrent();
    float t_entry = asfloat(attr.entry_t);

    uint id_entry = prim_idx * 2 + 1;
    uint id_exit = prim_idx * 2;

    // Skip hits already processed in previous trace
    bool skip_entry = (t_entry == cur_t && id_entry < buf.next_id);
    bool skip_exit = (t_exit == cur_t && id_exit < buf.next_id);

    if (t_entry >= cur_t && !skip_entry)
        insert_hit(buf, t_entry, id_entry);

    if (t_exit >= cur_t && !skip_exit)
        insert_hit(buf, t_exit, id_exit);

    IgnoreHit();
}

[shader("miss")]
void miss(inout HitBuffer buf)
{
}

[shader("raygeneration")]
void render_volume()
{
    uint ray_idx = DispatchRaysIndex().x;
    uint num_rays_dim = DispatchRaysDimensions().x;

    float3 ray_origin = load_float3(ray_origins, ray_idx);
    float3 ray_dir = l2_normalize(load_float3(ray_directions, ray_idx));
    float ray_tmin = max(tmin, 0.0f);
    float ray_tmax = tmax[ray_idx];

    IntegratorState state = {};
    state.t = ray_tmin;

    SamplePoint sample = {};
    int iter = 0;
    uint hit_id = 0;
    uint next_id = 0;

    while (state.logT > min_logT && iter < max_hits)
    {
        HitBuffer buf = init_hit_buffer(next_id);
        TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 0, 0,
                 { ray_origin, state.t, ray_dir, ray_tmax }, buf);

        bool done = false;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++)
        {
            float hit_t = buf.t[i];
            if (hit_t >= T_SENTINEL)
            {
                done = true;
                break;
            }

            hit_id = buf.hit_id[i];
            next_id = hit_id + 1;

            sample = create_sample(hit_id, hit_t, ray_dir);
            state = integrate_step(state, sample);

            int tmp;
            InterlockedAdd(prim_hits[hit_id >> 1], 1, tmp);
            hit_collection[ray_idx + iter * num_rays_dim] = hit_id;
            iter++;

            if (!(state.logT > min_logT && iter < max_hits))
                break;
        }
        if (done)
            break;
    }

    // Output
    let output = extract_output(state);
    store_float4(image, ray_idx, float4(output.C, output.alpha));
    depth[ray_idx] = output.depth;

    // Save for backward pass
    last_state[ray_idx] = state;
    last_prim[ray_idx] = hit_id;
    ray_hits[ray_idx] = iter;
    store_float4(last_contrib, ray_idx, sample.contrib);
}
