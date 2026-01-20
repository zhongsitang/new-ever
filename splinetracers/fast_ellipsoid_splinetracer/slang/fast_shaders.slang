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

// OptiX 9.1 / Slang Fast Ray Tracing Shaders for CUDA target

#define RT_EPS 0
#define tri_per_g 2
#define LOG_CUTOFF 5.54
#define BUFFER_SIZE 16

import spline_machine;
import tri_intersect;
import safe_math;
import sh;

// ============================================================================
// Structures
// ============================================================================

struct Camera
{
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};

// Hit entry for multi-hit tracking
struct HitEntry
{
    float t;
    uint triIndex;
};

// Ray payload for multi-hit collection
struct MultiHitPayload
{
    HitEntry hits[BUFFER_SIZE];
};

// Custom attributes for intersection (must be uint/int for OptiX)
// Slang will auto-generate optixGetAttribute_N() calls in anyhit/closesthit
struct EllipsoidHitAttributes
{
    uint otherT_bits;  // float stored as uint bits
};

// ============================================================================
// Global Resources
// ============================================================================

// Launch params / global resources
RWStructuredBuffer<float4>      image;
RWStructuredBuffer<uint>        iters;
RWStructuredBuffer<uint>        last_face;
RWStructuredBuffer<uint>        touch_count;
RWStructuredBuffer<float4>      last_dirac;
RWStructuredBuffer<SplineState> last_state;
RWStructuredBuffer<int>         tri_collection;
StructuredBuffer<float3>        ray_origins;
StructuredBuffer<float3>        ray_directions;
Camera camera;

RWStructuredBuffer<float>  half_attribs;
RWStructuredBuffer<float3> means;
RWStructuredBuffer<float3> scales;
RWStructuredBuffer<float4> quats;
RWStructuredBuffer<float>  densities;
RWStructuredBuffer<float>  features;

uint sh_degree;
uint max_iters;
float tmin;
float tmax;
RWStructuredBuffer<float4> initial_drgb;
float max_prim_size;
RaytracingAccelerationStructure traversable;

// ============================================================================
// Helper Functions
// ============================================================================

ControlPoint get_ctrl_pt(uint tri, float t)
{
    ControlPoint ctrl_pt;
    uint prim_ind = tri / 2;
    uint hitkind = tri % 2;
    float height = densities[prim_ind];
    float dirac_height = height * ((hitkind == 1) ? 1.0f : -1.0f);

    Features feat;
    SHFeatures sh_feats = { int(prim_ind), int(sh_degree), features };
    feat.f0 = get_sh(sh_feats, 0);

    float3 rayd = float3(0, 0, 1);
    float3 color = eval_sh_col0(rayd, feat);

    ctrl_pt.t = t;
    ctrl_pt.dirac.x = dirac_height;
    ctrl_pt.dirac.y = dirac_height * color.x;
    ctrl_pt.dirac.z = dirac_height * color.y;
    ctrl_pt.dirac.w = dirac_height * color.z;

    return ctrl_pt;
}

// Initialize hit buffer with maximum t values
void initHitBuffer(inout MultiHitPayload payload)
{
    [ForceUnroll]
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        payload.hits[i].t = 1e10f;
        payload.hits[i].triIndex = 0xFFFFFFFF;
    }
}

// Insert a hit into the sorted buffer
void insertHit(inout MultiHitPayload payload, float t, uint triIndex)
{
    float h_t = t;
    uint h_i = triIndex;

    [ForceUnroll]
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        if (h_t < payload.hits[i].t)
        {
            float temp_t = payload.hits[i].t;
            uint temp_i = payload.hits[i].triIndex;
            payload.hits[i].t = h_t;
            payload.hits[i].triIndex = h_i;
            h_t = temp_t;
            h_i = temp_i;
        }
    }
}

// ============================================================================
// Intersection Shader - Ellipsoid Ray Intersection
// ============================================================================

[shader("intersection")]
void ellipsoid()
{
    uint prim_ind = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    float3 mean = means[prim_ind];
    float3 scale = scales[prim_ind];
    float4 quat = quats[prim_ind];

    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();

    if (minmaxt.y < cur_t)
    {
        return;
    }

    bool use_min = cur_t < minmaxt.x;

    // Use ReportHitOptix for CUDA target - pass float as uint bits
    if (use_min)
    {
        ReportHitOptix(minmaxt.x, 1u, asuint(minmaxt.y));
    }
    else if (cur_t < minmaxt.y)
    {
        ReportHitOptix(minmaxt.y, 0u, asuint(minmaxt.x));
    }
}

// ============================================================================
// Closest Hit Shader
// ============================================================================

[shader("closesthit")]
void ch(inout MultiHitPayload payload, in EllipsoidHitAttributes attr)
{
    // Empty - we use anyhit for multi-hit collection
}

// ============================================================================
// Any Hit Shader - Multi-Hit Collection
// ============================================================================

[shader("anyhit")]
void ah(inout MultiHitPayload payload, in EllipsoidHitAttributes attr)
{
    float t = RayTCurrent();
    float other_t = asfloat(attr.otherT_bits);
    uint ind = PrimitiveIndex();
    uint hitkind = HitKind();

    // Swap if needed based on hit kind
    if (hitkind == 0)
    {
        float temp_t = other_t;
        other_t = t;
        t = temp_t;
    }

    float cur_t = RayTMin();

    // Insert both hits into the buffer
    // Hit kind 1 = entry (min t), Hit kind 0 = exit (max t)
    if (t > cur_t)
    {
        insertHit(payload, t, 2 * ind + 1);  // Entry hit
    }

    if (other_t > cur_t)
    {
        insertHit(payload, other_t, 2 * ind);  // Exit hit
    }

    // Continue traversal if we might find closer hits
    if (t < payload.hits[BUFFER_SIZE - 1].t)
    {
        IgnoreHit();
    }
}

// ============================================================================
// Miss Shader
// ============================================================================

[shader("miss")]
void ms(inout MultiHitPayload payload)
{
    // Nothing to do on miss
}

// ============================================================================
// Ray Generation Shader
// ============================================================================

[shader("raygeneration")]
void rg_float()
{
    uint3 idx = DispatchRaysIndex();
    uint3 dim = DispatchRaysDimensions();

    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];

    SplineState state = make_empty_state();
    state.t = 0;
    state.drgb = initial_drgb[idx.x];

    uint start_id = idx.x * max_iters;

    ControlPoint ctrl_pt = {};
    uint last_tri = 0xFFFFFFFF;
    float prev_t = state.t;
    uint next_tri = 0xFFFFFFFF;
    float next_t = 1e20f;

    uint tri;
    int iter = 0;

    while (state.logT < LOG_CUTOFF && iter < int(max_iters))
    {
        float start_t = abs(state.t);

        // Initialize payload
        MultiHitPayload payload;
        initHitBuffer(payload);

        // Create ray descriptor
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = start_t;
        ray.TMax = tmax;

        // Trace ray
        TraceRay(
            traversable,           // Acceleration structure
            RAY_FLAG_NONE,         // Ray flags
            0xFF,                  // Instance inclusion mask
            0,                     // Ray contribution to hit group index
            1,                     // Multiplier for geometry contribution to hit group index
            0,                     // Miss shader index
            ray,                   // Ray descriptor
            payload                // Payload
        );

        bool endLoop = false;

        for (int i = 0; i < BUFFER_SIZE; i++)
        {
            ctrl_pt.t = payload.hits[i].t;
            tri = payload.hits[i].triIndex;

            if (ctrl_pt.t > 1e9f)
            {
                endLoop = true;
                break;
            }

            ctrl_pt = get_ctrl_pt(tri, ctrl_pt.t);
            state = update(state, ctrl_pt, tmin, tmax, max_prim_size);
            iter++;

            if (!(state.logT < LOG_CUTOFF && iter < int(max_iters)))
            {
                break;
            }
        }

        if (endLoop)
        {
            break;
        }
    }

    SplineOutput output = extract_color(state, tmin);
    image[idx.x] = float4(output.C.x, output.C.y, output.C.z, output.depth);

    SplineState dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = uint(iter);
}
