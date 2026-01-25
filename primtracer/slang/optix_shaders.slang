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
//   hit_id = (hit_type << 28) | prim_idx
//   - Exit (type=0) sorts before Entry (type=1) at same t
//   - This ensures correct density accumulation order
//
// =============================================================================

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// =============================================================================
// Constants
// =============================================================================

#define HIT_TYPE_EXIT 0
#define HIT_TYPE_ENTRY 1
#define HIT_TYPE_VIRTUAL_ENTRY 2  // Entry clamped to cur_t (ray inside ellipsoid)
#define HIT_BUFFER_SIZE 16

static const float LOG_CUTOFF = 5.54f;      // Transmittance cutoff: -log(0.004)
static const float EPS_T = 1e-5f;           // Epsilon for "same t" comparisons
static const float MIN_THICKNESS = 1e-6f;   // Filter grazing intersections

// Hit ID encoding
static const uint HIT_ID_TYPE_SHIFT = 28;
static const uint HIT_ID_PRIM_MASK = (1u << HIT_ID_TYPE_SHIFT) - 1;

uint make_hit_id(uint prim_idx, uint hit_type) {
    return (hit_type << HIT_ID_TYPE_SHIFT) | prim_idx;
}

uint get_prim_idx(uint hit_id) { return hit_id & HIT_ID_PRIM_MASK; }
uint get_hit_type(uint hit_id) { return hit_id >> HIT_ID_TYPE_SHIFT; }

// =============================================================================
// Data Structures
// =============================================================================

struct HitBuffer {
    float t[HIT_BUFFER_SIZE];
    uint hit_id[HIT_BUFFER_SIZE];
    uint skip_threshold;  // Skip hits at trace_t with hit_id < threshold
};

struct HitAttr {
    uint exit_t;  // Bit-cast float via asuint/asfloat
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
RWStructuredBuffer<float> last_delta_contrib;
RWStructuredBuffer<int> iters;
RWStructuredBuffer<int> last_prim;
RWStructuredBuffer<int> prim_hits;
RWStructuredBuffer<int> hit_collection;

// Scalars
int num_prims;
int num_rays;
int sh_degree;
int max_iters;
float tmin;
float max_prim_size;

// Acceleration structure
RaytracingAccelerationStructure traversable;

// =============================================================================
// Helper Functions
// =============================================================================

float3 load_float3(StructuredBuffer<float> buf, uint idx) {
    return float3(buf[idx * 3], buf[idx * 3 + 1], buf[idx * 3 + 2]);
}

float4 load_float4(StructuredBuffer<float> buf, uint idx) {
    return float4(buf[idx * 4], buf[idx * 4 + 1], buf[idx * 4 + 2], buf[idx * 4 + 3]);
}

void store_float4(RWStructuredBuffer<float> buf, uint idx, float4 val) {
    buf[idx * 4 + 0] = val.x;
    buf[idx * 4 + 1] = val.y;
    buf[idx * 4 + 2] = val.z;
    buf[idx * 4 + 3] = val.w;
}

HitBuffer init_hit_buffer(uint skip_threshold) {
    HitBuffer buf;
    buf.skip_threshold = skip_threshold;
    [ForceUnroll] for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
        buf.t[i] = 1e10f;
        buf.hit_id[i] = 0;
    }
    return buf;
}

SamplePoint create_sample(uint hit_id, float t, float3 ray_dir) {
    uint prim_idx = get_prim_idx(hit_id);
    uint hit_type = get_hit_type(hit_id);
    float density = densities[prim_idx];
    float signed_density = (hit_type == HIT_TYPE_ENTRY) ? density : -density;
    float3 color = eval_color(ray_dir, { prim_idx, sh_degree, features });
    return { t, float4(signed_density, signed_density * color) };
}

/// Insert hit maintaining (t, hit_id) order. Uses EPS_T for same-layer grouping.
void insert_hit(inout HitBuffer buf, float t, uint hit_id) {
    [ForceUnroll] for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
        bool same_t = abs(t - buf.t[i]) <= EPS_T;
        bool should_swap = (t < buf.t[i] - EPS_T) || (same_t && hit_id < buf.hit_id[i]);
        if (should_swap) {
            float tmp_t = buf.t[i]; buf.t[i] = t; t = tmp_t;
            uint tmp_id = buf.hit_id[i]; buf.hit_id[i] = hit_id; hit_id = tmp_id;
        }
    }
}

// =============================================================================
// OptiX Shaders
// =============================================================================

[shader("intersection")]
void intersect_ellipsoid() {
    uint prim_idx = PrimitiveIndex();
    float3 ray_dir = WorldRayDirection();
    float3 ray_origin = WorldRayOrigin();
    float cur_t = RayTMin();

    float3 mean = load_float3(means, prim_idx);
    float3 scale = load_float3(scales, prim_idx);
    float4 quat = load_float4(quats, prim_idx);

    float2 t = ray_intersect_ellipsoid(ray_origin - mean, ray_dir, scale, quat);

    // Skip if exit is behind cur_t (use < to allow exit at exactly cur_t)
    if (t.y < cur_t) return;

    // Filter grazing hits by TRUE thickness (not clamped)
    if (t.y - t.x < MIN_THICKNESS) return;

    // Determine if this is a real entry or virtual entry (ray inside ellipsoid)
    // Virtual entries are skipped during retrace to avoid double-counting density
    bool is_virtual = (t.x < cur_t);
    float t_entry = is_virtual ? cur_t : t.x;
    uint hit_kind = is_virtual ? HIT_TYPE_VIRTUAL_ENTRY : HIT_TYPE_ENTRY;

    ReportHitOptix(t_entry, hit_kind, asuint(t.y));
}

[shader("closesthit")]
void closest_hit(inout HitBuffer buf, in HitAttr attr) {}

[shader("anyhit")]
void collect_hits(inout HitBuffer buf, in HitAttr attr) {
    uint prim_idx = PrimitiveIndex();
    float cur_t = RayTMin();
    float t_entry = RayTCurrent();
    float t_exit = asfloat(attr.exit_t);
    uint hit_kind = HitKind();

    uint id_entry = make_hit_id(prim_idx, HIT_TYPE_ENTRY);
    uint id_exit = make_hit_id(prim_idx, HIT_TYPE_EXIT);

    // Skip entry only if already processed (at cur_t with hit_id < skip_threshold)
    // Virtual entries NOT processed in first trace (buffer overflow) will have
    // hit_id >= skip_threshold and will be collected in retrace
    bool at_cur_entry = (t_entry <= cur_t + EPS_T);
    bool skip_entry = (at_cur_entry && id_entry < buf.skip_threshold);

    if (!skip_entry)
        insert_hit(buf, t_entry, id_entry);

    // Exit is always collected (if not already processed)
    bool at_cur_exit = (t_exit <= cur_t + EPS_T);
    bool skip_exit = (at_cur_exit && id_exit < buf.skip_threshold);

    if (t_exit >= cur_t && !skip_exit)
        insert_hit(buf, t_exit, id_exit);

    IgnoreHit();
}

[shader("miss")]
void miss(inout HitBuffer buf) {}

[shader("raygeneration")]
void render_volume() {
    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();
    const uint ray_idx = idx.x;

    float3 ray_dir = l2_normalize(load_float3(ray_directions, ray_idx));
    float3 ray_origin = load_float3(ray_origins, ray_idx);
    float ray_tmax = tmax[ray_idx];

    IntegratorState state = {};
    state.t = max(tmin, 0.0f);

    SamplePoint sample = {};
    uint hit_id = 0;
    int iter = 0;

    // trace_t: current layer's t value (used as TraceRay tmin)
    // skip_threshold: skip hit_id < threshold at trace_t
    float trace_t = state.t;
    uint skip_threshold = 0;

    while (state.logT < LOG_CUTOFF && iter < max_iters) {
        HitBuffer buf = init_hit_buffer(skip_threshold);
        TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 0, 0,
                 { ray_origin, trace_t, ray_dir, ray_tmax }, buf);

        bool done = false;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            float hit_t = buf.t[i];
            if (hit_t > 1e9f) { done = true; break; }

            hit_id = buf.hit_id[i];

            // Layer transition: update trace_t BEFORE creating sample
            if (hit_t > trace_t + EPS_T) {
                trace_t = hit_t;
                skip_threshold = hit_id + 1;
            } else {
                skip_threshold = max(skip_threshold, hit_id + 1);
            }

            // Quantize sample.t to trace_t (same-layer hits share t)
            sample = create_sample(hit_id, trace_t, ray_dir);
            state = integrate_step(state, sample, tmin, ray_tmax, max_prim_size);

            // Record for backward pass
            prim_hits[get_prim_idx(hit_id)]++;
            hit_collection[ray_idx + iter * dim.x] = hit_id;
            iter++;

            if (!(state.logT < LOG_CUTOFF && iter < max_iters)) break;
        }
        if (done) break;
    }

    // Output
    let output = extract_color(state, tmin);
    store_float4(image, ray_idx, float4(output.C, output.alpha));
    depth[ray_idx] = output.depth;

    // Save for backward pass
    last_state[ray_idx] = state;
    store_float4(last_delta_contrib, ray_idx, sample.delta_contrib);
    last_prim[ray_idx] = hit_id;
    iters[ray_idx] = iter;
}
