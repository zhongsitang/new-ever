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
// OptiX Ray Tracing Shaders for Volume Rendering
// =============================================================================
// Implements ellipsoid primitive ray tracing with sorted hit buffer for
// front-to-back volume integration.

#define RT_EPS 0
#define SURFACES_PER_PRIM 2          // Each primitive has entry and exit surfaces
#define TRANSMITTANCE_CUTOFF 5.54    // log(T) threshold for early termination
#define HIT_BUFFER_SIZE 16           // Sorted hit buffer capacity

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// -----------------------------------------------------------------------------
// Camera Parameters
// -----------------------------------------------------------------------------

struct Camera {
    float fx, fy;           // Focal lengths
    int image_height;       // Image dimensions
    int image_width;
    float3 U, V, W;         // Camera basis vectors
    float3 eye;             // Camera position
};

// -----------------------------------------------------------------------------
// Hit Buffer: sorted list of surface intersections
// -----------------------------------------------------------------------------

struct HitBuffer {
    float t[HIT_BUFFER_SIZE];           // Intersection distances (sorted)
    uint surface_id[HIT_BUFFER_SIZE];   // Surface IDs: prim_idx * 2 + (1=entry, 0=exit)
};

// Intersection attribute: stores the complementary intersection distance
struct IntersectAttr {
    uint other_t_bits;  // asuint(float) of the other intersection t
};

// -----------------------------------------------------------------------------
// Launch Parameters (global buffers)
// -----------------------------------------------------------------------------

// Output buffers
RWStructuredBuffer<float4>      image;
RWStructuredBuffer<uint>        iters;
RWStructuredBuffer<uint>        last_surface;
RWStructuredBuffer<uint>        prim_hit_count;
RWStructuredBuffer<float4>      last_delta;
RWStructuredBuffer<VolumeState> last_state;
RWStructuredBuffer<int>         hit_log;

// Ray parameters
StructuredBuffer<float3>        ray_origins;
StructuredBuffer<float3>        ray_directions;
Camera camera;

// Primitive data
RWStructuredBuffer<float3>      means;
RWStructuredBuffer<float3>      scales;
RWStructuredBuffer<float4>      quats;
RWStructuredBuffer<float>       densities;
RWStructuredBuffer<float>       features;

// Rendering parameters
size_t sh_degree;
size_t max_iters;
float tmin;
float tmax;
RWStructuredBuffer<float4>      initial_accumulated_drgb;
float max_prim_size;
RaytracingAccelerationStructure traversable;

// -----------------------------------------------------------------------------
// Sample Point Construction
// -----------------------------------------------------------------------------

// Create a sample point from a surface hit
// surface_id encodes: prim_idx * 2 + hit_kind (1=entry, 0=exit)
SamplePoint get_sample(uint surface_id, float t) {
    SamplePoint sample;
    let prim_idx = surface_id / SURFACES_PER_PRIM;
    let is_entry = (surface_id % SURFACES_PER_PRIM) == 1;

    // Get density with sign: positive for entry, negative for exit
    let density = densities[prim_idx];
    let signed_density = density * (is_entry ? 1.0f : -1.0f);

    // Evaluate color from spherical harmonics
    Features feat;
    SHFeatures sh_feats = {prim_idx, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    let color = eval_sh_col0(rayd, feat);

    // Build delta contribution: (+-sigma, +-sigma*r, +-sigma*g, +-sigma*b)
    sample.t = t;
    sample.delta.x = signed_density;
    sample.delta.y = signed_density * color.x;
    sample.delta.z = signed_density * color.y;
    sample.delta.w = signed_density * color.z;

    return sample;
}

// -----------------------------------------------------------------------------
// Intersection Shader: Ray-Ellipsoid Intersection
// -----------------------------------------------------------------------------

[shader("intersection")]
void ellipsoid() {
    uint prim_idx = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    let mean = means[prim_idx];
    let scale = scales[prim_idx];
    let quat = quats[prim_idx];

    // Compute entry/exit t values
    float2 t_entry_exit = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();
    if (t_entry_exit.y < cur_t) {
        return;  // Ray starts after ellipsoid
    }

    // Report the first valid intersection
    bool hit_entry = cur_t < t_entry_exit.x;
    if (hit_entry) {
        // Entry hit: report entry t, store exit t as attribute
        ReportHitOptix(t_entry_exit.x, 1u, asuint(t_entry_exit.y));
    } else if (cur_t < t_entry_exit.y) {
        // Exit hit: report exit t, store entry t as attribute
        ReportHitOptix(t_entry_exit.y, 0u, asuint(t_entry_exit.x));
    }
}

// -----------------------------------------------------------------------------
// Closest Hit Shader (unused but required)
// -----------------------------------------------------------------------------

[shader("closesthit")]
void closesthit(inout HitBuffer buf, in IntersectAttr attr)
{
}

// -----------------------------------------------------------------------------
// Any Hit Shader: Insert hits into sorted buffer
// -----------------------------------------------------------------------------

[shader("anyhit")]
void anyhit(inout HitBuffer buf, in IntersectAttr attr)
{
    float t = RayTCurrent();
    float other_t = asfloat(attr.other_t_bits);
    uint prim_idx = PrimitiveIndex();
    uint hit_kind = HitKind();

    // Ensure t is entry and other_t is exit
    if (hit_kind == 0) {
        float temp = other_t;
        other_t = t;
        t = temp;
    }

    float cur_t = RayTMin();

    // Insert both entry and exit into sorted buffer
    for (int n = 0; n < 2; n++) {
        float hit_t;
        uint surface_id;

        if (n == 0) {
            hit_t = t;
            surface_id = 2 * prim_idx + 1;  // Entry surface
        } else {
            hit_t = other_t;
            surface_id = 2 * prim_idx;      // Exit surface
        }

        if (hit_t > cur_t) {
            // Insertion sort into buffer
            [ForceUnroll]
            for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
                float test_t = buf.t[i];
                if (hit_t < test_t) {
                    buf.t[i] = hit_t;
                    uint test_id = buf.surface_id[i];
                    buf.surface_id[i] = surface_id;
                    surface_id = test_id;
                    hit_t = test_t;
                }
            }
        }
    }

    // Continue tracing if buffer not full with closer hits
    if (t < buf.t[HIT_BUFFER_SIZE - 1]) {
        IgnoreHit();
    }
}

// -----------------------------------------------------------------------------
// Miss Shader
// -----------------------------------------------------------------------------

[shader("miss")]
void miss(inout HitBuffer buf)
{
}

// -----------------------------------------------------------------------------
// Ray Generation Shader: Main rendering entry point
// -----------------------------------------------------------------------------

[shader("raygeneration")]
void rg_float()
{
    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();

    // Initialize ray
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];

    // Initialize volume state
    VolumeState state = make_initial_state();
    state.t = 0;
    state.accumulated_drgb = initial_accumulated_drgb[idx.x];

    SamplePoint sample = {};
    uint prev_surface = -1;
    int iter = 0;

    // Main ray marching loop
    while (state.log_transmittance < TRANSMITTANCE_CUTOFF && iter < max_iters)
    {
        let start_t = abs(state.t);

        // Initialize hit buffer with sentinel values
        HitBuffer buf;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            buf.t[i] = 1e10f;
            buf.surface_id[i] = 0;
        }

        // Trace ray
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = start_t;
        ray.TMax = tmax;

        TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, buf);

        // Process hits from buffer
        bool ray_escaped = false;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            sample.t = buf.t[i];
            uint surface_id = buf.surface_id[i];

            if (sample.t > 1e9) {
                ray_escaped = true;
                break;
            }

            // Integrate sample into volume state
            sample = get_sample(surface_id, sample.t);
            state = update(state, sample, tmin, tmax, max_prim_size);

            // Record hit
            prim_hit_count[surface_id / SURFACES_PER_PRIM]++;
            hit_log[idx.x + iter * dim.x] = surface_id;
            iter++;

            if (!(state.log_transmittance < TRANSMITTANCE_CUTOFF && iter < max_iters)) break;
        }
        if (ray_escaped) break;
    }

    // Extract final color and write outputs
    let output = extract_color(state, tmin);
    image[idx.x] = {output.C.x, output.C.y, output.C.z, output.depth};

    let dual_state = to_dual(state, sample);
    last_state[idx.x] = dual_state;
    last_delta[idx.x] = sample.delta;
    last_surface[idx.x] = prev_surface;
    iters[idx.x] = iter;
}
