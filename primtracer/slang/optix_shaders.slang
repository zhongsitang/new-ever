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
// This module implements OptiX ray tracing shaders for volume rendering of
// ellipsoid primitives. The rendering algorithm works as follows:
//
// 1. Each ellipsoid primitive generates TWO hit events per ray intersection:
//    - Entry hit (hit_type=1): where the ray enters the ellipsoid
//    - Exit hit (hit_type=0): where the ray exits the ellipsoid
//
// 2. Hit IDs encode both primitive index and hit type:
//    hit_id = prim_index * HITS_PER_PRIMITIVE + hit_type
//    prim_index = hit_id / HITS_PER_PRIMITIVE
//    hit_type = hit_id % HITS_PER_PRIMITIVE
//
// 3. Volume rendering accumulates density contributions:
//    - At entry: add +density to the accumulated density field
//    - At exit: add -density to the accumulated density field
//    The integral of this piecewise-constant function gives proper volume opacity.
//
// =============================================================================

#define RT_EPS 0
#define HITS_PER_PRIMITIVE 2    // Each ellipsoid has 2 hits: entry and exit
#define HIT_TYPE_EXIT 0         // Exit point hit type
#define HIT_TYPE_ENTRY 1        // Entry point hit type
#define LOG_CUTOFF 5.54         // Transmittance cutoff (-log(0.004))
#define HIT_BUFFER_SIZE 16      // Size of sorted hit buffer

import volume_integrator;
import ellipsoid_intersect;
import safe_math;
import spherical_harmonics;

// =============================================================================
// Data Structures
// =============================================================================

struct Camera {
    float fx, fy;
    int height;
    int width;
    float4 U, V, W;  // xyz = direction, w = padding
    float4 eye;      // xyz = position, w = padding
};

/// Buffer for collecting and sorting ray-primitive intersections.
/// Maintains hits sorted by t value for front-to-back traversal.
struct HitBuffer {
    float t[HIT_BUFFER_SIZE];       // Hit distances along ray
    uint hit_id[HIT_BUFFER_SIZE];   // Encoded hit IDs (prim_index * 2 + hit_type)
};

/// Intersection attribute passed between intersection and hit shaders.
struct HitAttr {
    uint other_t;  // The other intersection t value (as uint bits)
};

// =============================================================================
// Launch Parameters (set by host via cudaMemcpy, must match C++ Params layout)
// =============================================================================

struct Params {
    // Output buffers
    RWStructuredBuffer<float4>          image;
    RWStructuredBuffer<float>           depth_out;
    RWStructuredBuffer<uint>            iters;
    RWStructuredBuffer<uint>            last_prim;
    RWStructuredBuffer<uint>            prim_hits;
    RWStructuredBuffer<float4>          last_delta_contrib;
    RWStructuredBuffer<IntegratorState> last_state;
    RWStructuredBuffer<int>             hit_collection;

    // Ray data
    StructuredBuffer<float3>            ray_origins;
    StructuredBuffer<float3>            ray_directions;
    Camera                              camera;

    // Primitive data
    RWStructuredBuffer<float3>          means;
    RWStructuredBuffer<float3>          scales;
    RWStructuredBuffer<float4>          quats;
    RWStructuredBuffer<float>           densities;
    RWStructuredBuffer<float>           features;

    // Render settings
    uint                                sh_degree;
    uint                                max_iters;
    float                               tmin;
    StructuredBuffer<float>             tmax;
    RWStructuredBuffer<float4>          initial_contrib;
    float                               max_prim_size;
    uint                                _pad_traversable;
    RaytracingAccelerationStructure     traversable;
};

// Global launch parameters (name must match pipelineLaunchParamsVariableName)
Params SLANG_globalParams;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a SamplePoint from a hit ID and t value.
///
/// Decodes the hit_id to get primitive index and hit type, then evaluates
/// the color and computes the delta contribution.
SamplePoint create_sample(uint hit_id, float t) {
    let params = SLANG_globalParams;
    SamplePoint sample;
    let prim_index = hit_id / HITS_PER_PRIMITIVE;
    let hit_type = hit_id % HITS_PER_PRIMITIVE;

    // Get density and compute signed contribution
    let density = params.densities[prim_index];
    let signed_density = density * ((hit_type == HIT_TYPE_ENTRY) ? 1.0f : -1.0f);

    // Evaluate base color from SH degree 0
    Features feat;
    SHFeatures sh_feats = {prim_index, params.sh_degree, params.features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};  // Default direction for SH evaluation
    let color = eval_sh_col0(rayd, feat);

    // Pack into sample
    sample.t = t;
    sample.delta_contrib.x = signed_density;
    sample.delta_contrib.y = signed_density * color.x;
    sample.delta_contrib.z = signed_density * color.y;
    sample.delta_contrib.w = signed_density * color.z;

    return sample;
}

// =============================================================================
// OptiX Shaders
// =============================================================================

/// Intersection shader: computes ray-ellipsoid intersection.
///
/// For each ellipsoid, computes both entry and exit t values.
/// Reports the appropriate hit based on current ray t minimum.
[shader("intersection")]
void intersect_ellipsoid() {
    let params = SLANG_globalParams;
    uint prim_index = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    let mean = params.means[prim_index];
    let scale = params.scales[prim_index];
    let quat = params.quats[prim_index];
    float2 t_entry_exit = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();
    if (t_entry_exit.y < cur_t) {
        // Both intersections are behind the ray start
        return;
    } else {
        bool use_entry = cur_t < t_entry_exit.x;
        if (use_entry) {
            // Report entry point, pass exit t as attribute
            ReportHitOptix(t_entry_exit.x, HIT_TYPE_ENTRY, asuint(t_entry_exit.y));
        } else if (cur_t < t_entry_exit.y) {
            // Report exit point, pass entry t as attribute
            ReportHitOptix(t_entry_exit.y, HIT_TYPE_EXIT, asuint(t_entry_exit.x));
        }
    }
}

/// Closest hit shader (currently unused, hits are processed in any-hit).
[shader("closesthit")]
void closest_hit(inout HitBuffer buf, in HitAttr attr)
{
}

/// Any-hit shader: collects and sorts hits into the hit buffer.
///
/// This shader is called for every potential hit along the ray.
/// It inserts both the current hit and its paired hit (entry/exit)
/// into a sorted buffer, maintaining front-to-back order.
[shader("anyhit")]
void collect_hits(inout HitBuffer buf, in HitAttr attr)
{
    float t = RayTCurrent();
    float other_t = asfloat(attr.other_t);
    uint prim_index = PrimitiveIndex();
    uint hit_type = HitKind();

    // Ensure t_entry < t_exit for proper ordering
    float t_entry, t_exit;
    if (hit_type == HIT_TYPE_EXIT) {
        t_entry = other_t;
        t_exit = t;
    } else {
        t_entry = t;
        t_exit = other_t;
    }

    float cur_t = RayTMin();

    // Insert both entry and exit hits into the sorted buffer
    float insert_t;
    uint insert_hit_id;

    for (int n = 0; n < 2; n++) {
        if (n == 0) {
            insert_t = t_entry;
            insert_hit_id = prim_index * HITS_PER_PRIMITIVE + HIT_TYPE_ENTRY;
        } else {
            insert_t = t_exit;
            insert_hit_id = prim_index * HITS_PER_PRIMITIVE + HIT_TYPE_EXIT;
        }

        // Only insert hits that are in front of the ray start
        if (insert_t > cur_t) {
            // Insertion sort into the buffer
            [ForceUnroll]
            for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
                float buf_t = buf.t[i];
                if (insert_t < buf_t) {
                    // Swap and continue with the displaced element
                    buf.t[i] = insert_t;
                    uint buf_hit_id = buf.hit_id[i];
                    buf.hit_id[i] = insert_hit_id;
                    insert_hit_id = buf_hit_id;
                    insert_t = buf_t;
                }
            }
        }
    }

    // Continue traversal if the buffer might not contain all relevant hits
    if (t_entry < buf.t[HIT_BUFFER_SIZE - 1]) {
        IgnoreHit();
    }
}

/// Miss shader (called when ray hits nothing).
[shader("miss")]
void miss(inout HitBuffer buf)
{
}

/// Ray generation shader: main entry point for rendering.
///
/// For each pixel/ray:
/// 1. Initialize volume rendering state
/// 2. Trace ray through scene, collecting sorted hits
/// 3. Process hits front-to-back, updating volume state
/// 4. Output final color
[shader("raygeneration")]
void render_volume()
{
    let params = SLANG_globalParams;
    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();

    // Initialize ray
    float3 direction = l2_normalize(params.ray_directions[idx.x]);
    float3 origin = params.ray_origins[idx.x] + params.tmin * params.ray_directions[idx.x];
    float ray_tmax = params.tmax[idx.x];  // Per-ray tmax

    // Initialize volume rendering state
    IntegratorState state = make_initial_state();
    state.t = 0;
    state.accumulated_contrib = params.initial_contrib[idx.x];

    SamplePoint sample = {};
    uint last_hit_id = -1;

    uint hit_id;

    int iter = 0;
    while (state.logT < LOG_CUTOFF && iter < params.max_iters)
    {
        let start_t = abs(state.t);

        // Initialize hit buffer with large t values
        HitBuffer buf;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            buf.t[i] = 1e10f;
            buf.hit_id[i] = 0;
        }

        // Trace ray to collect hits
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = start_t;
        ray.TMax = ray_tmax;

        TraceRay(params.traversable, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, buf);

        // Process collected hits front-to-back
        bool end = false;
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            sample.t = buf.t[i];
            hit_id = buf.hit_id[i];

            if (sample.t > 1e9) {
                // No more valid hits
                end = true;
                break;
            }

            // Create sample and update volume state
            sample = create_sample(hit_id, sample.t);
            state = update(state, sample, params.tmin, ray_tmax, params.max_prim_size);

            // Record hit for backward pass
            params.prim_hits[hit_id / HITS_PER_PRIMITIVE]++;
            params.hit_collection[idx.x + iter * dim.x] = hit_id;
            iter++;

            if (!(state.logT < LOG_CUTOFF && iter < params.max_iters)) break;
        }
        if (end) break;
    }

    // Extract and output final color (RGBA) and depth
    let output = extract_color(state, params.tmin);
    params.image[idx.x] = {output.C.x, output.C.y, output.C.z, output.alpha};
    params.depth_out[idx.x] = output.depth;

    // Save state for backward pass
    let dual_state = to_dual(state, sample);
    params.last_state[idx.x] = dual_state;
    params.last_delta_contrib[idx.x] = sample.delta_contrib;
    params.last_prim[idx.x] = last_hit_id;
    params.iters[idx.x] = iter;
}
