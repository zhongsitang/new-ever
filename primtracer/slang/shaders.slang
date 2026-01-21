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
//
// This shader implements differentiable volume rendering using OptiX ray tracing.
// Rays are traced through ellipsoid primitives, and intersections are sorted
// and processed to compute the volume rendering integral.
//
// Hit types:
//   - ENTRY (1): Ray enters the ellipsoid (near intersection)
//   - EXIT  (0): Ray exits the ellipsoid (far intersection)
//
// Triangle indexing:
//   tri_idx = prim_idx * 2 + hit_type
//   - Even indices (tri_idx % 2 == 0): exit faces
//   - Odd indices  (tri_idx % 2 == 1): entry faces
// =============================================================================

import spline_machine;
import ellipsoid;
import safe_math;
import sh;

// Transmittance cutoff: stop tracing when -log(T) exceeds this value
// (corresponds to T < exp(-5.54) ≈ 0.004)
#define LOG_TRANSMITTANCE_CUTOFF 5.54

// Maximum number of hits to buffer per trace call
#define HIT_BUFFER_SIZE 16

// Hit type constants
static const uint HIT_TYPE_EXIT = 0;
static const uint HIT_TYPE_ENTRY = 1;

// =============================================================================
// Data Structures
// =============================================================================

struct Camera {
    float fx, fy;           // Focal lengths
    int height, width;      // Image dimensions
    float3 U, V, W;         // Camera basis vectors
    float3 eye;             // Camera position
};

// Sorted buffer of ray-primitive intersections
struct SortedHitBuffer {
    float t[HIT_BUFFER_SIZE];       // Sorted intersection distances
    uint tri_idx[HIT_BUFFER_SIZE];  // Triangle indices (prim_idx * 2 + hit_type)
};

// Intersection attribute passed from intersection shader to hit shaders
// Stores the paired t-value (entry stores exit t, exit stores entry t)
struct IntersectionAttrib {
    uint paired_t_bits;  // Bit representation of float (use asfloat to decode)
};

// =============================================================================
// Launch Parameters (set by host)
// =============================================================================

// Output buffers
RWStructuredBuffer<float4>            image;          // RGBA output
RWStructuredBuffer<uint>              iters;          // Iteration count per ray
RWStructuredBuffer<uint>              last_face;      // Last hit face index
RWStructuredBuffer<uint>              touch_count;    // Hit count per primitive
RWStructuredBuffer<float4>            last_sample;    // Last sample value
RWStructuredBuffer<IntegrationState>  last_state;     // Final integration state
RWStructuredBuffer<int>               tri_collection; // All hit triangles per ray

// Input buffers
StructuredBuffer<float3>              ray_origins;
StructuredBuffer<float3>              ray_directions;
Camera camera;

// Primitive data
RWStructuredBuffer<float3> means;      // Ellipsoid centers
RWStructuredBuffer<float3> scales;     // Ellipsoid axis scales
RWStructuredBuffer<float4> quats;      // Ellipsoid rotations (quaternions)
RWStructuredBuffer<float> densities;   // Peak density values
RWStructuredBuffer<float> features;    // SH coefficients

// Rendering parameters
size_t sh_degree;                      // Spherical harmonics degree
size_t max_iters;                      // Maximum iterations per ray
float tmin;                            // Near clipping distance
float tmax;                            // Far clipping distance
float max_prim_size;                   // Maximum primitive bounding size
RWStructuredBuffer<float4> initial_sample;  // Initial sample for each ray
RaytracingAccelerationStructure traversable;

// =============================================================================
// Helper Functions
// =============================================================================

// Create a Sample from a triangle hit
// tri_idx encodes both primitive index and hit type (entry/exit)
Sample create_sample_from_hit(uint tri_idx, float t) {
    uint prim_idx = tri_idx / tri_per_g;
    uint hit_type = tri_idx % tri_per_g;

    // Get density with sign based on entry/exit
    // Entry (hit_type=1): positive contribution
    // Exit  (hit_type=0): negative contribution
    float density = densities[prim_idx];
    float signed_density = density * ((hit_type == HIT_TYPE_ENTRY) ? 1.0f : -1.0f);

    // Evaluate base color from SH (degree 0 only for speed)
    SHFeatures sh_feats = {prim_idx, sh_degree, features};
    Features feat;
    feat.f0 = get_sh(sh_feats, 0);
    float3 view_dir = float3(0, 0, 1);  // Placeholder direction
    float3 color = eval_sh_col0(view_dir, feat);

    // Pack into sample: (signed_density, signed_density * color)
    Sample sample;
    sample.t = t;
    sample.value = float4(
        signed_density,
        signed_density * color.x,
        signed_density * color.y,
        signed_density * color.z
    );
    return sample;
}

// =============================================================================
// Intersection Shader
// =============================================================================
// Tests ray-ellipsoid intersection and reports hits with both entry and exit t values

[shader("intersection")]
void intersection_ellipsoid() {
    uint prim_idx = PrimitiveIndex();

    float3 ray_dir = WorldRayDirection();
    float3 ray_origin = WorldRayOrigin();

    float3 center = means[prim_idx];
    float3 scale = scales[prim_idx];
    float4 rotation = quats[prim_idx];

    // Compute entry (min) and exit (max) intersection distances
    float2 t_entry_exit = ray_intersect_ellipsoid(ray_origin - center, ray_dir, scale, rotation);
    float t_entry = t_entry_exit.x;
    float t_exit = t_entry_exit.y;

    float ray_t_min = RayTMin();

    // No valid intersection if exit is before ray start
    if (t_exit < ray_t_min) {
        return;
    }

    // Report the relevant hit based on current ray position
    if (ray_t_min < t_entry) {
        // Ray hasn't reached entry yet - report entry hit with exit as paired value
        ReportHitOptix(t_entry, HIT_TYPE_ENTRY, asuint(t_exit));
    } else if (ray_t_min < t_exit) {
        // Ray is inside ellipsoid - report exit hit with entry as paired value
        ReportHitOptix(t_exit, HIT_TYPE_EXIT, asuint(t_entry));
    }
}

// =============================================================================
// Closest Hit Shader (empty - we use any-hit for sorting)
// =============================================================================

[shader("closesthit")]
void closest_hit(inout SortedHitBuffer hit_buffer, in IntersectionAttrib attrib) {
    // All processing done in any-hit shader
}

// =============================================================================
// Any Hit Shader
// =============================================================================
// Maintains a sorted buffer of the closest HIT_BUFFER_SIZE intersections

[shader("anyhit")]
void any_hit(inout SortedHitBuffer hit_buffer, in IntersectionAttrib attrib) {
    float current_t = RayTCurrent();
    float paired_t = asfloat(attrib.paired_t_bits);
    uint prim_idx = PrimitiveIndex();
    uint hit_type = HitKind();

    // Normalize so t_entry < t_exit regardless of which was reported
    float t_entry, t_exit;
    if (hit_type == HIT_TYPE_ENTRY) {
        t_entry = current_t;
        t_exit = paired_t;
    } else {
        t_entry = paired_t;
        t_exit = current_t;
    }

    float ray_t_min = RayTMin();

    // Insert both entry and exit into sorted buffer
    // Process entry first, then exit (smaller t first for proper sorting)
    [unroll]
    for (int hit_idx = 0; hit_idx < 2; hit_idx++) {
        float candidate_t;
        uint candidate_tri;

        if (hit_idx == 0) {
            candidate_t = t_entry;
            candidate_tri = prim_idx * tri_per_g + HIT_TYPE_ENTRY;
        } else {
            candidate_t = t_exit;
            candidate_tri = prim_idx * tri_per_g + HIT_TYPE_EXIT;
        }

        // Skip if before ray start
        if (candidate_t <= ray_t_min) {
            continue;
        }

        // Insertion sort into buffer
        [ForceUnroll]
        for (int slot = 0; slot < HIT_BUFFER_SIZE; slot++) {
            float slot_t = hit_buffer.t[slot];
            if (candidate_t < slot_t) {
                // Swap candidate with slot, continue inserting displaced item
                hit_buffer.t[slot] = candidate_t;
                uint slot_tri = hit_buffer.tri_idx[slot];
                hit_buffer.tri_idx[slot] = candidate_tri;
                candidate_tri = slot_tri;
                candidate_t = slot_t;
            }
        }
    }

    // Continue tracing if entry t is less than the furthest buffered hit
    // (we might find closer hits that should be inserted)
    if (t_entry < hit_buffer.t[HIT_BUFFER_SIZE - 1]) {
        IgnoreHit();
    }
}

// =============================================================================
// Miss Shader
// =============================================================================

[shader("miss")]
void miss(inout SortedHitBuffer hit_buffer) {
    // No action needed - buffer already initialized
}

// =============================================================================
// Ray Generation Shader
// =============================================================================
// Main entry point: traces rays and accumulates volume rendering integral

[shader("raygeneration")]
void raygen() {
    const uint3 ray_idx = DispatchRaysIndex();
    const uint3 ray_count = DispatchRaysDimensions();

    // Initialize ray
    float3 direction = l2_normalize(ray_directions[ray_idx.x]);
    float3 origin = ray_origins[ray_idx.x] + tmin * direction;

    // Initialize integration state
    IntegrationState state = make_empty_state();
    state.t = 0;
    state.sample = initial_sample[ray_idx.x];

    Sample sample = {};
    uint last_hit_tri = 0xFFFFFFFF;  // Invalid sentinel
    int iteration = 0;

    // Main rendering loop
    while (state.log_transmittance < LOG_TRANSMITTANCE_CUTOFF && iteration < max_iters) {
        float trace_start_t = abs(state.t);

        // Initialize hit buffer with sentinel values
        SortedHitBuffer hit_buffer;
        [unroll]
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            hit_buffer.t[i] = 1e10f;
            hit_buffer.tri_idx[i] = 0;
        }

        // Trace ray segment
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = trace_start_t;
        ray.TMax = tmax;

        TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, hit_buffer);

        // Process buffered hits in sorted order
        bool no_more_hits = false;
        [unroll]
        for (int i = 0; i < HIT_BUFFER_SIZE; i++) {
            float hit_t = hit_buffer.t[i];
            uint tri_idx = hit_buffer.tri_idx[i];

            // Check for sentinel (no more valid hits)
            if (hit_t > 1e9f) {
                no_more_hits = true;
                break;
            }

            // Create sample and update integration state
            sample = create_sample_from_hit(tri_idx, hit_t);
            state = update(state, sample, tmin, tmax, max_prim_size);

            // Record hit for backward pass
            uint prim_idx = tri_idx / tri_per_g;
            touch_count[prim_idx]++;
            tri_collection[ray_idx.x + iteration * ray_count.x] = tri_idx;
            iteration++;

            // Early termination check
            if (!(state.log_transmittance < LOG_TRANSMITTANCE_CUTOFF && iteration < max_iters)) {
                break;
            }
        }

        if (no_more_hits) {
            break;
        }
    }

    // Extract final color and write outputs
    RenderOutput output = extract_color(state, tmin);
    image[ray_idx.x] = float4(output.color, output.depth);

    IntegrationState dual_state = to_dual(state, sample);
    last_state[ray_idx.x] = dual_state;
    last_sample[ray_idx.x] = sample.value;
    last_face[ray_idx.x] = last_hit_tri;
    iters[ray_idx.x] = iteration;
}
