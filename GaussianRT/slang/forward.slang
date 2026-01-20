// Forward ray tracing shaders for GaussianRT
// Implements differentiable volume rendering with ellipsoid primitives

import common;

// ============================================================================
// Shader Parameters (Uniforms)
// ============================================================================

struct RenderUniforms {
    uint width;
    uint height;
    float t_min;
    float t_max;
    uint max_samples;
    uint sh_degree;
    float3 background_color;
    uint num_elements;
};

// Scene data buffers
StructuredBuffer<float3> positions;      // [N] Element centers
StructuredBuffer<float3> scales;         // [N] Element scales
StructuredBuffer<float4> rotations;      // [N] Element rotations
StructuredBuffer<float> opacities;       // [N] Element opacities
StructuredBuffer<float> features;        // [N * feature_dim] SH coefficients

// Ray data
StructuredBuffer<float3> ray_origins;    // [num_rays]
StructuredBuffer<float3> ray_directions; // [num_rays]

// Output buffers
RWStructuredBuffer<float4> output_colors;        // [num_rays] RGBA
RWStructuredBuffer<RenderState> output_states;   // [num_rays] Final states
RWStructuredBuffer<ControlPoint> output_last_points;  // [num_rays]
RWStructuredBuffer<uint> output_sample_counts;   // [num_rays]
RWStructuredBuffer<int> output_sample_indices;   // [num_rays * max_samples]
RWStructuredBuffer<uint> element_touch_counts;   // [N] Per-element hits

// Acceleration structure
RaytracingAccelerationStructure scene_bvh;

// Uniforms
ConstantBuffer<RenderUniforms> uniforms;

// ============================================================================
// Ray Payload
// ============================================================================

static const uint HIT_BUFFER_SIZE = 16;

struct RayPayload {
    // Hit buffer: stores sorted list of closest hits
    float hit_t[HIT_BUFFER_SIZE];
    uint hit_element_id[HIT_BUFFER_SIZE];
    float hit_t_exit[HIT_BUFFER_SIZE];
    uint hit_count;

    // Ray state
    RenderState state;
    uint sample_count;
    bool terminated;
};

// ============================================================================
// Ellipsoid Ray Intersection
// ============================================================================

// Compute ray-ellipsoid intersection
// Returns (t_near, t_far) or (-1, -1) if no intersection
[Differentiable]
float2 ray_ellipsoid_intersect(
    float3 ray_origin,
    float3 ray_direction,
    float3 center,
    float3 scale,
    float4 rotation
) {
    // Transform ray to ellipsoid local coordinates
    float3 local_origin = ray_origin - center;
    local_origin = rotate_by_quaternion_inverse(local_origin, rotation);

    float3 local_direction = rotate_by_quaternion_inverse(ray_direction, rotation);

    // Scale to unit sphere
    float3 scaled_origin = local_origin / scale;
    float3 scaled_direction = local_direction / scale;

    // Solve quadratic: |o + t*d|^2 = 1
    float a = dot(scaled_direction, scaled_direction);
    float b = 2.0f * dot(scaled_origin, scaled_direction);
    float c = dot(scaled_origin, scaled_origin) - 1.0f;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return float2(-1.0f, -1.0f);
    }

    float sqrt_disc = sqrt(discriminant);
    float t_near = (-b - sqrt_disc) / (2.0f * a);
    float t_far = (-b + sqrt_disc) / (2.0f * a);

    return float2(t_near, t_far);
}

// ============================================================================
// Intersection Shader
// ============================================================================

struct IntersectionAttributes {
    float t_far;  // Exit distance (entry distance is reported automatically)
};

[shader("intersection")]
void intersection_shader() {
    uint element_id = PrimitiveIndex();

    float3 ray_origin = WorldRayOrigin();
    float3 ray_direction = WorldRayDirection();

    float3 center = positions[element_id];
    float3 scale = scales[element_id];
    float4 rotation = rotations[element_id];

    float2 t_values = ray_ellipsoid_intersect(
        ray_origin, ray_direction, center, scale, rotation
    );

    if (t_values.x > 0.0f && t_values.x < RayTCurrent()) {
        IntersectionAttributes attribs;
        attribs.t_far = t_values.y;
        ReportHit(t_values.x, 0, attribs);
    }
}

// ============================================================================
// Any-Hit Shader
// ============================================================================

[shader("anyhit")]
void anyhit_shader(inout RayPayload payload, IntersectionAttributes attribs) {
    if (payload.hit_count >= HIT_BUFFER_SIZE) {
        // Buffer full, ignore if this hit is farther than the farthest stored hit
        if (RayTCurrent() >= payload.hit_t[HIT_BUFFER_SIZE - 1]) {
            IgnoreHit();
            return;
        }
    }

    float t_hit = RayTCurrent();
    uint element_id = PrimitiveIndex();
    float t_exit = attribs.t_far;

    // Insert into sorted buffer
    uint insert_pos = payload.hit_count;
    for (uint i = 0; i < payload.hit_count; ++i) {
        if (t_hit < payload.hit_t[i]) {
            insert_pos = i;
            break;
        }
    }

    // Shift elements to make room
    if (insert_pos < HIT_BUFFER_SIZE) {
        uint shift_count = min(payload.hit_count - insert_pos, HIT_BUFFER_SIZE - insert_pos - 1);
        for (uint i = shift_count; i > 0; --i) {
            uint dst = insert_pos + i;
            uint src = insert_pos + i - 1;
            if (dst < HIT_BUFFER_SIZE) {
                payload.hit_t[dst] = payload.hit_t[src];
                payload.hit_element_id[dst] = payload.hit_element_id[src];
                payload.hit_t_exit[dst] = payload.hit_t_exit[src];
            }
        }

        // Insert new hit
        payload.hit_t[insert_pos] = t_hit;
        payload.hit_element_id[insert_pos] = element_id;
        payload.hit_t_exit[insert_pos] = t_exit;

        payload.hit_count = min(payload.hit_count + 1, HIT_BUFFER_SIZE);
    }

    // Always ignore the hit to continue traversal
    IgnoreHit();
}

// ============================================================================
// Miss Shader
// ============================================================================

[shader("miss")]
void miss_shader(inout RayPayload payload) {
    // No more hits to find, payload already contains all collected hits
}

// ============================================================================
// Ray Generation Shader
// ============================================================================

// Load SH coefficients for an element
void load_sh_coefficients(uint element_id, out float3 coeffs[SH_COEFF_COUNT]) {
    uint base_offset = element_id * SH_COEFF_COUNT * 3;
    for (uint i = 0; i < SH_COEFF_COUNT; ++i) {
        uint offset = base_offset + i * 3;
        coeffs[i] = float3(
            features[offset + 0],
            features[offset + 1],
            features[offset + 2]
        );
    }
}

[shader("raygeneration")]
void raygen_shader() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint ray_index = pixel.y * uniforms.width + pixel.x;

    float3 ray_origin = ray_origins[ray_index];
    float3 ray_direction = ray_directions[ray_index];

    // Initialize payload
    RayPayload payload;
    payload.hit_count = 0;
    payload.state = init_render_state(uniforms.t_min);
    payload.sample_count = 0;
    payload.terminated = false;

    // Ray descriptor
    RayDesc ray;
    ray.Origin = ray_origin;
    ray.Direction = ray_direction;
    ray.TMin = uniforms.t_min;
    ray.TMax = uniforms.t_max;

    // Main rendering loop
    uint iteration = 0;
    while (!payload.terminated && iteration < uniforms.max_samples) {
        // Reset hit buffer
        payload.hit_count = 0;

        // Trace ray to collect hits
        TraceRay(
            scene_bvh,
            RAY_FLAG_NONE,
            0xFF,
            0,  // Hit group index
            1,  // Hit group stride
            0,  // Miss shader index
            ray,
            payload
        );

        // Process collected hits
        for (uint i = 0; i < payload.hit_count && !payload.terminated; ++i) {
            float t_hit = payload.hit_t[i];
            uint element_id = payload.hit_element_id[i];
            float t_exit = payload.hit_t_exit[i];

            // Record sample
            if (payload.sample_count < uniforms.max_samples) {
                output_sample_indices[ray_index * uniforms.max_samples + payload.sample_count] = element_id;
                payload.sample_count++;
            }

            // Update element touch count
            InterlockedAdd(element_touch_counts[element_id], 1);

            // Load element data
            float opacity = opacities[element_id];
            float3 coeffs[SH_COEFF_COUNT];
            load_sh_coefficients(element_id, coeffs);

            // Evaluate color using spherical harmonics
            float3 color = eval_sh_color(-ray_direction, coeffs, uniforms.sh_degree);

            // Create control point
            // Using midpoint of intersection as sample point
            float t_sample = (t_hit + t_exit) * 0.5f;

            ControlPoint point;
            point.t = t_sample;
            point.sigma = opacity;
            point.color = color;

            // Update render state
            payload.state = update_render_state(payload.state, point);

            // Check for early termination
            if (should_terminate(payload.state)) {
                payload.terminated = true;
            }
        }

        // Update ray for next iteration
        if (payload.hit_count > 0 && !payload.terminated) {
            float last_t = payload.hit_t_exit[payload.hit_count - 1];
            ray.TMin = last_t + 1e-4f;
        }

        iteration++;
    }

    // Finalize and write outputs
    float4 final_color = finalize_color(payload.state, uniforms.background_color);
    output_colors[ray_index] = final_color;
    output_states[ray_index] = payload.state;
    output_sample_counts[ray_index] = payload.sample_count;

    // Store last control point (for backward pass)
    if (payload.sample_count > 0) {
        uint last_element = output_sample_indices[ray_index * uniforms.max_samples + payload.sample_count - 1];
        ControlPoint last_point;
        last_point.t = payload.state.t_current;
        last_point.sigma = opacities[last_element];

        float3 coeffs[SH_COEFF_COUNT];
        load_sh_coefficients(last_element, coeffs);
        last_point.color = eval_sh_color(-ray_direction, coeffs, uniforms.sh_degree);

        output_last_points[ray_index] = last_point;
    }
}
