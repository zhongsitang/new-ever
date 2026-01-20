// Modern Splinetracer - Main Ray Tracing Shaders
// Hardware ray tracing pipeline for volume rendering
// Apache License 2.0

// Constants
#define BUFFER_SIZE 16      // Number of hits to accumulate before processing
#define LOG_CUTOFF 5.54f    // Transmittance cutoff for early termination

// Import modules
import safe_math;
import sh;
import volume;
import ellipsoid;

// ============================================================================
// Shader Resources
// ============================================================================

// Camera parameters
struct Camera {
    float fx, fy;       // Focal lengths
    int width, height;  // Image dimensions
    float2 _pad0;
    float3 U, V, W;     // Basis vectors
    float3 eye;         // Position
    float _pad1;
};

// Output buffers (read-write)
RWStructuredBuffer<float4> image;
RWStructuredBuffer<uint> iters;
RWStructuredBuffer<uint> last_face;
RWStructuredBuffer<uint> touch_count;
RWStructuredBuffer<float4> last_dirac;
RWStructuredBuffer<VolumeState> last_state;

// Input ray buffers
StructuredBuffer<float3> ray_origins;
StructuredBuffer<float3> ray_directions;

// Primitive data buffers
RWStructuredBuffer<float3> means;
RWStructuredBuffer<float3> scales;
RWStructuredBuffer<float4> quats;
RWStructuredBuffer<float> densities;
RWStructuredBuffer<float> features;

// Uniform parameters
Camera camera;
uint sh_degree;
uint max_iters;
float tmin;
float tmax;
float max_prim_size;

// Initial state
RWStructuredBuffer<float4> initial_drgb;

// Acceleration structure
RaytracingAccelerationStructure traversable;

// ============================================================================
// Ray Payload and Hit Attributes
// ============================================================================

// Payload carries sorted hit list
struct [raypayload] RayPayload {
    uint data[2 * BUFFER_SIZE];  // Interleaved (t_as_uint, primitive_id)
};

// Custom hit attributes for ellipsoid intersection
struct EllipsoidHitAttributes {
    float2 t_pair;  // (t_enter, t_exit)
};

// ============================================================================
// Helper Functions
// ============================================================================

// Get volume sample from hit information
VolumeSample get_volume_sample(uint tri, float t) {
    VolumeSample sample;

    uint prim_id = tri / 2;
    uint hit_kind = tri % 2;  // 0 = exit, 1 = enter

    float density = densities[prim_id];
    float dirac_density = density * ((hit_kind == 1) ? 1.0f : -1.0f);

    // Get color from spherical harmonics
    SHFeatures sh_feats;
    sh_feats.prim_index = prim_id;
    sh_feats.degree = sh_degree;
    sh_feats.features = features;

    Features feat;
    feat.f0 = get_sh(sh_feats, 0);

    float3 ray_dir = float3(0.0f, 0.0f, 1.0f);  // Simplified for now
    float3 color = eval_sh_col0(ray_dir, feat);

    sample.t = t;
    sample.dirac = float4(
        dirac_density,
        dirac_density * color.x,
        dirac_density * color.y,
        dirac_density * color.z
    );

    return sample;
}

// ============================================================================
// Intersection Shader
// ============================================================================

[shader("intersection")]
void intersectionShader() {
    uint prim_id = PrimitiveIndex();

    float3 ray_dir = WorldRayDirection();
    float3 ray_origin = WorldRayOrigin();

    float3 mean = means[prim_id];
    float3 scale = scales[prim_id];
    float4 quat = quats[prim_id];

    // Compute ray-ellipsoid intersection
    float2 t_pair = ray_intersect_ellipsoid(ray_origin - mean, ray_dir, scale, quat);

    float cur_t = RayTMin();
    if (t_pair.y < cur_t) {
        return;  // No valid intersection
    }

    EllipsoidHitAttributes attrs;
    attrs.t_pair = t_pair;

    // Report the appropriate hit
    bool use_min = cur_t < t_pair.x;
    if (use_min) {
        ReportHit(t_pair.x, 1u, attrs);  // Enter hit
    } else if (cur_t < t_pair.y) {
        ReportHit(t_pair.y, 0u, attrs);  // Exit hit
    }
}

// ============================================================================
// Any Hit Shader
// ============================================================================

[shader("anyhit")]
void anyHitShader(inout RayPayload payload, in EllipsoidHitAttributes attrs : SV_IntersectionAttributes) {
    float t = RayTCurrent();
    uint prim_id = PrimitiveIndex();
    uint hit_kind = HitKind();

    // Get the other intersection point
    float other_t = (hit_kind == 1) ? attrs.t_pair.y : attrs.t_pair.x;

    // Ensure we process both entry (t_min) and exit (t_max)
    if (hit_kind == 0) {
        float temp = other_t;
        other_t = t;
        t = temp;
    }

    float cur_t = RayTMin();

    // Insert both hits into the sorted buffer
    for (int n = 0; n < 2; n++) {
        float h_t = (n == 0) ? t : other_t;
        uint h_id = (n == 0) ? (2 * prim_id + 1) : (2 * prim_id);  // +1 for enter

        if (h_t > cur_t) {
            // Insertion sort into payload buffer
            [ForceUnroll]
            for (int i = 0; i < BUFFER_SIZE; i++) {
                float test_t = asfloat(payload.data[i * 2]);
                if (h_t < test_t) {
                    // Swap
                    payload.data[i * 2] = asuint(h_t);
                    uint test_id = payload.data[i * 2 + 1];
                    payload.data[i * 2 + 1] = h_id;
                    h_id = test_id;
                    h_t = test_t;
                }
            }
        }
    }

    // Continue if we might find closer hits
    if (t < asfloat(payload.data[2 * (BUFFER_SIZE - 1)])) {
        IgnoreHit();
    }
}

// ============================================================================
// Miss Shader
// ============================================================================

[shader("miss")]
void missShader(inout RayPayload payload) {
    // Nothing to do - payload already initialized
}

// ============================================================================
// Ray Generation Shader
// ============================================================================

[shader("raygeneration")]
void rayGenShader() {
    uint3 idx = DispatchRaysIndex();
    uint3 dim = DispatchRaysDimensions();

    // Get ray from input or generate from camera
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + tmin * ray_directions[idx.x];

    // Initialize volume state
    VolumeState volume_state = make_empty_volume_state();
    volume_state.drgb = initial_drgb[idx.x];

    VolumeSample sample = {};
    uint last_tri = 0xFFFFFFFF;

    // Main ray marching loop
    uint iter = 0;
    while (volume_state.logT < LOG_CUTOFF && iter < max_iters) {
        float start_t = abs(volume_state.t);

        // Initialize payload
        RayPayload payload = {};
        [ForceUnroll]
        for (int i = 0; i < BUFFER_SIZE; i++) {
            payload.data[2 * i] = asuint(1e10f);
            payload.data[2 * i + 1] = 0xFFFFFFFF;
        }

        // Trace ray
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = start_t;
        ray.TMax = tmax;

        TraceRay(traversable, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

        // Process all hits in buffer
        bool done = false;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            sample.t = asfloat(payload.data[2 * i]);
            uint tri = payload.data[2 * i + 1];

            if (sample.t > 1e9f) {
                done = true;
                break;
            }

            sample = get_volume_sample(tri, sample.t);
            volume_state = integrate_volume_sample(volume_state, sample, tmin, tmax, max_prim_size);
            last_tri = tri;
            iter++;

            if (!(volume_state.logT < LOG_CUTOFF && iter < max_iters)) {
                break;
            }
        }

        if (done) break;
    }

    // Extract output
    VolumeOutput output = extract_volume_output(volume_state, tmin);

    // Write results
    image[idx.x] = float4(output.C, output.depth);
    last_state[idx.x] = to_dual_volume_state(volume_state, sample);
    last_dirac[idx.x] = sample.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
