// Modern Splinetracer - Ellipsoid Intersection
// Ray-ellipsoid intersection for procedural geometry
// Apache License 2.0

module ellipsoid;

import safe_math;

// Compute ray-ellipsoid intersection
// Returns (t_enter, t_exit), both negative if no intersection
[Differentiable]
public float2 ray_intersect_ellipsoid(
    float3 ray_origin,      // Ray origin in ellipsoid local space (origin at center)
    float3 ray_direction,   // Ray direction
    float3 scale,           // Ellipsoid semi-axes
    float4 quaternion) {    // Rotation quaternion

    // Transform ray to ellipsoid's local coordinate system
    float3 local_dir = rotate_vector(ray_direction, quaternion);
    float3 local_origin = rotate_vector(ray_origin, quaternion);

    // Normalize by scale to transform to unit sphere intersection
    float3 scaled_origin = safe_div(local_origin, scale);
    float3 scaled_dir = safe_div(local_dir, scale);

    // Solve quadratic: |origin + t*dir|^2 = 1
    float a = dot(scaled_dir, scaled_dir);
    float b_half = dot(scaled_origin, scaled_dir);
    float c = dot(scaled_origin, scaled_origin) - 1.0f;

    float discriminant = b_half * b_half - a * c;

    if (discriminant < 0.0f) {
        return float2(-1.0f, -1.0f);  // No intersection
    }

    // Use numerically stable quadratic formula
    float sqrt_disc = safe_sqrt(discriminant);
    float q = b_half > 0.0f ? -b_half - sqrt_disc : -b_half + sqrt_disc;

    float t0 = safe_div(c, q);
    float t1 = safe_div(q, a);

    // Return sorted (min, max)
    return float2(min(t0, t1), max(t0, t1));
}

// Half-precision version for performance
public float2 ray_intersect_ellipsoid_half(
    half3 ray_origin,
    half3 ray_direction,
    half3 scale,
    half4 quaternion) {

    // Convert and use full precision for accuracy
    float3 origin = float3(ray_origin);
    float3 dir = float3(ray_direction);
    float3 s = float3(scale);
    float4 q = float4(quaternion);

    return ray_intersect_ellipsoid(origin, dir, s, q);
}

// Check if ray intersects ellipsoid (faster than full intersection)
public bool ray_hits_ellipsoid(
    float3 ray_origin,
    float3 ray_direction,
    float3 scale,
    float4 quaternion,
    float t_min,
    float t_max) {

    float2 t = ray_intersect_ellipsoid(ray_origin, ray_direction, scale, quaternion);
    return t.y >= t_min && t.x <= t_max;
}
