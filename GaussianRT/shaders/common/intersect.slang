// GaussianRT - Ray-Ellipsoid intersection
// Differentiable intersection for 3D Gaussians

module intersect;

import types;
import math;

//------------------------------------------------------------------------------
// Ray-Ellipsoid Intersection
//------------------------------------------------------------------------------

// Result of intersection test
struct IntersectResult {
    float tNear;    // Entry point parameter
    float tFar;     // Exit point parameter
    bool hit;       // Whether ray intersects ellipsoid
};

// Ray-ellipsoid intersection (differentiable)
// Ellipsoid is defined by center, scale (radii), and rotation quaternion
[Differentiable]
IntersectResult rayIntersectEllipsoid(
    float3 rayOrigin,   // Ray origin in world space
    float3 rayDir,      // Ray direction (not necessarily normalized)
    float3 center,      // Ellipsoid center
    float3 scale,       // Ellipsoid scale (radii along local axes)
    float4 quat)        // Rotation quaternion
{
    IntersectResult result;
    result.hit = false;
    result.tNear = -1.0f;
    result.tFar = -1.0f;

    // Transform ray to ellipsoid's local coordinate system
    float3 ro = rayOrigin - center;

    // Apply inverse rotation
    float3x3 R = quatToMatrix(quat);
    float3x3 Rt = transpose3x3(R);

    float3 roLocal = mulMat3Vec(Rt, ro);
    float3 rdLocal = mulMat3Vec(Rt, rayDir);

    // Normalize by scale to transform to unit sphere
    float3 roNorm = roLocal / scale;
    float3 rdNorm = rdLocal / scale;

    // Solve quadratic: |roNorm + t * rdNorm|^2 = 1
    float a = dot3(rdNorm, rdNorm);
    float b = 2.0f * dot3(roNorm, rdNorm);
    float c = dot3(roNorm, roNorm) - 1.0f;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return result;
    }

    float sqrtD = safeSqrt(discriminant);
    float t1 = (-b - sqrtD) / (2.0f * a);
    float t2 = (-b + sqrtD) / (2.0f * a);

    result.tNear = min(t1, t2);
    result.tFar = max(t1, t2);
    result.hit = true;

    return result;
}

// Simplified intersection for AABB-based primitives
// Uses the fact that Gaussian is ~3-sigma bounded
[Differentiable]
IntersectResult rayIntersectGaussian(
    float3 rayOrigin,
    float3 rayDir,
    float3 mean,
    float3 scale,
    float4 quat,
    float sigmaBound = 3.0f)
{
    // Scale by sigma bound for effective radius
    float3 effectiveScale = scale * sigmaBound;
    return rayIntersectEllipsoid(rayOrigin, rayDir, mean, effectiveScale, quat);
}

//------------------------------------------------------------------------------
// Gaussian Density Evaluation
//------------------------------------------------------------------------------

// Evaluate Gaussian density at a point (differentiable)
[Differentiable]
float evaluateGaussianDensity(
    float3 point,
    float3 mean,
    float3 scale,
    float4 quat,
    float density)
{
    // Transform point to local coordinates
    float3 localPoint = point - mean;
    float3x3 R = quatToMatrix(quat);
    float3x3 Rt = transpose3x3(R);
    localPoint = mulMat3Vec(Rt, localPoint);

    // Normalize by scale
    float3 normalizedPoint = localPoint / scale;

    // Compute squared Mahalanobis distance
    float dist2 = dot3(normalizedPoint, normalizedPoint);

    // Gaussian: exp(-0.5 * dist^2)
    return density * safeExp(-0.5f * dist2);
}

// Evaluate Gaussian along ray at parameter t
[Differentiable]
float evaluateGaussianAlongRay(
    float3 rayOrigin,
    float3 rayDir,
    float t,
    float3 mean,
    float3 scale,
    float4 quat,
    float density)
{
    float3 point = rayOrigin + t * rayDir;
    return evaluateGaussianDensity(point, mean, scale, quat, density);
}

//------------------------------------------------------------------------------
// Control Point Generation
//------------------------------------------------------------------------------

// Generate control point for volume rendering
// A control point represents a Dirac impulse at the intersection boundary
[Differentiable]
ControlPoint generateControlPoint(
    float t,
    float3 rayOrigin,
    float3 rayDir,
    float3 mean,
    float3 scale,
    float4 quat,
    float density,
    float3 color,
    bool isEntry)
{
    ControlPoint cp;
    cp.t = t;

    // Evaluate density at intersection point
    float sigma = evaluateGaussianAlongRay(rayOrigin, rayDir, t, mean, scale, quat, density);

    // For volume rendering, we use the density-weighted color
    // Sign depends on whether entering or exiting
    float sign = isEntry ? 1.0f : -1.0f;

    cp.dirac = float4(
        sign * sigma,
        sign * sigma * color.x,
        sign * sigma * color.y,
        sign * sigma * color.z
    );

    return cp;
}
