// GaussianRT - Ray-Ellipsoid intersection
// Differentiable intersection for ellipsoid volume primitives

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

// Bounded ellipsoid intersection for volume primitives
// Uses sigma bound to define the effective radius (e.g., 3-sigma for Gaussian falloff)
[Differentiable]
IntersectResult rayIntersectEllipsoidBounded(
    float3 rayOrigin,
    float3 rayDir,
    float3 position,
    float3 scale,
    float4 orientation,
    float sigmaBound = 3.0f)
{
    // Scale by sigma bound for effective radius
    float3 effectiveScale = scale * sigmaBound;
    return rayIntersectEllipsoid(rayOrigin, rayDir, position, effectiveScale, orientation);
}

//------------------------------------------------------------------------------
// Ellipsoid Density Evaluation
// Uses Gaussian falloff for smooth density field
//------------------------------------------------------------------------------

// Evaluate ellipsoid density at a point (differentiable)
// Uses Gaussian falloff: density * exp(-0.5 * mahalanobis_dist^2)
[Differentiable]
float evaluateEllipsoidDensity(
    float3 point,
    float3 position,
    float3 scale,
    float4 orientation,
    float density)
{
    // Transform point to local coordinates
    float3 localPoint = point - position;
    float3x3 R = quatToMatrix(orientation);
    float3x3 Rt = transpose3x3(R);
    localPoint = mulMat3Vec(Rt, localPoint);

    // Normalize by scale
    float3 normalizedPoint = localPoint / scale;

    // Compute squared Mahalanobis distance
    float dist2 = dot3(normalizedPoint, normalizedPoint);

    // Gaussian falloff: exp(-0.5 * dist^2)
    return density * safeExp(-0.5f * dist2);
}

// Evaluate ellipsoid density along ray at parameter t
[Differentiable]
float evaluateEllipsoidAlongRay(
    float3 rayOrigin,
    float3 rayDir,
    float t,
    float3 position,
    float3 scale,
    float4 orientation,
    float density)
{
    float3 point = rayOrigin + t * rayDir;
    return evaluateEllipsoidDensity(point, position, scale, orientation, density);
}

//------------------------------------------------------------------------------
// Volume Sample Generation
// Creates a sample for volume rendering integration at a ray-primitive intersection
// This is primitive-type agnostic - works with any ellipsoid-based primitive
//------------------------------------------------------------------------------

// Create a volume sample for the discrete volume rendering equation:
//   C = Σᵢ αᵢ · cᵢ · Πⱼ₌₁^{i-1}(1 - αⱼ)
[Differentiable]
VolumeSample createVolumeSample(
    float t,                    // Ray parameter at intersection
    float3 rayOrigin,
    float3 rayDir,
    float3 position,
    float3 scale,
    float4 orientation,
    float density,              // Base opacity σ
    float3 color,               // RGB color at this point
    bool isEntry)               // true if entering primitive, false if exiting
{
    VolumeSample sample;
    sample.t = t;

    // Evaluate the density σ(t) at the intersection point
    float sigma = evaluateEllipsoidAlongRay(rayOrigin, rayDir, t, position, scale, orientation, density);

    // For the discrete volume rendering formulation:
    // - Entry points contribute positively (add to accumulated density)
    // - Exit points contribute negatively (subtract from accumulated density)
    float sign = isEntry ? 1.0f : -1.0f;

    // Store [α, α·R, α·G, α·B] where α is the signed density contribution
    sample.alphaColorProduct = float4(
        sign * sigma,
        sign * sigma * color.x,
        sign * sigma * color.y,
        sign * sigma * color.z
    );

    return sample;
}
