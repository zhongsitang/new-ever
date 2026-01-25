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

import volume_integrator;
import safe_math;

// =============================================================================
// Ray-Ellipsoid Intersection
// =============================================================================

/// Intersect ray with axis-aligned ellipsoid at origin.
/// Returns (t_near, t_far) or (-1, -1) if no intersection.
[Differentiable]
float2 eliIntersect(in float3 ro, in float3 rd, in float3 ra)
{
    float3 ocn = safe_div(ro, ra);
    float3 rdn = safe_div(rd, ra);
    float a = dot(rdn, rdn);
    float bp = -dot(ocn, rdn);
    float3 l = (ocn + safe_div(bp, a) * rdn);
    float h = a * (1.f - dot(l, l));
    float c = dot(ocn, ocn) - 1.f;
    if (h < 0.0) return float2(-1.0);
    let q = bp + sign(bp) * safe_sqrt(h);
    return {safe_div(c, q), safe_div(q, a)};
}

/// Intersect ray with transformed ellipsoid.
/// Returns (t_entry, t_exit) sorted.
[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 rayo,
    in float3 rayd,
    in float3 scales,
    in float4 quat)
{
    let Trayd = l2_normalize(rotate_vector(rayd, quat));
    let Trayo = rotate_vector(rayo, quat);
    float2 fminmaxt = eliIntersect(Trayo, Trayd, scales);
    return {min(fminmaxt.x, fminmaxt.y), max(fminmaxt.x, fminmaxt.y)};
}

// =============================================================================
// Sample Point Generation
// =============================================================================

/// Compute intersection sample point with t clamping for rays starting inside.
///
/// When the ray origin is inside the ellipsoid, t_entry < tmin. Using max(t_entry, t_clamp)
/// ensures consistent behavior between forward and backward passes:
/// - Forward: OptiX reports t = tmin for virtual entry hits
/// - Backward: This function computes t = max(t_entry, tmin), giving the same value
///
/// The max operation naturally handles gradients:
/// - When t_entry >= t_clamp: gradient flows to geometry parameters normally
/// - When t_entry < t_clamp: gradient is zero (t is clamped, no dependency on geometry)
[Differentiable]
SamplePoint intersect_clamped(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat,
    float3 color, float density, uint hit_type, no_diff float t_clamp
) {
    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scales, quat);

    float raw_t = (hit_type == 1) ? minmaxt.x : minmaxt.y;

    // Clamp t to handle rays starting inside primitives
    // For entry hits: max(t_entry, t_clamp) handles the inside case
    // For exit hits: no clamping needed (exit is always ahead)
    float t = (hit_type == 1) ? max(raw_t, t_clamp) : raw_t;

    // Entry point adds +density, exit point adds -density
    float signed_density = ((hit_type == 1) ? density : -density);

    SamplePoint out = {
        t, signed_density, signed_density * color.x, signed_density * color.y, signed_density * color.z
    };
    return out;
}
