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
    // Transform to unit sphere space
    float3 ocn = safe_div(ro, ra);
    float3 rdn = safe_div(rd, ra);

    // Quadratic: a*t² - 2*bp*t + c = 0
    float a = dot(rdn, rdn);
    float bp = -dot(ocn, rdn);
    float c = dot(ocn, ocn) - 1.0;

    float h = bp * bp - a * c;
    if (h < 0.0)
        return float2(-1.0);

    // Numerically stable root finding
    float q = bp + sign(bp) * safe_sqrt(h);
    return float2(safe_div(c, q), safe_div(q, a));
}

/// Intersect ray with transformed ellipsoid. Returns (t_entry, t_exit) sorted.
[Differentiable]
float2 ray_intersect_ellipsoid(
    in float3 rayo,
    in float3 rayd,
    in float3 scales,
    in float4 quat)
{
    float3 localDir = l2_normalize(rotate_vector(rayd, quat));
    float3 localOrigin = rotate_vector(rayo, quat);

    float2 t = eliIntersect(localOrigin, localDir, scales);

    return float2(min(t.x, t.y), max(t.x, t.y));
}

// =============================================================================
// Sample Point Generation
// =============================================================================

/// Compute intersection sample point with t-clamping for rays starting inside.
/// Ensures consistent behavior between forward pass (OptiX) and backward pass.
[Differentiable]
SamplePoint intersect_clamped(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat,
    float3 color, float density, uint hit_type, no_diff float t_clamp)
{
    float2 t_range = ray_intersect_ellipsoid(rayo - mean, rayd, scales, quat);

    bool is_entry = (hit_type == 1);
    float raw_t = is_entry ? t_range.x : t_range.y;

    // Clamp entry hits to handle rays starting inside primitives
    float t = is_entry ? max(raw_t, t_clamp) : raw_t;

    // Entry: +density, Exit: -density
    float signed_density = is_entry ? density : -density;

    return {
        t,
        signed_density,
        signed_density * color.x,
        signed_density * color.y,
        signed_density * color.z
    };
}
