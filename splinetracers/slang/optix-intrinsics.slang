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

// OptiX Intrinsics Module for Slang (OptiX 9.1+ / CUDA target)
//
// For CUDA target, use these Slang intrinsics:
//
// Ray Query:
//   - DispatchRaysIndex(), DispatchRaysDimensions()
//   - WorldRayOrigin(), WorldRayDirection()
//   - ObjectRayOrigin(), ObjectRayDirection()
//   - RayTMin(), RayTCurrent(), RayTMax()
//   - PrimitiveIndex(), InstanceIndex(), InstanceID()
//   - HitKind(), RayFlags()
//
// Intersection:
//   - ReportHitOptix(float t, uint hitKind, uint attr0, ...) - for CUDA target
//   - IgnoreHit(), AcceptHitAndEndSearch()
//
// Tracing:
//   - TraceRay(accel, flags, mask, sbtOffset, sbtStride, missIndex, ray, payload)

module optix_intrinsics;

// ============================================================================
// Utility Functions
// ============================================================================

// Convert float to uint bits
uint floatBitsToUint(float f)
{
    return asuint(f);
}

// Convert uint bits to float
float uintBitsToFloat(uint u)
{
    return asfloat(u);
}

// ============================================================================
// Color Utilities
// ============================================================================

// Convert linear color to sRGB
float3 linearToSRGB(float3 c)
{
    const float invGamma = 1.0f / 2.4f;
    float3 powed = float3(pow(c.x, invGamma), pow(c.y, invGamma), pow(c.z, invGamma));
    return float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
    );
}

// Pack RGB color (0-1 range) into a 32-bit RGBA value
uint packColor(float3 c)
{
    uint r = uint(255.99f * saturate(c.x));
    uint g = uint(255.99f * saturate(c.y));
    uint b = uint(255.99f * saturate(c.z));
    return (r << 0) | (g << 8) | (b << 16) | (255u << 24);
}

// ============================================================================
// Ray Construction Helpers
// ============================================================================

// Create a RayDesc from components
RayDesc makeRay(float3 origin, float3 direction, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    return ray;
}
