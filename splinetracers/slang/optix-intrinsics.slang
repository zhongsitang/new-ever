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

// OptiX Intrinsics for Modern Slang (OptiX 9.1+ compatible)
//
// Modern Slang natively supports HLSL-style ray tracing intrinsics that map directly
// to OptiX calls. This module provides additional utilities and compatibility wrappers.
//
// Native Slang ray tracing intrinsics (no import needed):
//   - DispatchRaysIndex(), DispatchRaysDimensions()
//   - WorldRayOrigin(), WorldRayDirection()
//   - ObjectRayOrigin(), ObjectRayDirection()
//   - RayTMin(), RayTCurrent(), RayTMax()
//   - PrimitiveIndex(), InstanceIndex(), InstanceID()
//   - HitKind(), RayFlags()
//   - ObjectToWorld3x4(), WorldToObject3x4()
//   - ReportHit(), IgnoreHit(), AcceptHitAndEndSearch()
//   - TraceRay()

module optix_intrinsics;

// ============================================================================
// Custom Hit Attributes
// ============================================================================

// Custom attributes structure for procedural intersection
// Can pass up to 8 uint values through attributes (a0-a7)
struct CustomHitAttributes
{
    uint a0;
    uint a1;
    uint a2;
    uint a3;
    uint a4;
    uint a5;
    uint a6;
    uint a7;
};

// ============================================================================
// Hit Buffer for Multi-Hit Tracking
// ============================================================================

static const uint MAX_HITS = 16;

struct HitEntry
{
    float t;
    uint triIndex;
};

struct HitBuffer
{
    HitEntry entries[MAX_HITS];

    [mutating]
    void init()
    {
        [ForceUnroll]
        for (int i = 0; i < MAX_HITS; i++)
        {
            entries[i].t = 1e10f;
            entries[i].triIndex = 0xFFFFFFFF;
        }
    }

    [mutating]
    void insert(float t, uint triIndex)
    {
        float h_t = t;
        uint h_i = triIndex;

        [ForceUnroll]
        for (int i = 0; i < MAX_HITS; i++)
        {
            if (h_t < entries[i].t)
            {
                float temp_t = entries[i].t;
                uint temp_i = entries[i].triIndex;
                entries[i].t = h_t;
                entries[i].triIndex = h_i;
                h_t = temp_t;
                h_i = temp_i;
            }
        }
    }

    float getMaxT()
    {
        return entries[MAX_HITS - 1].t;
    }
};

// ============================================================================
// Ray Payload Types
// ============================================================================

// Generic ray payload for closest-hit queries
struct RayPayload
{
    float3 color;
    float hitT;
    uint hitKind;
    bool hit;
};

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

// Pack two floats into a float2 for attribute passing
float2 packFloats(float a, float b)
{
    return float2(a, b);
}

// ============================================================================
// ReportHit Wrappers
// ============================================================================

// Report a procedural hit with a single float attribute
bool ReportHitWithAttr1(float hitT, uint hitKind, float attr0)
{
    return ReportHit(hitT, hitKind, float2(attr0, 0));
}

// Report a procedural hit with two float attributes
bool ReportHitWithAttr2(float hitT, uint hitKind, float attr0, float attr1)
{
    return ReportHit(hitT, hitKind, float2(attr0, attr1));
}

// ============================================================================
// Attribute Access Helpers
// ============================================================================

// Get the first attribute as float (for procedural intersections)
float GetAttribute0AsFloat()
{
    BuiltInTriangleIntersectionAttributes attr;
    return attr.barycentrics.x;
}

// Get the second attribute as float (for procedural intersections)
float GetAttribute1AsFloat()
{
    BuiltInTriangleIntersectionAttributes attr;
    return attr.barycentrics.y;
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
