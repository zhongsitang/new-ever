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

// OptiX Utility Module for Modern Slang (OptiX 9.1+ compatible)
//
// This module provides common utility functions for OptiX ray tracing.
// Modern Slang natively supports all HLSL ray tracing intrinsics.

module optix;

// ============================================================================
// Color Conversion Utilities
// ============================================================================

// Convert linear color to sRGB color space
float3 toSRGB(float3 c)
{
    const float invGamma = 1.0f / 2.4f;
    float3 powed = float3(pow(c.x, invGamma), pow(c.y, invGamma), pow(c.z, invGamma));
    return float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
    );
}

// Convert sRGB color to linear color space
float3 fromSRGB(float3 c)
{
    const float gamma = 2.4f;
    return float3(
        c.x < 0.04045f ? c.x / 12.92f : pow((c.x + 0.055f) / 1.055f, gamma),
        c.y < 0.04045f ? c.y / 12.92f : pow((c.y + 0.055f) / 1.055f, gamma),
        c.z < 0.04045f ? c.z / 12.92f : pow((c.z + 0.055f) / 1.055f, gamma)
    );
}

// Pack RGB float color (0-1 range) into a 32-bit RGBA uint value
uint make_color(float3 c)
{
    uint r = uint(255.99f * saturate(c.x));
    uint g = uint(255.99f * saturate(c.y));
    uint b = uint(255.99f * saturate(c.z));

    // RGBA format with alpha = 0xFF
    return (r << 0) | (g << 8) | (b << 16) | (255u << 24);
}

// Pack RGBA float color (0-1 range) into a 32-bit RGBA uint value
uint make_color_rgba(float4 c)
{
    uint r = uint(255.99f * saturate(c.x));
    uint g = uint(255.99f * saturate(c.y));
    uint b = uint(255.99f * saturate(c.z));
    uint a = uint(255.99f * saturate(c.w));

    return (r << 0) | (g << 8) | (b << 16) | (a << 24);
}

// Unpack 32-bit RGBA uint value to float4 color (0-1 range)
float4 unpack_color(uint rgba)
{
    float r = float((rgba >> 0) & 0xFF) / 255.0f;
    float g = float((rgba >> 8) & 0xFF) / 255.0f;
    float b = float((rgba >> 16) & 0xFF) / 255.0f;
    float a = float((rgba >> 24) & 0xFF) / 255.0f;

    return float4(r, g, b, a);
}

// ============================================================================
// Math Utilities
// ============================================================================

// Safe modulo operation
uint mod(uint a, uint b)
{
    return a % b;
}

// Luminance of an RGB color
float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// ============================================================================
// Ray Utilities
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

// Offset ray origin to avoid self-intersection
float3 offsetRay(float3 p, float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z))
    );

    return float3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z
    );
}
