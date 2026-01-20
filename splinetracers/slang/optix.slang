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

// ============================================================================
// OptiX Utilities for Slang
// Compatible with OptiX 9.1 and Slang 2025+
// ============================================================================

// Import the generated intrinsics
import optix_intrinsics;

// ============================================================================
// Utility Functions
// ============================================================================

/// Modulo operation
uint mod(uint a, uint b) {
    __target_switch {
        case cuda: __intrinsic_asm "$0 % $1";
    }
}

/// Convert linear RGB to sRGB
float3 toSRGB(float3 c) {
    const float invGamma = 1.f / 2.4f;
    const float3 powed = { pow(c.x, invGamma), pow(c.y, invGamma), pow(c.z, invGamma) };
    return {
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
    };
}

/// Convert sRGB to linear RGB
float3 fromSRGB(float3 c) {
    const float gamma = 2.4f;
    return {
        c.x < 0.04045f ? c.x / 12.92f : pow((c.x + 0.055f) / 1.055f, gamma),
        c.y < 0.04045f ? c.y / 12.92f : pow((c.y + 0.055f) / 1.055f, gamma),
        c.z < 0.04045f ? c.z / 12.92f : pow((c.z + 0.055f) / 1.055f, gamma)
    };
}

/// Convert float3 color to packed RGBA uint (alpha = 255)
uint make_color(float3 c) {
    float3 srgbC = c;

    const int r = int(255.99f * clamp(srgbC.x, 0, 1));
    const int g = int(255.99f * clamp(srgbC.y, 0, 1));
    const int b = int(255.99f * clamp(srgbC.z, 0, 1));

    // Convert to 32-bit RGBA value (alpha = 0xff)
    const uint rgba = 0x00000000
        | (r << 0) | (g << 8) | (b << 16) | (255u << 24);
    return rgba;
}

/// Unpack RGBA uint to float3 color
float3 unpack_color(uint rgba) {
    return {
        float((rgba >> 0) & 0xFF) / 255.0f,
        float((rgba >> 8) & 0xFF) / 255.0f,
        float((rgba >> 16) & 0xFF) / 255.0f
    };
}

/// Luminance of a color
float luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
