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

#pragma once

#include <cstdint>
#include <cstddef>

#include "structs.h"

using uint = uint32_t;

struct alignas(16) Params
{
    StructuredBuffer<float4> image;
    StructuredBuffer<uint> iters;
    StructuredBuffer<uint> last_face;
    StructuredBuffer<uint> touch_count;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int> tri_collection;
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    uint sh_degree;
    uint max_iters;
    float tmin;
    float tmax;
    StructuredBuffer<float4> initial_drgb;
    float max_prim_size;
};
