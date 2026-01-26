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

// =============================================================================
// Tensor Utilities
// =============================================================================

import volume_integrator;
import spherical_harmonics;

// =============================================================================
// Tensor Accessors
// =============================================================================

float3 get_float3(TensorView<float> view, uint i) {
    return {view[i, 0], view[i, 1], view[i, 2]};
}

float3 get_float3(TensorView<float> view, uint i, uint j) {
    return {view[i, j, 0], view[i, j, 1], view[i, j, 2]};
}

float4 get_float4(TensorView<float> view, uint i) {
    return {view[i, 0], view[i, 1], view[i, 2], view[i, 3]};
}

// =============================================================================
// Atomic Operations
// =============================================================================

void atomic_add_float(TensorView<float> view, uint i, float val) {
    float temp;
    view.InterlockedAdd(i, val, temp);
}

void atomic_add_float3(TensorView<float> view, uint i, float3 val) {
    float temp;
    view.InterlockedAdd(uint2(i, 0u), val.x, temp);
    view.InterlockedAdd(uint2(i, 1u), val.y, temp);
    view.InterlockedAdd(uint2(i, 2u), val.z, temp);
}

void atomic_add_float3(TensorView<float> view, uint i, uint j, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(i, j, 0u), val.x, temp);
    view.InterlockedAdd(uint3(i, j, 1u), val.y, temp);
    view.InterlockedAdd(uint3(i, j, 2u), val.z, temp);
}

void atomic_add_float4(TensorView<float> view, uint i, float4 val) {
    float temp;
    view.InterlockedAdd(uint2(i, 0u), val.x, temp);
    view.InterlockedAdd(uint2(i, 1u), val.y, temp);
    view.InterlockedAdd(uint2(i, 2u), val.z, temp);
    view.InterlockedAdd(uint2(i, 3u), val.w, temp);
}

// =============================================================================
// Non-Atomic Operations
// =============================================================================

void add_float3(TensorView<float> view, uint i, float3 val) {
    view[uint2(i, 0u)] += val.x;
    view[uint2(i, 1u)] += val.y;
    view[uint2(i, 2u)] += val.z;
}

void add_float3(TensorView<float> view, uint i, uint j, float3 val) {
    view[uint3(i, j, 0u)] += val.x;
    view[uint3(i, j, 1u)] += val.y;
    view[uint3(i, j, 2u)] += val.z;
}

// =============================================================================
// Structured Type Loading
// =============================================================================

/// Load IntegratorState from TensorView (48 bytes = 12 floats).
IntegratorState get_state(TensorView<float> view, uint idx) {
    IntegratorState state;
    state.contrib = float4(view[idx, 0], view[idx, 1], view[idx, 2], view[idx, 3]);
    state.C = float4(view[idx, 4], view[idx, 5], view[idx, 6], view[idx, 7]);
    state.logT = view[idx, 8];
    state.depth_num = view[idx, 9];
    state.t = view[idx, 10];
    state._pad = view[idx, 11];
    return state;
}

/// Load Features from TensorView.
Features get_feats(TensorView<float> features, uint prim_idx, uint sh_degree) {
    Features feat;
    feat.degree = sh_degree;
    uint num_coeff = (sh_degree + 1) * (sh_degree + 1);
    for (uint i = 0; i < num_coeff; i++) {
        feat.c[i] = get_float3(features, prim_idx, i);
    }
    return feat;
}
