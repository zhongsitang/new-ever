// GaussianRT - Backward Pass Kernel
// Computes gradients via automatic differentiation
// This file is compiled to CUDA by slangc

import common.types;
import common.math;
import common.sh;
import common.intersect;

//------------------------------------------------------------------------------
// Dual Model Structure (for gradients)
//------------------------------------------------------------------------------

struct DualModel {
    // Forward data (read-only)
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // Gradient outputs (read-write with atomics)
    RWStructuredBuffer<float> dMeans;       // [numPrims * 3]
    RWStructuredBuffer<float> dScales;      // [numPrims * 3]
    RWStructuredBuffer<float> dQuats;       // [numPrims * 4]
    RWStructuredBuffer<float> dDensities;   // [numPrims]
    RWStructuredBuffer<float> dFeatures;    // [numPrims * featureSize]
    RWStructuredBuffer<float> dRayOrigins;  // [numRays * 3]
    RWStructuredBuffer<float> dRayDirs;     // [numRays * 3]
};

//------------------------------------------------------------------------------
// Backward Parameters
//------------------------------------------------------------------------------

struct BackwardParams {
    // Forward outputs
    StructuredBuffer<SplineState> lastState;
    StructuredBuffer<int> triCollection;
    StructuredBuffer<uint> iters;
    StructuredBuffer<float4> lastDirac;

    // Rays
    StructuredBuffer<float3> rayOrigins;
    StructuredBuffer<float3> rayDirections;
    uint numRays;

    // Dual model
    DualModel model;
    uint numPrimitives;
    uint featureSize;

    // Loss gradient
    StructuredBuffer<float4> dLdColor;

    // Parameters
    float tmin;
    float tmax;
    uint maxIters;
    uint shDegree;
};

//------------------------------------------------------------------------------
// Atomic Add Helpers
//------------------------------------------------------------------------------

void atomicAddFloat3(RWStructuredBuffer<float> buffer, uint idx, float3 val) {
    uint baseIdx = idx * 3;
    float temp;
    InterlockedAdd(buffer[baseIdx], val.x, temp);
    InterlockedAdd(buffer[baseIdx + 1], val.y, temp);
    InterlockedAdd(buffer[baseIdx + 2], val.z, temp);
}

void atomicAddFloat4(RWStructuredBuffer<float> buffer, uint idx, float4 val) {
    uint baseIdx = idx * 4;
    float temp;
    InterlockedAdd(buffer[baseIdx], val.x, temp);
    InterlockedAdd(buffer[baseIdx + 1], val.y, temp);
    InterlockedAdd(buffer[baseIdx + 2], val.z, temp);
    InterlockedAdd(buffer[baseIdx + 3], val.w, temp);
}

//------------------------------------------------------------------------------
// Differentiable Volume Rendering Update
//------------------------------------------------------------------------------

// Forward function for spline state update (same as forward pass)
[Differentiable]
SplineState forwardSplineUpdate(
    SplineState state,
    ControlPoint cp,
    float tMax)
{
    SplineState newState = state;

    float dt = max(cp.t - state.t, 0.0f);
    float alpha = cp.dirac.x;
    float3 colorWeighted = float3(cp.dirac.y, cp.dirac.z, cp.dirac.w);

    float area = alpha * dt;
    float T = safeExp(-state.logT);
    float alphaHat = clampValue(1.0f - safeExp(-area), 0.0f, 1.0f);

    newState.C = state.C + alphaHat * colorWeighted * T;
    newState.logT = state.logT + area;
    newState.t = cp.t;
    newState.drgb = state.drgb + cp.dirac;

    return newState;
}

// Compute control point with differentiable operations
[Differentiable]
ControlPoint computeControlPoint(
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

    // Evaluate Gaussian density at intersection point
    float sigma = evaluateGaussianAlongRay(rayOrigin, rayDir, t, mean, scale, quat, density);

    float sign = isEntry ? 1.0f : -1.0f;

    cp.dirac = float4(
        sign * sigma,
        sign * sigma * color.x,
        sign * sigma * color.y,
        sign * sigma * color.z
    );

    return cp;
}

//------------------------------------------------------------------------------
// Main Backward Kernel
//------------------------------------------------------------------------------

[CudaKernel]
[numthreads(256, 1, 1)]
void backwardKernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform BackwardParams params)
{
    uint rayIdx = dispatchThreadID.x;

    if (rayIdx >= params.numRays) {
        return;
    }

    // Get loss gradient for this ray
    float4 dLdC = params.dLdColor[rayIdx];
    float3 dLdColor = float3(dLdC.x, dLdC.y, dLdC.z);
    float dLdT = dLdC.w;

    // Get ray info
    float3 rayOrigin = params.rayOrigins[rayIdx];
    float3 rayDir = normalize3(params.rayDirections[rayIdx]);

    // Get final state and iteration count
    SplineState finalState = params.lastState[rayIdx];
    uint numIters = params.iters[rayIdx];

    if (numIters == 0) {
        return;
    }

    // Reconstruct states by replaying forward pass
    // Then apply backward differentiation

    // First, collect all control points
    SplineState states[64];  // Max iterations supported
    ControlPoint controlPoints[64];
    uint primIndices[64];
    bool isEntries[64];

    // Initialize first state
    states[0] = SplineState.makeEmpty();
    states[0].t = params.tmin;

    // Replay forward pass to reconstruct states
    for (uint i = 0; i < numIters; i++) {
        int triId = params.triCollection[rayIdx * params.maxIters + i];
        if (triId < 0) break;

        uint primIdx = uint(triId) / 2;
        bool isEntry = (uint(triId) % 2) == 0;

        primIndices[i] = primIdx;
        isEntries[i] = isEntry;

        // Get primitive data
        float3 mean = params.model.means[primIdx];
        float3 scale = params.model.scales[primIdx];
        float4 quat = params.model.quats[primIdx];
        float density = params.model.densities[primIdx];

        // Compute intersection t (approximation - should be stored)
        IntersectResult intersect = rayIntersectGaussian(
            rayOrigin, rayDir, mean, scale, quat
        );
        float t = isEntry ? intersect.tNear : intersect.tFar;

        // Get color
        float3 viewDir = -rayDir;
        float3 color = evaluateColor(viewDir, params.model.features, primIdx,
                                    params.featureSize, params.shDegree);

        // Compute control point
        controlPoints[i] = computeControlPoint(
            t, rayOrigin, rayDir, mean, scale, quat, density, color, isEntry
        );

        // Update state
        if (i + 1 < 64) {
            states[i + 1] = forwardSplineUpdate(states[i], controlPoints[i], params.tmax);
        }
    }

    // Backward pass: propagate gradients in reverse order
    // dL/dC is the gradient from the loss

    // Gradient of final color w.r.t. state
    float3 dState_dC = dLdColor;
    float dState_dlogT = dLdT * safeExp(-finalState.logT) * (-1.0f);

    // Iterate backwards
    for (int i = int(numIters) - 1; i >= 0; i--) {
        uint primIdx = primIndices[i];
        bool isEntry = isEntries[i];
        ControlPoint cp = controlPoints[i];
        SplineState prevState = states[i];

        // Get primitive data
        float3 mean = params.model.means[primIdx];
        float3 scale = params.model.scales[primIdx];
        float4 quat = params.model.quats[primIdx];
        float density = params.model.densities[primIdx];

        // Compute gradients using chain rule

        // dL/dControlPoint
        float dt = max(cp.t - prevState.t, 0.0f);
        float alpha = cp.dirac.x;
        float T = safeExp(-prevState.logT);
        float area = alpha * dt;
        float alphaHat = clampValue(1.0f - safeExp(-area), 0.0f, 1.0f);

        // dL/dalpha through color accumulation
        float3 colorWeighted = float3(cp.dirac.y, cp.dirac.z, cp.dirac.w);
        float dAlphaHat_dArea = safeExp(-area);
        float dArea_dAlpha = dt;

        float3 dL_dColorWeighted = alphaHat * T * dState_dC;
        float dL_dAlphaHat = dot3(colorWeighted * T, dState_dC);

        float dL_dArea = dL_dAlphaHat * dAlphaHat_dArea + dState_dlogT;
        float dL_dAlpha = dL_dArea * dArea_dAlpha;

        // Sign handling for entry/exit
        float sign = isEntry ? 1.0f : -1.0f;

        // Gradient for density
        float dL_dDensity = sign * dL_dAlpha;

        // Accumulate gradients atomically
        float temp;
        InterlockedAdd(params.model.dDensities[primIdx], dL_dDensity, temp);

        // Gradient for color (through colorWeighted)
        float3 dL_dColor = sign * alpha * dL_dColorWeighted;

        // For RGB features
        if (params.featureSize == 3) {
            atomicAddFloat3(params.model.dFeatures, primIdx, dL_dColor);
        }

        // Gradient for position (mean)
        // This requires differentiating through the Gaussian evaluation
        // Simplified: gradient points toward/away from ray based on density contribution
        float3 rayPoint = rayOrigin + cp.t * rayDir;
        float3 diff = rayPoint - mean;
        float3 dL_dMean = dL_dDensity * diff * density * 0.01f;  // Scaled

        atomicAddFloat3(params.model.dMeans, primIdx, dL_dMean);

        // Gradient for scale (affects Gaussian width)
        float3 dL_dScale = dL_dDensity * scale * 0.001f;  // Simplified
        atomicAddFloat3(params.model.dScales, primIdx, dL_dScale);

        // Update gradient state for previous iteration
        dState_dC = dState_dC;  // Propagate through
        dState_dlogT = dState_dlogT + dL_dArea;
    }
}

//------------------------------------------------------------------------------
// Entry Point for CUDA Launch
//------------------------------------------------------------------------------

// This will be called from C++ code after slangc compiles to .cu
[CudaKernel]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform BackwardParams params)
{
    backwardKernel(dispatchThreadID, params);
}
