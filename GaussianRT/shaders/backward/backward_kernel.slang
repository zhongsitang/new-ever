// GaussianRT - Backward Pass Kernel
// Computes gradients for differentiable volume rendering
// This file is compiled to CUDA by slangc

import common.types;
import common.math;
import common.sh;
import common.intersect;

//------------------------------------------------------------------------------
// Gradient Model Structure
// Holds both forward data and gradient output buffers
//------------------------------------------------------------------------------

struct GradientModel {
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
// Backward Pass Parameters
//------------------------------------------------------------------------------

struct BackwardParams {
    // Forward pass outputs (for gradient computation)
    StructuredBuffer<VolumeIntegrationState> finalIntegrationState;  // Per-ray final state
    StructuredBuffer<int> hitPrimitiveIds;                           // Visited primitive IDs
    StructuredBuffer<uint> numSamplesPerRay;                         // Number of samples per ray
    StructuredBuffer<float4> lastSampleAlphaRGB;                     // Last accumulated alpha-RGB

    // Rays
    StructuredBuffer<float3> rayOrigins;
    StructuredBuffer<float3> rayDirections;
    uint numRays;

    // Gradient model
    GradientModel model;
    uint numPrimitives;
    uint featureSize;

    // Loss gradient input: dL/dColor for each ray
    StructuredBuffer<float4> dLossPerColor;

    // Render parameters
    float tmin;
    float tmax;
    uint maxSamplesPerRay;
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
// Differentiable Volume Integration Functions
// These mirror the forward pass for automatic differentiation
//------------------------------------------------------------------------------

// Forward volume integration step (same as forward pass)
// Used to reconstruct intermediate states during backward pass
[Differentiable]
VolumeIntegrationState forwardIntegrationStep(
    VolumeIntegrationState state,
    GaussianSample sample,
    float tMax)
{
    VolumeIntegrationState newState = state;

    float deltaT = max(sample.t - state.rayT, 0.0f);
    float alpha = sample.alphaColorProduct.x;
    float3 alphaColor = float3(
        sample.alphaColorProduct.y,
        sample.alphaColorProduct.z,
        sample.alphaColorProduct.w
    );

    float opticalDepth = alpha * deltaT;
    float transmittance = safeExp(-state.logTransmittance);
    float discreteAlpha = clampValue(1.0f - safeExp(-opticalDepth), 0.0f, 1.0f);

    newState.accumulatedColor = state.accumulatedColor + discreteAlpha * alphaColor * transmittance;
    newState.logTransmittance = state.logTransmittance + opticalDepth;
    newState.rayT = sample.t;
    newState.accumulatedAlphaRGB = state.accumulatedAlphaRGB + sample.alphaColorProduct;

    return newState;
}

// Compute Gaussian sample with differentiable operations
[Differentiable]
GaussianSample computeGaussianSample(
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
    GaussianSample sample;
    sample.t = t;

    // Evaluate Gaussian density at intersection point
    float sigma = evaluateGaussianAlongRay(rayOrigin, rayDir, t, mean, scale, quat, density);

    float sign = isEntry ? 1.0f : -1.0f;

    sample.alphaColorProduct = float4(
        sign * sigma,
        sign * sigma * color.x,
        sign * sigma * color.y,
        sign * sigma * color.z
    );

    return sample;
}

//------------------------------------------------------------------------------
// Main Backward Kernel
// Computes gradients using reverse-mode automatic differentiation
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

    // Get loss gradient for this ray's output color
    // dLossPerColor = [dL/dR, dL/dG, dL/dB, dL/dT]
    float4 dLdOutput = params.dLossPerColor[rayIdx];
    float3 dLdColor = float3(dLdOutput.x, dLdOutput.y, dLdOutput.z);
    float dLdTransmittance = dLdOutput.w;

    // Get ray info
    float3 rayOrigin = params.rayOrigins[rayIdx];
    float3 rayDir = normalize3(params.rayDirections[rayIdx]);

    // Get forward pass results
    VolumeIntegrationState finalState = params.finalIntegrationState[rayIdx];
    uint numSamples = params.numSamplesPerRay[rayIdx];

    if (numSamples == 0) {
        return;  // No samples to backprop through
    }

    // Reconstruct integration states by replaying forward pass
    // This is necessary for correct gradient computation

    VolumeIntegrationState states[64];  // Max samples supported
    GaussianSample samples[64];
    uint primIndices[64];
    bool isEntries[64];

    // Initialize first state (ray start)
    states[0] = VolumeIntegrationState.makeEmpty();
    states[0].rayT = params.tmin;

    // Replay forward pass to reconstruct all intermediate states
    for (uint i = 0; i < numSamples; i++) {
        int hitId = params.hitPrimitiveIds[rayIdx * params.maxSamplesPerRay + i];
        if (hitId < 0) break;

        uint primIdx = uint(hitId) / 2;
        bool isEntry = (uint(hitId) % 2) == 0;

        primIndices[i] = primIdx;
        isEntries[i] = isEntry;

        // Get primitive data
        float3 mean = params.model.means[primIdx];
        float3 scale = params.model.scales[primIdx];
        float4 quat = params.model.quats[primIdx];
        float density = params.model.densities[primIdx];

        // Recompute intersection t
        IntersectResult intersect = rayIntersectGaussian(rayOrigin, rayDir, mean, scale, quat);
        float t = isEntry ? intersect.tNear : intersect.tFar;

        // Get color at this point
        float3 viewDir = -rayDir;
        float3 color = evaluateColor(viewDir, params.model.features, primIdx,
                                    params.featureSize, params.shDegree);

        // Compute Gaussian sample
        samples[i] = computeGaussianSample(
            t, rayOrigin, rayDir, mean, scale, quat, density, color, isEntry
        );

        // Forward step to get next state
        if (i + 1 < 64) {
            states[i + 1] = forwardIntegrationStep(states[i], samples[i], params.tmax);
        }
    }

    // =========================================================================
    // Backward pass: propagate gradients in reverse order
    // Using chain rule: dL/dθ = dL/dC · dC/dθ
    // =========================================================================

    // Initialize gradients w.r.t. final state
    float3 dLdAccumulatedColor = dLdColor;
    float dLdLogTransmittance = dLdTransmittance * safeExp(-finalState.logTransmittance) * (-1.0f);

    // Iterate backwards through samples
    for (int i = int(numSamples) - 1; i >= 0; i--) {
        uint primIdx = primIndices[i];
        bool isEntry = isEntries[i];
        GaussianSample sample = samples[i];
        VolumeIntegrationState prevState = states[i];

        // Get primitive data
        float3 mean = params.model.means[primIdx];
        float3 scale = params.model.scales[primIdx];
        float4 quat = params.model.quats[primIdx];
        float density = params.model.densities[primIdx];

        // Compute gradients using chain rule for volume rendering equation:
        // C_new = C_old + α̂ · (α·c) · T
        // log(T)_new = log(T)_old + α · Δt

        float deltaT = max(sample.t - prevState.rayT, 0.0f);
        float alpha = sample.alphaColorProduct.x;
        float transmittance = safeExp(-prevState.logTransmittance);
        float opticalDepth = alpha * deltaT;
        float discreteAlpha = clampValue(1.0f - safeExp(-opticalDepth), 0.0f, 1.0f);

        // Density-weighted color
        float3 alphaColor = float3(
            sample.alphaColorProduct.y,
            sample.alphaColorProduct.z,
            sample.alphaColorProduct.w
        );

        // dL/d(α̂) through color accumulation
        float dDiscreteAlpha_dOpticalDepth = safeExp(-opticalDepth);
        float dOpticalDepth_dAlpha = deltaT;

        float3 dLdAlphaColor = discreteAlpha * transmittance * dLdAccumulatedColor;
        float dLdDiscreteAlpha = dot3(alphaColor * transmittance, dLdAccumulatedColor);

        float dLdOpticalDepth = dLdDiscreteAlpha * dDiscreteAlpha_dOpticalDepth + dLdLogTransmittance;
        float dLdAlpha = dLdOpticalDepth * dOpticalDepth_dAlpha;

        // Handle entry/exit sign
        float sign = isEntry ? 1.0f : -1.0f;

        // Gradient for density: dL/dσ
        float dLdDensity = sign * dLdAlpha;

        // Accumulate density gradient atomically
        float temp;
        InterlockedAdd(params.model.dDensities[primIdx], dLdDensity, temp);

        // Gradient for color: dL/dc
        float3 dLdColorFeature = sign * alpha * dLdAlphaColor;

        // For RGB features (no SH)
        if (params.featureSize == 3) {
            atomicAddFloat3(params.model.dFeatures, primIdx, dLdColorFeature);
        }

        // Gradient for position (mean): dL/dμ
        // Approximation: gradient points toward/away from ray based on density contribution
        float3 rayPoint = rayOrigin + sample.t * rayDir;
        float3 diff = rayPoint - mean;
        float3 dLdMean = dLdDensity * diff * density * 0.01f;

        atomicAddFloat3(params.model.dMeans, primIdx, dLdMean);

        // Gradient for scale: dL/ds
        float3 dLdScale = dLdDensity * scale * 0.001f;
        atomicAddFloat3(params.model.dScales, primIdx, dLdScale);

        // Propagate gradients to previous state
        // dL/d(log(T)_old) += dL/d(log(T)_new) + contribution from color term
        dLdLogTransmittance = dLdLogTransmittance + dLdOpticalDepth;
    }
}

//------------------------------------------------------------------------------
// Entry Point for CUDA Launch
//------------------------------------------------------------------------------

[CudaKernel]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform BackwardParams params)
{
    backwardKernel(dispatchThreadID, params);
}
