// GaussianRT - Forward Ray Tracing Shaders
// Hardware-accelerated differentiable volume rendering for ellipsoid primitives

import common.types;
import common.math;
import common.sh;
import common.intersect;

//------------------------------------------------------------------------------
// Global Parameters
//------------------------------------------------------------------------------

// Launch parameters (set from CPU)
[[vk::push_constant]]
ConstantBuffer<LaunchParams> params;

//------------------------------------------------------------------------------
// Volume Rendering Integration
// Implements the discrete volume rendering equation:
//   C = Σᵢ αᵢ · cᵢ · Tᵢ
// where:
//   αᵢ = 1 - exp(-σᵢ · Δtᵢ)  is the discrete opacity
//   Tᵢ = Πⱼ₌₁^{i-1}(1 - αⱼ)  is the accumulated transmittance
//------------------------------------------------------------------------------

// Accumulate a volume sample into the integration state (differentiable)
[Differentiable]
VolumeIntegrationState accumulateSample(
    VolumeIntegrationState state,
    VolumeSample sample,
    float tMax)
{
    VolumeIntegrationState newState = state;

    // Compute step size Δt
    float deltaT = max(sample.t - state.rayT, 0.0f);

    // Extract density (α) and density-weighted color (α·c) from sample
    float alpha = sample.alphaColorProduct.x;
    float3 alphaColor = float3(
        sample.alphaColorProduct.y,
        sample.alphaColorProduct.z,
        sample.alphaColorProduct.w
    );

    // Volume rendering integration using log-space for numerical stability:
    //   T = exp(-∫σdt) → log(T) = -∫σdt
    //   α̂ = 1 - exp(-σ·Δt)  (discrete opacity)
    //   C += α̂ · c · T       (color contribution)

    float opticalDepth = alpha * deltaT;                            // σ·Δt
    float transmittance = safeExp(-state.logTransmittance);         // T = exp(-log(T))
    float discreteAlpha = clampValue(1.0f - safeExp(-opticalDepth), 0.0f, 1.0f);  // α̂

    // Accumulate color: C += α̂ · c · T
    newState.accumulatedColor = state.accumulatedColor + discreteAlpha * alphaColor * transmittance;

    // Update log-transmittance: log(T) += σ·Δt
    newState.logTransmittance = state.logTransmittance + opticalDepth;

    // Update ray position
    newState.rayT = sample.t;

    // Accumulate running sum of alpha-weighted colors (for backward pass)
    newState.accumulatedAlphaRGB = state.accumulatedAlphaRGB + sample.alphaColorProduct;

    return newState;
}

//------------------------------------------------------------------------------
// Ray Generation Shader
//------------------------------------------------------------------------------

[shader("raygeneration")]
void __raygen__main() {
    uint rayIdx = DispatchRaysIndex().x;

    if (rayIdx >= params.numRays) {
        return;
    }

    // Get ray
    float3 origin = params.rayOrigins[rayIdx];
    float3 direction = params.rayDirections[rayIdx];

    // Normalize direction
    direction = normalize3(direction);

    // Initialize volume integration state
    VolumeIntegrationState integrationState = VolumeIntegrationState.makeEmpty();
    integrationState.rayT = params.tmin;

    uint sampleCount = 0;

    // Main ray marching loop
    while (integrationState.logTransmittance > LOG_TRANSMITTANCE_CUTOFF &&
           sampleCount < params.maxSamplesPerRay) {

        // Create payload for this trace iteration
        RayPayload payload = RayPayload.create();
        payload.integrationState = integrationState;

        // Setup ray descriptor
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = integrationState.rayT + EPSILON;
        ray.TMax = params.tmax;

        // Trace ray to collect Gaussian intersections
        TraceRay(
            params.accel,
            RAY_FLAG_NONE,
            0xFF,
            0,  // Hit group index
            1,  // Hit group stride
            0,  // Miss shader index
            ray,
            payload
        );

        // Process collected hits (sorted front-to-back)
        uint numHits = min(payload.hitCount, HIT_BUFFER_SIZE);

        if (numHits == 0) {
            break;  // No more primitive intersections
        }

        [ForceUnroll]
        for (uint i = 0; i < HIT_BUFFER_SIZE; i++) {
            if (i >= numHits) break;

            float t = asfloat(payload.hitBuffer[i * 2]);
            uint hitId = payload.hitBuffer[i * 2 + 1];

            if (hitId == 0xFFFFFFFF) break;

            // Decode primitive index and entry/exit flag
            // hitId = primitiveIndex * 2 + (isEntry ? 0 : 1)
            uint primIdx = hitId / 2;
            bool isEntry = (hitId % 2) == 0;

            // Store hit ID for backward pass gradient computation
            params.hitPrimitiveIds[rayIdx * params.maxSamplesPerRay + sampleCount] = int(hitId);

            // Get primitive data
            float3 position = params.positions[primIdx];
            float3 scale = params.scales[primIdx];
            float4 orientation = params.orientations[primIdx];
            float density = params.densities[primIdx];

            // Evaluate view-dependent color (SH or RGB)
            float3 viewDir = -direction;
            float3 color = evaluateColor(viewDir, params.features, primIdx,
                                        params.featureSize, params.shDegree);

            // Create volume sample at this intersection
            VolumeSample sample = createVolumeSample(
                t, origin, direction,
                position, scale, orientation, density, color,
                isEntry
            );

            // Accumulate sample into integration state
            integrationState = accumulateSample(integrationState, sample, params.tmax);

            sampleCount++;

            // Early termination if ray is essentially opaque
            if (sampleCount >= params.maxSamplesPerRay ||
                integrationState.logTransmittance <= LOG_TRANSMITTANCE_CUTOFF) {
                break;
            }
        }

        // Update ray start position for next iteration
        if (numHits > 0) {
            float lastT = asfloat(payload.hitBuffer[(numHits - 1) * 2]);
            integrationState.rayT = lastT + EPSILON;
        }
    }

    // Mark remaining hit slots as unused (-1) for backward pass
    for (uint j = sampleCount; j < params.maxSamplesPerRay; j++) {
        params.hitPrimitiveIds[rayIdx * params.maxSamplesPerRay + j] = -1;
    }

    // Compute final pixel color
    // Remaining transmittance shows through to background (black)
    float finalTransmittance = safeExp(-integrationState.logTransmittance);
    float3 finalColor = integrationState.accumulatedColor;

    // Write outputs
    params.colorOutput[rayIdx] = float4(finalColor, finalTransmittance);
    params.stateOutput[rayIdx] = integrationState;
    params.numSamplesPerRay[rayIdx] = sampleCount;
    params.lastSampleAlphaRGB[rayIdx] = integrationState.accumulatedAlphaRGB;
}

//------------------------------------------------------------------------------
// Miss Shader
//------------------------------------------------------------------------------

[shader("miss")]
void __miss__main(inout RayPayload payload) {
    // Background color is handled in raygen shader
    // No action needed here
}

//------------------------------------------------------------------------------
// Intersection Shader (Custom Ellipsoid Primitive)
//------------------------------------------------------------------------------

[shader("intersection")]
void __intersection__ellipsoid() {
    uint primIdx = PrimitiveIndex();

    // Get primitive data
    float3 position = params.positions[primIdx];
    float3 scale = params.scales[primIdx];
    float4 orientation = params.orientations[primIdx];

    // Get ray in world space
    float3 rayOrigin = WorldRayOrigin();
    float3 rayDir = WorldRayDirection();

    // Compute ray-ellipsoid intersection (3-sigma bounding ellipsoid)
    IntersectResult result = rayIntersectEllipsoidBounded(
        rayOrigin, rayDir, position, scale, orientation
    );

    if (!result.hit) {
        return;
    }

    float tMin = RayTMin();
    float tMax = RayTCurrent();

    // Report entry point (hitKind = 0: entering the primitive)
    if (result.tNear >= tMin && result.tNear <= tMax) {
        ReportHit(result.tNear, 0, /* no attributes */);
    }

    // Report exit point (hitKind = 1: exiting the primitive)
    if (result.tFar >= tMin && result.tFar <= tMax) {
        ReportHit(result.tFar, 1, /* no attributes */);
    }
}

//------------------------------------------------------------------------------
// Any-Hit Shader (Front-to-Back Sorting)
// Collects and sorts primitive intersections by distance
//------------------------------------------------------------------------------

[shader("anyhit")]
void __anyhit__sorting(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    float t = RayTCurrent();
    uint primIdx = PrimitiveIndex();
    uint hitKind = HitKind();  // 0 = entry, 1 = exit

    // Encode hit: primitiveIndex * 2 + hitKind
    uint hitId = primIdx * 2 + hitKind;

    // Insert into sorted hit buffer using insertion sort
    // This maintains hits in front-to-back order for proper compositing
    uint tBits = asuint(t);

    [ForceUnroll]
    for (uint i = 0; i < HIT_BUFFER_SIZE; i++) {
        uint storedTBits = payload.hitBuffer[i * 2];

        if (tBits < storedTBits) {
            // Shift elements to make room for new hit
            [ForceUnroll]
            for (uint j = HIT_BUFFER_SIZE - 1; j > i; j--) {
                payload.hitBuffer[j * 2] = payload.hitBuffer[(j - 1) * 2];
                payload.hitBuffer[j * 2 + 1] = payload.hitBuffer[(j - 1) * 2 + 1];
            }

            // Insert new hit at position i
            payload.hitBuffer[i * 2] = tBits;
            payload.hitBuffer[i * 2 + 1] = hitId;

            payload.hitCount = min(payload.hitCount + 1, HIT_BUFFER_SIZE);
            break;
        }
    }

    // Ignore this hit to continue searching for more primitives
    IgnoreHit();
}
