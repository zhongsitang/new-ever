// GaussianRT - Forward Ray Tracing Shaders
// Hardware-accelerated differentiable Gaussian rendering

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
// Volume Rendering State Machine
//------------------------------------------------------------------------------

// Update spline state with a control point (differentiable)
[Differentiable]
SplineState updateSplineState(SplineState state, ControlPoint cp, float tMax) {
    SplineState newState = state;

    float dt = max(cp.t - state.t, 0.0f);

    // Extract density and color from Dirac pulse
    float alpha = cp.dirac.x;
    float3 colorWeighted = float3(cp.dirac.y, cp.dirac.z, cp.dirac.w);

    // Volume rendering integration:
    // C = ∫ σ(t) * c(t) * T(t) dt
    // where T(t) = exp(-∫₀ᵗ σ(s) ds)

    // Using log-space for numerical stability
    float area = alpha * dt;
    float T = safeExp(-state.logT);
    float alphaHat = clampValue(1.0f - safeExp(-area), 0.0f, 1.0f);

    // Accumulate color
    newState.C = state.C + alphaHat * colorWeighted * T;

    // Update transmittance
    newState.logT = state.logT + area;

    // Update position
    newState.t = cp.t;

    // Accumulate Dirac pulse
    newState.drgb = state.drgb + cp.dirac;

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

    // Initialize state
    SplineState state = SplineState.makeEmpty();
    state.t = params.tmin;

    uint iter = 0;

    // Main tracing loop
    while (state.logT > LOG_TRANSMITTANCE_CUTOFF && iter < params.maxIters) {
        // Create payload
        RayPayload payload = RayPayload.create();
        payload.state = state;

        // Setup ray descriptor
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = state.t + EPSILON;
        ray.TMax = params.tmax;

        // Trace ray
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

        // Process collected hits
        uint numHits = min(payload.hitCount, HIT_BUFFER_SIZE);

        if (numHits == 0) {
            break;  // No more intersections
        }

        [ForceUnroll]
        for (uint i = 0; i < HIT_BUFFER_SIZE; i++) {
            if (i >= numHits) break;

            float t = asfloat(payload.hitBuffer[i * 2]);
            uint triId = payload.hitBuffer[i * 2 + 1];

            if (triId == 0xFFFFFFFF) break;

            // Decode primitive index and entry/exit
            uint primIdx = triId / 2;
            bool isEntry = (triId % 2) == 0;

            // Store triangle ID for backward pass
            params.triCollection[rayIdx * params.maxIters + iter] = int(triId);

            // Get primitive data
            float3 mean = params.means[primIdx];
            float3 scale = params.scales[primIdx];
            float4 quat = params.quats[primIdx];
            float density = params.densities[primIdx];

            // Evaluate color
            float3 viewDir = -direction;
            float3 color = evaluateColor(viewDir, params.features, primIdx,
                                        params.featureSize, params.shDegree);

            // Generate control point
            ControlPoint cp = generateControlPoint(
                t, origin, direction,
                mean, scale, quat, density, color,
                isEntry
            );

            // Update state
            state = updateSplineState(state, cp, params.tmax);

            iter++;

            // Early termination
            if (iter >= params.maxIters || state.logT <= LOG_TRANSMITTANCE_CUTOFF) {
                break;
            }
        }

        // Update state for next iteration
        if (numHits > 0) {
            float lastT = asfloat(payload.hitBuffer[(numHits - 1) * 2]);
            state.t = lastT + EPSILON;
        }
    }

    // Fill remaining tri_collection with -1
    for (uint j = iter; j < params.maxIters; j++) {
        params.triCollection[rayIdx * params.maxIters + j] = -1;
    }

    // Compute final color with background
    float T = safeExp(-state.logT);
    float3 finalColor = state.C;  // Background is black

    // Write outputs
    params.colorOutput[rayIdx] = float4(finalColor, T);
    params.stateOutput[rayIdx] = state;
    params.itersOutput[rayIdx] = iter;
    params.lastDirac[rayIdx] = state.drgb;
}

//------------------------------------------------------------------------------
// Miss Shader
//------------------------------------------------------------------------------

[shader("miss")]
void __miss__main(inout RayPayload payload) {
    // Nothing to do - background is handled in raygen
}

//------------------------------------------------------------------------------
// Intersection Shader (Custom Ellipsoid)
//------------------------------------------------------------------------------

[shader("intersection")]
void __intersection__ellipsoid() {
    uint primIdx = PrimitiveIndex();

    // Get primitive data
    float3 mean = params.means[primIdx];
    float3 scale = params.scales[primIdx];
    float4 quat = params.quats[primIdx];

    // Get ray
    float3 rayOrigin = WorldRayOrigin();
    float3 rayDir = WorldRayDirection();

    // Compute intersection
    IntersectResult result = rayIntersectGaussian(
        rayOrigin, rayDir, mean, scale, quat
    );

    if (!result.hit) {
        return;
    }

    float tMin = RayTMin();
    float tMax = RayTCurrent();

    // Report entry point (hitKind = 0)
    if (result.tNear >= tMin && result.tNear <= tMax) {
        ReportHit(result.tNear, 0, /* no attributes */);
    }

    // Report exit point (hitKind = 1)
    if (result.tFar >= tMin && result.tFar <= tMax) {
        ReportHit(result.tFar, 1, /* no attributes */);
    }
}

//------------------------------------------------------------------------------
// Any-Hit Shader (Sorting)
//------------------------------------------------------------------------------

[shader("anyhit")]
void __anyhit__sorting(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    float t = RayTCurrent();
    uint primIdx = PrimitiveIndex();
    uint hitKind = HitKind();

    // Encode triId: primIdx * 2 + hitKind
    // hitKind: 0 = entry, 1 = exit
    uint triId = primIdx * 2 + hitKind;

    // Insertion sort to maintain sorted hit buffer
    uint tBits = asuint(t);

    [ForceUnroll]
    for (uint i = 0; i < HIT_BUFFER_SIZE; i++) {
        uint storedTBits = payload.hitBuffer[i * 2];

        if (tBits < storedTBits) {
            // Shift elements to make room
            [ForceUnroll]
            for (uint j = HIT_BUFFER_SIZE - 1; j > i; j--) {
                payload.hitBuffer[j * 2] = payload.hitBuffer[(j - 1) * 2];
                payload.hitBuffer[j * 2 + 1] = payload.hitBuffer[(j - 1) * 2 + 1];
            }

            // Insert new hit
            payload.hitBuffer[i * 2] = tBits;
            payload.hitBuffer[i * 2 + 1] = triId;

            payload.hitCount = min(payload.hitCount + 1, HIT_BUFFER_SIZE);
            break;
        }
    }

    // Ignore this hit to continue searching for more
    IgnoreHit();
}
