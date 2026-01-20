// GaussianRT - Common type definitions
// Shared between CPU and GPU code

module types;

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

static const float LOG_TRANSMITTANCE_CUTOFF = -10.0f;
static const uint HIT_BUFFER_SIZE = 16;
static const float PI = 3.14159265358979323846f;
static const float EPSILON = 1e-6f;

//------------------------------------------------------------------------------
// Basic Math Types
//------------------------------------------------------------------------------

// Spherical harmonics constants
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;

static const float SH_C2_0 = 1.0925484305920792f;
static const float SH_C2_1 = 1.0925484305920792f;
static const float SH_C2_2 = 0.31539156525252005f;
static const float SH_C2_3 = 1.0925484305920792f;
static const float SH_C2_4 = 0.5462742152960396f;

static const float SH_C3_0 = 0.5900435899266435f;
static const float SH_C3_1 = 2.890611442640554f;
static const float SH_C3_2 = 0.4570457994644658f;
static const float SH_C3_3 = 0.3731763325901154f;
static const float SH_C3_4 = 0.4570457994644658f;
static const float SH_C3_5 = 1.445305721320277f;
static const float SH_C3_6 = 0.5900435899266435f;

//------------------------------------------------------------------------------
// Gaussian Primitive
//------------------------------------------------------------------------------

struct GaussianPrimitive {
    float3 mean;        // Center position
    float density;      // Opacity/density σ

    float3 scale;       // Scale factors (radii)
    float _pad0;

    float4 quat;        // Rotation quaternion (x, y, z, w)
};

//------------------------------------------------------------------------------
// Volume Integration State
// Tracks the accumulated result of the volume rendering equation:
//   C = ∫ σ(t) · c(t) · T(t) dt
// where:
//   T(t) = exp(-∫₀ᵗ σ(s) ds)  is the transmittance
//   σ(t) is the density at point t
//   c(t) is the color at point t
//------------------------------------------------------------------------------

struct VolumeIntegrationState : IDifferentiable {
    float logTransmittance;     // log(T) where T = exp(-∫σdt), for numerical stability
    float3 accumulatedColor;    // C = ∫σ·c·T dt (accumulated color)

    float rayT;                 // Current ray parameter t
    float4 accumulatedAlphaRGB; // Running sum: [Σα, Σ(α·R), Σ(α·G), Σ(α·B)]

    // Factory for empty state (ray start)
    [Differentiable]
    static VolumeIntegrationState makeEmpty() {
        VolumeIntegrationState state;
        state.logTransmittance = 0.0f;      // T = 1 (fully transparent)
        state.accumulatedColor = float3(0.0f);
        state.rayT = 0.0f;
        state.accumulatedAlphaRGB = float4(0.0f);
        return state;
    }
};

//------------------------------------------------------------------------------
// Gaussian Sample
// Represents the contribution of a single Gaussian at a ray intersection point.
// Used in the discrete approximation of the volume rendering integral.
//------------------------------------------------------------------------------

struct GaussianSample : IDifferentiable {
    float t;                    // Ray parameter at this sample
    float4 alphaColorProduct;   // [α, α·R, α·G, α·B] - opacity-weighted color
                                // α = 1 - exp(-σ·Δt) is the discrete opacity
};

//------------------------------------------------------------------------------
// Ray Payload for Tracing
//------------------------------------------------------------------------------

struct RayPayload {
    VolumeIntegrationState integrationState;

    // Hit buffer: stores (t, primitiveId) pairs for front-to-back sorting
    // Using uint for bit manipulation (float bits stored as uint)
    uint hitBuffer[HIT_BUFFER_SIZE * 2];
    uint hitCount;

    // Initialize payload for a new ray
    static RayPayload create() {
        RayPayload payload;
        payload.integrationState = VolumeIntegrationState.makeEmpty();
        payload.hitCount = 0;

        // Initialize hit buffer with large t values (far away)
        [ForceUnroll]
        for (uint i = 0; i < HIT_BUFFER_SIZE; i++) {
            payload.hitBuffer[i * 2] = asuint(1e10f);
            payload.hitBuffer[i * 2 + 1] = 0xFFFFFFFF;
        }

        return payload;
    }
};

//------------------------------------------------------------------------------
// Launch Parameters
//------------------------------------------------------------------------------

struct LaunchParams {
    // Output buffers
    RWStructuredBuffer<float4> colorOutput;
    RWStructuredBuffer<VolumeIntegrationState> stateOutput;
    RWStructuredBuffer<int> hitPrimitiveIds;        // Visited primitive IDs for backward pass
    RWStructuredBuffer<uint> numSamplesPerRay;      // Number of samples processed per ray
    RWStructuredBuffer<float4> lastSampleAlphaRGB;  // Last sample's alpha-weighted color

    // Input rays
    StructuredBuffer<float3> rayOrigins;
    StructuredBuffer<float3> rayDirections;
    uint numRays;

    // Gaussian primitives
    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;
    uint numPrimitives;
    uint featureSize;

    // Render parameters
    float tmin;
    float tmax;
    uint maxSamplesPerRay;      // Maximum Gaussian intersections per ray
    uint shDegree;
    float maxPrimSize;

    // Acceleration structure
    RaytracingAccelerationStructure accel;
};

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

// Quaternion to rotation matrix
[Differentiable]
float3x3 quatToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    float3x3 m;
    m[0] = float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy));
    m[1] = float3(2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx));
    m[2] = float3(2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));

    return m;
}

// Safe division (avoid divide by zero)
[Differentiable]
float safeDivide(float a, float b) {
    return abs(b) < EPSILON ? 0.0f : a / b;
}

// Clamp value
[Differentiable]
float clampValue(float v, float minV, float maxV) {
    return max(min(v, maxV), minV);
}

// Clamp float3
[Differentiable]
float3 clampFloat3(float3 v, float minV, float maxV) {
    return float3(
        clampValue(v.x, minV, maxV),
        clampValue(v.y, minV, maxV),
        clampValue(v.z, minV, maxV)
    );
}
