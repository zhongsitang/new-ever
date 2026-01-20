/**
 * @file test_types.cpp
 * @brief Unit tests for GaussianRT type definitions
 */

#include "gaussian_rt/Types.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace gaussian_rt;

// Simple test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))
#define ASSERT_TRUE(x) assert(x)

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

TEST(Float3_creation) {
    Float3 v1;
    ASSERT_EQ(v1.x, 0.0f);
    ASSERT_EQ(v1.y, 0.0f);
    ASSERT_EQ(v1.z, 0.0f);

    Float3 v2(1.0f, 2.0f, 3.0f);
    ASSERT_EQ(v2.x, 1.0f);
    ASSERT_EQ(v2.y, 2.0f);
    ASSERT_EQ(v2.z, 3.0f);
}

TEST(Float4_creation) {
    Float4 v1;
    ASSERT_EQ(v1.x, 0.0f);
    ASSERT_EQ(v1.w, 0.0f);

    Float4 v2(1.0f, 2.0f, 3.0f, 4.0f);
    ASSERT_EQ(v2.x, 1.0f);
    ASSERT_EQ(v2.w, 4.0f);
}

TEST(GaussianData_alignment) {
    // Check struct size and alignment
    ASSERT_TRUE(sizeof(GaussianData) >= 48);  // At least 12 floats
    ASSERT_TRUE(alignof(GaussianData) == 16);
}

TEST(VolumeIntegrationState_initialization) {
    // Test volume integration state used in differentiable rendering
    VolumeIntegrationState state;
    state.logTransmittance = 0.0f;      // log(T) where T is transmittance
    state.accumulatedColor = Float3(0.0f, 0.0f, 0.0f);  // Accumulated color C
    state.rayT = 0.0f;                  // Current ray parameter t
    state.accumulatedAlphaRGB = Float4(0.0f, 0.0f, 0.0f, 0.0f);  // [Σα, Σ(α·R), Σ(α·G), Σ(α·B)]

    ASSERT_EQ(state.logTransmittance, 0.0f);
    ASSERT_EQ(state.rayT, 0.0f);
}

TEST(RenderParams_defaults) {
    RenderParams params;
    ASSERT_NEAR(params.tmin, 0.01f, 1e-6f);
    ASSERT_NEAR(params.tmax, 100.0f, 1e-6f);
    ASSERT_EQ(params.maxIters, 512u);
    ASSERT_EQ(params.shDegree, 0u);
}

TEST(sh_coeffs_count) {
    ASSERT_EQ(shCoeffsCount(0), 1u);
    ASSERT_EQ(shCoeffsCount(1), 4u);
    ASSERT_EQ(shCoeffsCount(2), 9u);
    ASSERT_EQ(shCoeffsCount(3), 16u);
}

TEST(sh_feature_size) {
    ASSERT_EQ(shFeatureSize(0), 3u);   // 1 * 3
    ASSERT_EQ(shFeatureSize(1), 12u);  // 4 * 3
    ASSERT_EQ(shFeatureSize(2), 27u);  // 9 * 3
    ASSERT_EQ(shFeatureSize(3), 48u);  // 16 * 3
}

TEST(result_to_string) {
    ASSERT_TRUE(resultToString(Result::Success) != nullptr);
    ASSERT_TRUE(resultToString(Result::ErrorInvalidArgument) != nullptr);
    ASSERT_TRUE(resultToString(Result::ErrorOutOfMemory) != nullptr);
}

TEST(constants) {
    ASSERT_NEAR(LOG_TRANSMITTANCE_CUTOFF, -10.0f, 1e-6f);
    ASSERT_EQ(HIT_BUFFER_SIZE, 16u);
    ASSERT_EQ(MAX_SH_DEGREE, 3u);
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main() {
    printf("=== GaussianRT Type Tests ===\n\n");

    RUN_TEST(Float3_creation);
    RUN_TEST(Float4_creation);
    RUN_TEST(GaussianData_alignment);
    RUN_TEST(VolumeIntegrationState_initialization);
    RUN_TEST(RenderParams_defaults);
    RUN_TEST(sh_coeffs_count);
    RUN_TEST(sh_feature_size);
    RUN_TEST(result_to_string);
    RUN_TEST(constants);

    printf("\nAll tests passed!\n");
    return 0;
}
