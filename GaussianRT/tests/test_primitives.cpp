/**
 * @file test_primitives.cpp
 * @brief Unit tests for GaussianPrimitives class
 */

#include "gaussian_rt/PrimitiveSet.h"
#include <cstdio>
#include <cassert>
#include <vector>
#include <cmath>

using namespace gaussian_rt;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

void generateTestPrimitives(
    size_t numPrims,
    std::vector<float>& means,
    std::vector<float>& scales,
    std::vector<float>& quats,
    std::vector<float>& densities,
    std::vector<float>& features,
    size_t featureSize = 3)
{
    means.resize(numPrims * 3);
    scales.resize(numPrims * 3);
    quats.resize(numPrims * 4);
    densities.resize(numPrims);
    features.resize(numPrims * featureSize);

    for (size_t i = 0; i < numPrims; i++) {
        // Random-ish positions
        means[i * 3 + 0] = (float)(i % 10) - 5.0f;
        means[i * 3 + 1] = (float)((i / 10) % 10) - 5.0f;
        means[i * 3 + 2] = (float)((i / 100) % 10) - 5.0f;

        // Unit scale
        scales[i * 3 + 0] = 0.1f;
        scales[i * 3 + 1] = 0.1f;
        scales[i * 3 + 2] = 0.1f;

        // Identity quaternion
        quats[i * 4 + 0] = 0.0f;
        quats[i * 4 + 1] = 0.0f;
        quats[i * 4 + 2] = 0.0f;
        quats[i * 4 + 3] = 1.0f;

        // Unit density
        densities[i] = 1.0f;

        // RGB colors
        for (size_t j = 0; j < featureSize; j++) {
            features[i * featureSize + j] = (float)(i % 256) / 255.0f;
        }
    }
}

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

TEST(primitives_creation) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    PrimitiveSet prims(device);
    assert(!prims.isValid());
    assert(prims.getNumPrimitives() == 0);

    device.shutdown();
}

TEST(primitives_set_data) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    PrimitiveSet prims(device);

    // Generate test data
    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    // Set data
    result = prims.setData(
        numPrims,
        means.data(),
        scales.data(),
        quats.data(),
        densities.data(),
        features.data(),
        3
    );

    assert(result == Result::Success);
    assert(prims.isValid());
    assert(prims.getNumPrimitives() == numPrims);
    assert(prims.getFeatureSize() == 3);
    assert(prims.getSHDegree() == 0);

    // Check device pointers are valid
    assert(prims.getMeansDevice() != nullptr);
    assert(prims.getScalesDevice() != nullptr);
    assert(prims.getQuatsDevice() != nullptr);
    assert(prims.getDensitiesDevice() != nullptr);
    assert(prims.getFeaturesDevice() != nullptr);
    assert(prims.getAABBsDevice() != nullptr);

    device.shutdown();
}

TEST(primitives_sh_degree) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    // Test SH degree inference
    {
        PrimitiveSet prims(device);
        size_t numPrims = 10;
        std::vector<float> means, scales, quats, densities, features;
        generateTestPrimitives(numPrims, means, scales, quats, densities, features, 3);
        prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                      densities.data(), features.data(), 3);
        assert(prims.getSHDegree() == 0);
    }

    {
        PrimitiveSet prims(device);
        size_t numPrims = 10;
        std::vector<float> means, scales, quats, densities, features;
        generateTestPrimitives(numPrims, means, scales, quats, densities, features, 12);
        prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                      densities.data(), features.data(), 12);
        assert(prims.getSHDegree() == 1);
    }

    {
        PrimitiveSet prims(device);
        size_t numPrims = 10;
        std::vector<float> means, scales, quats, densities, features;
        generateTestPrimitives(numPrims, means, scales, quats, densities, features, 27);
        prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                      densities.data(), features.data(), 27);
        assert(prims.getSHDegree() == 2);
    }

    {
        PrimitiveSet prims(device);
        size_t numPrims = 10;
        std::vector<float> means, scales, quats, densities, features;
        generateTestPrimitives(numPrims, means, scales, quats, densities, features, 48);
        prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                      densities.data(), features.data(), 48);
        assert(prims.getSHDegree() == 3);
    }

    device.shutdown();
}

TEST(primitives_update_aabbs) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    PrimitiveSet prims(device);

    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    // Update AABBs
    result = prims.updateAABBs();
    assert(result == Result::Success);
    assert(prims.getAABBsDevice() != nullptr);

    device.shutdown();
}

TEST(primitives_move) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    PrimitiveSet prims1(device);
    prims1.setData(numPrims, means.data(), scales.data(), quats.data(),
                   densities.data(), features.data(), 3);

    // Move
    PrimitiveSet prims2 = std::move(prims1);

    assert(!prims1.isValid());
    assert(prims2.isValid());
    assert(prims2.getNumPrimitives() == numPrims);

    device.shutdown();
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main() {
    printf("=== GaussianRT Primitives Tests ===\n\n");

    RUN_TEST(primitives_creation);
    RUN_TEST(primitives_set_data);
    RUN_TEST(primitives_sh_degree);
    RUN_TEST(primitives_update_aabbs);
    RUN_TEST(primitives_move);

    printf("\nAll tests passed!\n");
    return 0;
}
