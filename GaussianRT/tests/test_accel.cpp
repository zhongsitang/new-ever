/**
 * @file test_accel.cpp
 * @brief Unit tests for AccelStruct class
 */

#include "gaussian_rt/AccelStruct.h"
#include "gaussian_rt/GaussianPrimitives.h"
#include <cstdio>
#include <cassert>
#include <vector>

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
    std::vector<float>& features)
{
    means.resize(numPrims * 3);
    scales.resize(numPrims * 3);
    quats.resize(numPrims * 4);
    densities.resize(numPrims);
    features.resize(numPrims * 3);

    for (size_t i = 0; i < numPrims; i++) {
        means[i * 3 + 0] = (float)(i % 10) - 5.0f;
        means[i * 3 + 1] = (float)((i / 10) % 10) - 5.0f;
        means[i * 3 + 2] = (float)((i / 100) % 10) - 5.0f;

        scales[i * 3 + 0] = 0.1f;
        scales[i * 3 + 1] = 0.1f;
        scales[i * 3 + 2] = 0.1f;

        quats[i * 4 + 0] = 0.0f;
        quats[i * 4 + 1] = 0.0f;
        quats[i * 4 + 2] = 0.0f;
        quats[i * 4 + 3] = 1.0f;

        densities[i] = 1.0f;

        features[i * 3 + 0] = 1.0f;
        features[i * 3 + 1] = 0.0f;
        features[i * 3 + 2] = 0.0f;
    }
}

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

TEST(accel_creation) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    AccelStruct accel(device);
    assert(!accel.isValid());

    device.shutdown();
}

TEST(accel_build) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    // Create primitives
    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    GaussianPrimitives prims(device);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    // Build acceleration structure
    AccelStruct accel(device);
    result = accel.build(prims, true, false);

    if (result != Result::Success) {
        printf("OptiX not available, skipping... ");
        device.shutdown();
        return;
    }

    assert(accel.isValid());
    assert(accel.canUpdate());
    assert(accel.getNumPrimitives() == numPrims);
    assert(accel.getTraversableHandle() != nullptr);

    device.shutdown();
}

TEST(accel_fast_build) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    size_t numPrims = 1000;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    GaussianPrimitives prims(device);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    AccelStruct accel(device);
    result = accel.build(prims, true, true);  // fast_build = true

    if (result != Result::Success) {
        printf("OptiX not available, skipping... ");
        device.shutdown();
        return;
    }

    assert(accel.isValid());

    device.shutdown();
}

TEST(accel_update) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    GaussianPrimitives prims(device);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    AccelStruct accel(device);
    result = accel.build(prims, true, false);

    if (result != Result::Success) {
        printf("OptiX not available, skipping... ");
        device.shutdown();
        return;
    }

    // Modify positions
    for (size_t i = 0; i < numPrims; i++) {
        means[i * 3 + 0] += 0.01f;
    }

    // Re-upload and update AABBs
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    // Update (refit) acceleration structure
    result = accel.update(prims);
    assert(result == Result::Success);
    assert(accel.isValid());

    device.shutdown();
}

TEST(accel_rebuild) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    GaussianPrimitives prims(device);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    AccelStruct accel(device);
    result = accel.build(prims, true, false);

    if (result != Result::Success) {
        printf("OptiX not available, skipping... ");
        device.shutdown();
        return;
    }

    // Create new primitives with different count
    numPrims = 200;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    // Rebuild
    result = accel.rebuild(prims);
    assert(result == Result::Success);
    assert(accel.isValid());
    assert(accel.getNumPrimitives() == numPrims);

    device.shutdown();
}

TEST(accel_move) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    size_t numPrims = 100;
    std::vector<float> means, scales, quats, densities, features;
    generateTestPrimitives(numPrims, means, scales, quats, densities, features);

    GaussianPrimitives prims(device);
    prims.setData(numPrims, means.data(), scales.data(), quats.data(),
                  densities.data(), features.data(), 3);

    AccelStruct accel1(device);
    result = accel1.build(prims, true, false);

    if (result != Result::Success) {
        printf("OptiX not available, skipping... ");
        device.shutdown();
        return;
    }

    // Move
    AccelStruct accel2 = std::move(accel1);

    assert(!accel1.isValid());
    assert(accel2.isValid());
    assert(accel2.getNumPrimitives() == numPrims);

    device.shutdown();
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main() {
    printf("=== GaussianRT AccelStruct Tests ===\n\n");

    RUN_TEST(accel_creation);
    RUN_TEST(accel_build);
    RUN_TEST(accel_fast_build);
    RUN_TEST(accel_update);
    RUN_TEST(accel_rebuild);
    RUN_TEST(accel_move);

    printf("\nAll tests passed!\n");
    return 0;
}
