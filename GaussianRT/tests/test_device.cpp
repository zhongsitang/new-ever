/**
 * @file test_device.cpp
 * @brief Unit tests for Device class
 */

#include "gaussian_rt/Device.h"
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

#define SKIP_TEST(name, reason) do { \
    printf("Skipping %s: %s\n", #name, reason); \
} while(0)

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

TEST(device_creation) {
    Device device;
    assert(!device.isInitialized());
}

TEST(device_initialization) {
    Device device;

    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    assert(device.isInitialized());
    assert(device.getCudaDeviceId() == 0);

    device.shutdown();
    assert(!device.isInitialized());
}

TEST(buffer_operations) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    // Create buffer
    size_t size = 1024 * sizeof(float);
    void* buffer = device.createBuffer(size);
    assert(buffer != nullptr);

    // Upload data
    std::vector<float> hostData(1024, 1.0f);
    result = device.uploadToBuffer(buffer, hostData.data(), size);
    assert(result == Result::Success);

    // Download and verify
    std::vector<float> downloadData(1024, 0.0f);
    result = device.downloadFromBuffer(buffer, downloadData.data(), size);
    assert(result == Result::Success);

    for (int i = 0; i < 1024; i++) {
        assert(downloadData[i] == 1.0f);
    }

    // Free buffer
    device.freeBuffer(buffer);

    device.shutdown();
}

TEST(buffer_with_initial_data) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    std::vector<float> hostData = {1.0f, 2.0f, 3.0f, 4.0f};
    void* buffer = device.createBuffer(hostData.size() * sizeof(float), hostData.data());
    assert(buffer != nullptr);

    // Download and verify
    std::vector<float> downloadData(4, 0.0f);
    result = device.downloadFromBuffer(buffer, downloadData.data(), 4 * sizeof(float));
    assert(result == Result::Success);

    assert(downloadData[0] == 1.0f);
    assert(downloadData[1] == 2.0f);
    assert(downloadData[2] == 3.0f);
    assert(downloadData[3] == 4.0f);

    device.freeBuffer(buffer);
    device.shutdown();
}

TEST(synchronize) {
    Device device;
    Result result = device.initialize(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    // Should not crash
    device.synchronize();

    device.shutdown();
}

TEST(global_device) {
    Result result = initializeGlobalDevice(0);
    if (result != Result::Success) {
        printf("CUDA device not available, skipping... ");
        return;
    }

    Device& device = getGlobalDevice();
    assert(device.isInitialized());

    shutdownGlobalDevice();
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main() {
    printf("=== GaussianRT Device Tests ===\n\n");

    RUN_TEST(device_creation);
    RUN_TEST(device_initialization);
    RUN_TEST(buffer_operations);
    RUN_TEST(buffer_with_initial_data);
    RUN_TEST(synchronize);
    RUN_TEST(global_device);

    printf("\nAll tests passed!\n");
    return 0;
}
