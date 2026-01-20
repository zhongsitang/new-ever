// Modern Splinetracer - Test Program
// Verifies the ray tracing implementation compiles and basic structures work
// Apache License 2.0

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>

#include "Types.h"
#include "Device.h"
#include "AccelerationStructure.h"
#include "RayTracer.h"

using namespace splinetracer;

// Generate test AABBs
std::vector<AABB> generate_test_aabbs(size_t count) {
    std::vector<AABB> aabbs(count);

    for (size_t i = 0; i < count; i++) {
        // Random position in [-10, 10]^3 with size ~1
        float x = (float)(rand() % 2000 - 1000) / 100.0f;
        float y = (float)(rand() % 2000 - 1000) / 100.0f;
        float z = (float)(rand() % 2000 - 1000) / 100.0f;
        float size = 0.5f + (float)(rand() % 100) / 100.0f;

        aabbs[i].minX = x - size;
        aabbs[i].minY = y - size;
        aabbs[i].minZ = z - size;
        aabbs[i].maxX = x + size;
        aabbs[i].maxY = y + size;
        aabbs[i].maxZ = z + size;
    }

    return aabbs;
}

// Test type sizes and alignment
bool test_type_layout() {
    std::cout << "Testing type layout... ";

    bool ok = true;

    // float2 should be 8 bytes
    if (sizeof(float2) != 8) {
        std::cout << "FAILED (float2 size: " << sizeof(float2) << ")\n";
        ok = false;
    }

    // float3 should be 16 bytes (with padding)
    if (sizeof(float3) != 16) {
        std::cout << "FAILED (float3 size: " << sizeof(float3) << ")\n";
        ok = false;
    }

    // float4 should be 16 bytes
    if (sizeof(float4) != 16) {
        std::cout << "FAILED (float4 size: " << sizeof(float4) << ")\n";
        ok = false;
    }

    // AABB should be 24 bytes
    if (sizeof(AABB) != 24) {
        std::cout << "FAILED (AABB size: " << sizeof(AABB) << ")\n";
        ok = false;
    }

    if (ok) {
        std::cout << "PASSED\n";
    }
    return ok;
}

// Test type constructors
bool test_type_constructors() {
    std::cout << "Testing type constructors... ";

    float3 v3(1.0f, 2.0f, 3.0f);
    if (v3.x != 1.0f || v3.y != 2.0f || v3.z != 3.0f) {
        std::cout << "FAILED (float3 constructor)\n";
        return false;
    }

    float4 v4(1.0f, 2.0f, 3.0f, 4.0f);
    if (v4.x != 1.0f || v4.y != 2.0f || v4.z != 3.0f || v4.w != 4.0f) {
        std::cout << "FAILED (float4 constructor)\n";
        return false;
    }

    float2 v2(1.0f, 2.0f);
    if (v2.x != 1.0f || v2.y != 2.0f) {
        std::cout << "FAILED (float2 constructor)\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// Test AABB generation
bool test_aabb_generation() {
    std::cout << "Testing AABB generation... ";

    auto aabbs = generate_test_aabbs(100);

    if (aabbs.size() != 100) {
        std::cout << "FAILED (wrong count)\n";
        return false;
    }

    // Check that all AABBs are valid (min < max)
    for (const auto& aabb : aabbs) {
        if (aabb.minX >= aabb.maxX || aabb.minY >= aabb.maxY || aabb.minZ >= aabb.maxZ) {
            std::cout << "FAILED (invalid AABB)\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

#ifndef SPLINETRACER_NO_CUDA
// Test device creation (only with CUDA)
bool test_device_creation() {
    std::cout << "Testing device creation... ";
    try {
        auto device = Device::create(0);
        if (!device || !device->get() || !device->queue()) {
            std::cout << "FAILED (null device/queue)\n";
            return false;
        }
        std::cout << "PASSED\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        return false;
    }
}

// Test acceleration structure building (only with CUDA)
bool test_acceleration_structure() {
    std::cout << "Testing acceleration structure... ";
    try {
        auto device = Device::create(0);
        auto aabbs = generate_test_aabbs(100);

        auto accel = AccelerationStructure::build(*device, aabbs);

        if (!accel || !accel->tlas()) {
            std::cout << "FAILED (null TLAS)\n";
            return false;
        }

        if (accel->primitive_count() != 100) {
            std::cout << "FAILED (wrong primitive count)\n";
            return false;
        }

        std::cout << "PASSED\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        return false;
    }
}

// Performance test (only with CUDA)
bool test_performance() {
    std::cout << "Testing performance... ";
    try {
        auto device = Device::create(0);

        const size_t num_primitives = 10000;
        auto aabbs = generate_test_aabbs(num_primitives);

        auto start = std::chrono::high_resolution_clock::now();
        auto accel = AccelerationStructure::build(*device, aabbs);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "PASSED (" << num_primitives << " primitives in "
                  << duration.count() << "ms)\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        return false;
    }
}
#endif

// Test shader file existence
bool test_shader_files() {
    std::cout << "Testing shader files... ";

    std::vector<std::string> shader_names = {
        "safe_math.slang",
        "sh.slang",
        "volume.slang",
        "ellipsoid.slang",
        "raytracer.slang"
    };

    std::vector<std::string> search_paths = {
        "shaders/",
        "../shaders/",
        "../../shaders/"
    };

    int found = 0;
    std::string found_path;

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path + shader_names[0])) {
            found_path = path;
            for (const auto& name : shader_names) {
                if (std::filesystem::exists(path + name)) {
                    found++;
                }
            }
            break;
        }
    }

    if (found == 0) {
        std::cout << "SKIPPED (shaders not found in search paths)\n";
        return true;
    }

    if (found != (int)shader_names.size()) {
        std::cout << "FAILED (only " << found << "/" << shader_names.size() << " shaders found)\n";
        return false;
    }

    std::cout << "PASSED (" << found << " shaders in " << found_path << ")\n";
    return true;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    std::cout << "========================================\n";
    std::cout << "Modern Splinetracer Test Suite\n";
    std::cout << "========================================\n";
#ifdef SPLINETRACER_NO_CUDA
    std::cout << "Mode: Compile-only (no CUDA)\n";
#else
    std::cout << "Mode: Full (with CUDA)\n";
#endif
    std::cout << "========================================\n\n";

    int passed = 0;
    int total = 0;

    auto run_test = [&](bool (*test_fn)()) {
        total++;
        if (test_fn()) {
            passed++;
        }
    };

    // Basic type tests (always run)
    run_test(test_type_layout);
    run_test(test_type_constructors);
    run_test(test_aabb_generation);
    run_test(test_shader_files);

#ifndef SPLINETRACER_NO_CUDA
    // GPU tests (only with CUDA)
    run_test(test_device_creation);
    run_test(test_acceleration_structure);
    run_test(test_performance);
#else
    std::cout << "\n[GPU tests skipped - CUDA not available]\n";
#endif

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
