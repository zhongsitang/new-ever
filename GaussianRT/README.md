# GaussianRT

Hardware-accelerated ray tracing for Gaussian/ellipsoid volume rendering using slang-rhi.

## Overview

GaussianRT is a modern reimplementation of ellipsoid-based volume rendering using slang-rhi hardware ray tracing. The core ray tracing pipeline has been designed with:

- Clean C++ class hierarchy
- Modern slang shaders with proper module structure
- Hardware ray tracing with procedural primitives
- Volume rendering with dirac delta integration

## Architecture

```
GaussianRT/
├── include/
│   ├── Types.h              # Core data types (float2/3/4, AABB, Camera, etc.)
│   ├── Device.h             # RHI device wrapper
│   ├── AccelerationStructure.h  # BLAS/TLAS builder
│   └── RayTracer.h          # Main ray tracing pipeline
├── src/
│   ├── Device.cpp           # CUDA/RHI device initialization
│   ├── AccelerationStructure.cpp  # Acceleration structure building
│   └── RayTracer.cpp        # Shader loading and ray dispatch
├── shaders/
│   ├── safe_math.slang      # Numerically stable math operations
│   ├── sh.slang             # Spherical harmonics evaluation
│   ├── volume.slang         # Volume rendering state machine
│   ├── ellipsoid.slang      # Ray-ellipsoid intersection
│   └── raytracer.slang      # Main ray tracing shaders
├── test/
│   └── main.cpp             # Test suite
└── CMakeLists.txt           # Build configuration
```

## Key Components

### Device (Device.h/cpp)
- RAII wrapper for slang-rhi CUDA device
- Buffer creation helpers
- Command queue management

### AccelerationStructure (AccelerationStructure.h/cpp)
- Builds BLAS from procedural AABBs
- Creates TLAS with single instance
- Supports compaction and fast build/trace modes

### RayTracer (RayTracer.h/cpp)
- Loads and compiles slang shaders
- Creates ray tracing pipeline with hit groups
- Dispatches rays and collects results

### Shaders
- **raytracer.slang**: Ray generation, intersection, any-hit, and miss shaders
- **volume.slang**: Volume state integration (transmittance, color accumulation)
- **ellipsoid.slang**: Ray-ellipsoid intersection math
- **safe_math.slang**: Numerically stable math utilities
- **sh.slang**: Spherical harmonics evaluation for view-dependent color

## Building

### Requirements
- CMake 3.20+
- CUDA Toolkit (with OptiX support)
- slang compiler and slang-rhi library

### Build Steps
```bash
cd GaussianRT
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Run Tests
```bash
./bin/gaussianrt_test
```

## Core Algorithm

The ray tracing follows this flow:

1. **Ray Generation** (`rayGenShader`):
   - Generate rays from camera or use provided ray origins/directions
   - Initialize volume rendering state

2. **Intersection** (`intersectionShader`):
   - Compute ray-ellipsoid intersection using transformed coordinates
   - Report both entry and exit hit points

3. **Any-Hit** (`anyHitShader`):
   - Buffer multiple hits in sorted order (BUFFER_SIZE = 16)
   - Allow ray to continue finding more intersections

4. **Volume Integration**:
   - Process buffered hits in order
   - Integrate dirac delta contributions
   - Accumulate color and transmittance

5. **Early Termination**:
   - Stop when transmittance drops below threshold (logT > LOG_CUTOFF)
   - Or when max iterations reached

## Namespace

All C++ code is in the `gaussianrt` namespace:

```cpp
#include "Device.h"
#include "AccelerationStructure.h"
#include "RayTracer.h"

using namespace gaussianrt;

auto device = Device::create(0);
auto accel = AccelerationStructure::build(*device, aabbs);
```

## License

Apache License 2.0
