# Modern Splinetracer

A modern, clean reimplementation of ellipsoid-based volume rendering using slang-rhi hardware ray tracing.

## Overview

This project is a complete rewrite of the legacy splinetracer using modern slang-rhi patterns. The core ray tracing pipeline (`Forward::trace_rays`) has been reimplemented with:

- Clean C++ class hierarchy
- Modern slang shaders with proper module structure
- Hardware ray tracing with procedural primitives
- Volume rendering with dirac delta integration

## Architecture

```
modern-splinetracer/
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
cd modern-splinetracer
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Integration with Parent Project
When building as part of the main repository:
```bash
cd /path/to/new-ever
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
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

## Differences from Legacy Code

| Aspect | Legacy | Modern |
|--------|--------|--------|
| API | OptiX 7.x direct | slang-rhi abstraction |
| Shaders | CUDA + OptiX | Slang |
| Buffer Management | Manual CUDA | RHI wrappers |
| Pipeline | OptiX programs | Slang shader tables |
| Portability | NVIDIA only | Cross-platform (via RHI) |

## License

Apache License 2.0
