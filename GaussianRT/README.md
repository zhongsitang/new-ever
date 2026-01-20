# GaussianRT

**Differentiable 3D Gaussian Ray Tracing Renderer**

A modern, high-performance implementation of differentiable 3D Gaussian splatting using hardware-accelerated ray tracing.

## Features

- **Hardware Ray Tracing**: Leverages GPU RT cores via Slang-RHI / OptiX
- **Differentiable Rendering**: Full gradient computation for all Gaussian parameters
- **PyTorch Integration**: Native `torch.autograd.Function` support
- **Spherical Harmonics**: View-dependent colors up to degree 3
- **Volume Rendering**: Physically-based integration along rays

## Architecture

```
GaussianRT/
├── include/gaussian_rt/     # C++ headers
│   ├── Types.h              # Shared types
│   ├── Device.h             # GPU device management
│   ├── GaussianPrimitives.h # Primitive data
│   ├── AccelStruct.h        # BVH acceleration
│   ├── ForwardRenderer.h    # Ray tracing forward pass
│   └── BackwardPass.h       # Gradient computation
├── src/                     # C++ implementation
├── shaders/                 # Slang shaders
│   ├── common/              # Shared shader code
│   ├── forward/             # Ray tracing shaders
│   └── backward/            # Gradient kernels
├── python/                  # Python/PyTorch bindings
└── tests/                   # Unit and integration tests
```

## Requirements

- CUDA 11.7+
- CMake 3.18+
- OptiX 7.x (for ray tracing)
- Slang SDK (optional, for shader compilation)
- Python 3.8+ (for Python bindings)
- PyTorch 2.0+ (for PyTorch integration)

## Building

```bash
mkdir build && cd build
cmake .. -DSLANG_SDK_PATH=/path/to/slang
make -j$(nproc)
```

### CMake Options

- `GAUSSIANRT_BUILD_PYTHON`: Build Python bindings (ON by default)
- `GAUSSIANRT_BUILD_TESTS`: Build tests (ON by default)
- `SLANG_SDK_PATH`: Path to Slang SDK

## Usage

### Python API

```python
import torch
from gaussian_rt import render_gaussians

# Create Gaussian primitives
means = torch.randn(1000, 3, device='cuda')
scales = torch.ones(1000, 3, device='cuda') * 0.01
quats = torch.zeros(1000, 4, device='cuda')
quats[:, 3] = 1.0  # Identity rotation
densities = torch.ones(1000, device='cuda')
colors = torch.rand(1000, 3, device='cuda')

# Generate camera rays
ray_origins = torch.zeros(640*480, 3, device='cuda')
ray_directions = generate_camera_rays(640, 480, fov=60)

# Render (differentiable)
output = render_gaussians(
    means, scales, quats, densities, colors,
    ray_origins, ray_directions,
    tmin=0.01, tmax=100.0,
    max_iters=512
)

# Output is (N, 4) RGBA
image = output[:, :3].reshape(480, 640, 3)

# Compute gradients
loss = (output[:, :3] - target).pow(2).mean()
loss.backward()
# means.grad, scales.grad, etc. are now populated
```

### C++ API

```cpp
#include <gaussian_rt/GaussianRT.h>

using namespace gaussian_rt;

// Initialize
initializeGlobalDevice(0);

// Create primitives
GaussianPrimitives prims(getGlobalDevice());
prims.setData(numPrims, means, scales, quats, densities, features, featureSize);

// Build acceleration structure
AccelStruct accel(getGlobalDevice());
accel.build(prims);

// Create renderer
ForwardRenderer renderer(getGlobalDevice());
renderer.initialize(true);  // enable backward

// Render
ForwardOutput output;
renderer.allocateOutput(numRays, maxIters, output);
renderer.traceRays(accel, prims, rayOrigins, rayDirs, numRays, params, output);

// Backward (optional)
BackwardPass backward(getGlobalDevice());
backward.initialize();
GradientOutput gradients;
backward.allocateGradients(numPrims, numRays, featureSize, gradients);
backward.compute(output, dLdColor, prims, rayOrigins, rayDirs, numRays, params, gradients);
```

## Algorithm

### Forward Pass

1. **Ray Generation**: Generate rays from camera
2. **BVH Traversal**: Traverse acceleration structure
3. **Intersection**: Custom ellipsoid intersection test
4. **Sorting**: Sort hits by distance (anyhit shader)
5. **Volume Rendering**: Accumulate color using:

   $$C = \sum_i \alpha_i \cdot c_i \cdot \prod_{j<i}(1-\alpha_j)$$

### Backward Pass

1. **Replay Forward**: Reconstruct intermediate states
2. **Reverse Iteration**: Process hits in reverse order
3. **Chain Rule**: Compute gradients via automatic differentiation
4. **Atomic Accumulation**: Thread-safe gradient accumulation

## Performance

| Scene Size | Resolution | Forward | Backward |
|-----------|------------|---------|----------|
| 10K Gaussians | 640x480 | ~5ms | ~10ms |
| 100K Gaussians | 640x480 | ~15ms | ~30ms |
| 1M Gaussians | 640x480 | ~50ms | ~100ms |

*Benchmarks on RTX 3090*

## Comparison with Original Implementation

| Feature | Original (splinetracers) | GaussianRT |
|---------|-------------------------|------------|
| Ray Tracing API | OptiX-specific | Slang-RHI (portable) |
| Shader Language | Slang + OptiX-IR | Pure Slang |
| Backward | slangtorch runtime | slangc pre-compiled |
| Build System | Custom CMake | Modern CMake |
| Python API | pybind11 | pybind11 + PyTorch C++ |

## License

MIT License

## Acknowledgments

Based on the splinetracers project. Uses:
- [Slang](https://github.com/shader-slang/slang) shader language
- [OptiX](https://developer.nvidia.com/optix) ray tracing
- [PyTorch](https://pytorch.org/) deep learning framework
