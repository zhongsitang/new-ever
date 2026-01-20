# GaussianRT

A modern differentiable volume renderer for ellipsoid primitives, implemented using Slang-RHI for ray tracing and Slang automatic differentiation for gradient computation.

## Features

- **Hardware-accelerated ray tracing** using Slang-RHI (CUDA backend)
- **Differentiable volume rendering** with automatic gradient computation
- **Ellipsoid volume elements** with position, scale, rotation, opacity, and spherical harmonics color
- **PyTorch integration** via autograd function for end-to-end training
- **Efficient backward pass** using Slang's reverse-mode autodiff

## Architecture

```
GaussianRT/
├── src/                    # C++ core implementation
│   ├── types.h             # Core data structures
│   ├── device.{h,cpp}      # Slang-RHI device management
│   ├── acceleration_structure.{h,cpp}  # BVH for ray tracing
│   ├── renderer.{h,cpp}    # Forward ray tracing pipeline
│   ├── volume_renderer.{h,cpp}  # High-level renderer
│   └── cuda/               # CUDA kernels
│       ├── aabb_builder.cu
│       └── initialize_state.cu
├── slang/                  # Slang shaders
│   ├── common.slang        # Shared types and utilities
│   ├── forward.slang       # Forward ray tracing shaders
│   └── backward.slang      # Backward gradient computation
├── python/                 # Python bindings
│   ├── __init__.py
│   ├── renderer.py         # PyTorch autograd function
│   └── bindings.cpp        # pybind11 bindings
└── examples/               # Example scripts
```

## Volume Rendering Algorithm

The renderer implements the volume rendering integral:

```
C = ∫ T(t) σ(t) c(t) dt
```

where:
- `T(t) = exp(-∫₀ᵗ σ(s) ds)` is transmittance
- `σ(t)` is density/opacity
- `c(t)` is color (from spherical harmonics evaluation)

For ellipsoid primitives, ray-ellipsoid intersections are computed analytically, and samples are taken at intersection points.

## Requirements

- CUDA Toolkit (11.0+)
- PyTorch (1.10+)
- Slang SDK
- pybind11
- CMake (3.20+)

## Installation

1. Set the Slang SDK path:
```bash
export SLANG_ROOT=/path/to/slang-sdk
```

2. Install the package:
```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

## Usage

### Basic Rendering

```python
import torch
from gaussian_rt import VolumeRendererModule

# Create renderer
renderer = VolumeRendererModule(
    device_index=0,
    t_min=0.1,
    t_max=100.0,
    max_samples=128,
    sh_degree=3,
)

# Create ellipsoid elements
N = 1000
positions = torch.randn(N, 3, device='cuda')       # [N, 3]
scales = torch.abs(torch.randn(N, 3, device='cuda') * 0.1)  # [N, 3]
rotations = torch.randn(N, 4, device='cuda')       # [N, 4] quaternions
rotations = rotations / rotations.norm(dim=1, keepdim=True)
opacities = torch.sigmoid(torch.randn(N, device='cuda'))    # [N]
features = torch.randn(N, 48, device='cuda')       # [N, 16*3] SH coefficients

# Enable gradients
positions.requires_grad_(True)

# Generate rays
ray_origins = torch.zeros(H*W, 3, device='cuda')
ray_directions = ...  # Normalized direction vectors

# Render
colors, depths, distortions = renderer(
    positions, scales, rotations, opacities, features,
    ray_origins, ray_directions
)

# Compute loss and gradients
loss = (colors[:, :3] - target).pow(2).mean()
loss.backward()

# positions.grad now contains gradients
```

### Rendering an Image

```python
camera = {
    'position': [0, 0, 3],
    'forward': [0, 0, -1],
    'up': [0, 1, 0],
    'right': [1, 0, 0],
    'focal_length': 1.0,
}

colors, depths, distortions = renderer.render_image(
    positions, scales, rotations, opacities, features,
    camera, width=512, height=512
)
```

## Data Structures

### Volume Element (Ellipsoid)

Each ellipsoid is defined by:
- **position** `[3]`: Center in world space
- **scale** `[3]`: Semi-axes lengths (a, b, c)
- **rotation** `[4]`: Unit quaternion (w, x, y, z)
- **opacity** `[1]`: Density coefficient σ
- **features** `[48]`: Spherical harmonics coefficients (16 RGB triplets)

### Render State

Accumulated along each ray:
- `log_transmittance`: log(T) for numerical stability
- `accumulated_color`: Weighted color sum
- `accumulated_depth`: Weighted depth sum
- `distortion_accum`: For distortion loss computation

## Extending to Other Primitives

The code is designed to support multiple primitive types. To add a new primitive:

1. Add new intersection code in `forward.slang`
2. Add AABB computation in `cuda/aabb_builder.cu`
3. Add backward differentiation in `backward.slang`

## Acknowledgments

This project is inspired by:
- 3D Gaussian Splatting
- Slang shader language and RHI
- NeRF and differentiable rendering research

## License

MIT License
