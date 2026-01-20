"""
GaussianRT - Differentiable Gaussian Ray Tracing

A modern implementation of differentiable 3D Gaussian splatting
using hardware-accelerated ray tracing with Slang-RHI.

Features:
- Hardware ray tracing via Slang-RHI
- Differentiable rendering with automatic gradient computation
- PyTorch integration for deep learning workflows
- Support for spherical harmonics view-dependent colors

Example:
    import torch
    from gaussian_rt import render_gaussians

    # Create Gaussian primitives
    means = torch.randn(1000, 3, device='cuda')
    scales = torch.ones(1000, 3, device='cuda') * 0.01
    quats = torch.zeros(1000, 4, device='cuda')
    quats[:, 3] = 1.0  # Identity rotation
    densities = torch.ones(1000, device='cuda')
    colors = torch.rand(1000, 3, device='cuda')

    # Generate rays
    ray_origins = torch.zeros(640*480, 3, device='cuda')
    ray_directions = generate_ray_directions(640, 480)

    # Render
    output = render_gaussians(
        means, scales, quats, densities, colors,
        ray_origins, ray_directions
    )

    # Compute gradients
    loss = output.sum()
    loss.backward()
"""

from .gaussian_rt import (
    GaussianRayTrace,
    render_gaussians,
    GaussianRenderer,
)

try:
    from .gaussian_rt_ext import (
        Device,
        Primitives,
        AccelStruct,
        ForwardRenderer,
        BackwardPass,
        get_version,
    )
    __version__ = get_version()
except ImportError:
    __version__ = "1.0.0"
    print("Warning: gaussian_rt_ext not found. Run cmake to build the extension.")

__all__ = [
    'GaussianRayTrace',
    'render_gaussians',
    'GaussianRenderer',
    'Device',
    'Primitives',
    'AccelStruct',
    'ForwardRenderer',
    'BackwardPass',
]
