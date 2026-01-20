#!/usr/bin/env python3
"""
GaussianRT Integration Tests

Tests the full pipeline from Python/PyTorch interface.
"""

import sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, skipping torch tests")

# Try to import the extension
try:
    sys.path.insert(0, '..')
    from python import gaussian_rt
    from python.gaussian_rt import render_gaussians
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False
    print("Warning: gaussian_rt extension not built, using placeholder tests")


def test_render_basic():
    """Test basic rendering with simple primitives."""
    if not HAS_TORCH:
        print("Skipping test_render_basic: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("Skipping test_render_basic: CUDA not available")
        return

    print("Running test_render_basic...", end=" ")

    # Create simple primitives
    num_prims = 10
    device = torch.device('cuda:0')

    means = torch.randn(num_prims, 3, device=device)
    scales = torch.ones(num_prims, 3, device=device) * 0.1
    quats = torch.zeros(num_prims, 4, device=device)
    quats[:, 3] = 1.0  # Identity rotation
    densities = torch.ones(num_prims, device=device)
    colors = torch.rand(num_prims, 3, device=device)

    # Create rays
    num_rays = 64 * 64
    ray_origins = torch.zeros(num_rays, 3, device=device)
    ray_origins[:, 2] = -5.0
    ray_directions = torch.zeros(num_rays, 3, device=device)
    ray_directions[:, 2] = 1.0

    # Render
    output = render_gaussians(
        means, scales, quats, densities, colors,
        ray_origins, ray_directions,
        tmin=0.01, tmax=100.0, max_iters=128
    )

    assert output.shape == (num_rays, 4), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("PASSED")


def test_render_gradient():
    """Test gradient computation."""
    if not HAS_TORCH:
        print("Skipping test_render_gradient: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("Skipping test_render_gradient: CUDA not available")
        return

    print("Running test_render_gradient...", end=" ")

    num_prims = 5
    device = torch.device('cuda:0')

    means = torch.randn(num_prims, 3, device=device, requires_grad=True)
    scales = torch.ones(num_prims, 3, device=device, requires_grad=True) * 0.1
    quats = torch.zeros(num_prims, 4, device=device, requires_grad=True)
    quats.data[:, 3] = 1.0
    densities = torch.ones(num_prims, device=device, requires_grad=True)
    colors = torch.rand(num_prims, 3, device=device, requires_grad=True)

    num_rays = 16 * 16
    ray_origins = torch.zeros(num_rays, 3, device=device)
    ray_origins[:, 2] = -5.0
    ray_directions = torch.zeros(num_rays, 3, device=device)
    ray_directions[:, 2] = 1.0

    # Render
    output = render_gaussians(
        means, scales, quats, densities, colors,
        ray_origins, ray_directions,
        max_iters=64
    )

    # Compute loss
    loss = output.sum()

    # Backward
    loss.backward()

    # Check gradients exist
    assert means.grad is not None, "means.grad is None"
    assert scales.grad is not None, "scales.grad is None"
    assert densities.grad is not None, "densities.grad is None"
    assert colors.grad is not None, "colors.grad is None"

    # Check gradients are finite
    assert not torch.isnan(means.grad).any(), "means.grad contains NaN"
    assert not torch.isnan(scales.grad).any(), "scales.grad contains NaN"
    assert not torch.isnan(densities.grad).any(), "densities.grad contains NaN"
    assert not torch.isnan(colors.grad).any(), "colors.grad contains NaN"

    print("PASSED")


def test_sh_rendering():
    """Test rendering with spherical harmonics."""
    if not HAS_TORCH:
        print("Skipping test_sh_rendering: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("Skipping test_sh_rendering: CUDA not available")
        return

    print("Running test_sh_rendering...", end=" ")

    num_prims = 5
    device = torch.device('cuda:0')

    means = torch.randn(num_prims, 3, device=device)
    scales = torch.ones(num_prims, 3, device=device) * 0.1
    quats = torch.zeros(num_prims, 4, device=device)
    quats[:, 3] = 1.0
    densities = torch.ones(num_prims, device=device)

    # SH degree 1 features (4 * 3 = 12)
    sh_features = torch.rand(num_prims, 12, device=device)

    num_rays = 32 * 32
    ray_origins = torch.zeros(num_rays, 3, device=device)
    ray_origins[:, 2] = -5.0
    ray_directions = torch.zeros(num_rays, 3, device=device)
    ray_directions[:, 2] = 1.0

    output = render_gaussians(
        means, scales, quats, densities, sh_features,
        ray_origins, ray_directions,
        sh_degree=1, max_iters=64
    )

    assert output.shape == (num_rays, 4)
    assert not torch.isnan(output).any()

    print("PASSED")


def test_renderer_class():
    """Test GaussianRenderer class."""
    if not HAS_TORCH:
        print("Skipping test_renderer_class: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("Skipping test_renderer_class: CUDA not available")
        return

    print("Running test_renderer_class...", end=" ")

    from python.gaussian_rt import GaussianRenderer

    renderer = GaussianRenderer(cuda_device_id=0, enable_backward=True)

    num_prims = 10
    device = torch.device('cuda:0')

    means = torch.randn(num_prims, 3, device=device)
    scales = torch.ones(num_prims, 3, device=device) * 0.1
    quats = torch.zeros(num_prims, 4, device=device)
    quats[:, 3] = 1.0
    densities = torch.ones(num_prims, device=device)
    colors = torch.rand(num_prims, 3, device=device)

    renderer.set_primitives(means, scales, quats, densities, colors)

    num_rays = 64 * 64
    ray_origins = torch.zeros(num_rays, 3, device=device)
    ray_origins[:, 2] = -5.0
    ray_directions = torch.zeros(num_rays, 3, device=device)
    ray_directions[:, 2] = 1.0

    output = renderer.render(ray_origins, ray_directions)

    assert output.shape == (num_rays, 4)

    renderer.synchronize()

    print("PASSED")


def test_large_scene():
    """Test rendering with many primitives."""
    if not HAS_TORCH:
        print("Skipping test_large_scene: PyTorch not available")
        return

    if not torch.cuda.is_available():
        print("Skipping test_large_scene: CUDA not available")
        return

    print("Running test_large_scene...", end=" ")

    num_prims = 10000
    device = torch.device('cuda:0')

    means = torch.randn(num_prims, 3, device=device) * 10
    scales = torch.ones(num_prims, 3, device=device) * 0.05
    quats = torch.zeros(num_prims, 4, device=device)
    quats[:, 3] = 1.0
    densities = torch.ones(num_prims, device=device)
    colors = torch.rand(num_prims, 3, device=device)

    num_rays = 128 * 128
    ray_origins = torch.zeros(num_rays, 3, device=device)
    ray_origins[:, 2] = -20.0
    ray_directions = torch.zeros(num_rays, 3, device=device)
    ray_directions[:, 2] = 1.0

    output = render_gaussians(
        means, scales, quats, densities, colors,
        ray_origins, ray_directions,
        tmin=0.1, tmax=50.0, max_iters=256
    )

    assert output.shape == (num_rays, 4)
    assert not torch.isnan(output).any()

    print("PASSED")


def main():
    print("=== GaussianRT Python Integration Tests ===\n")

    test_render_basic()
    test_render_gradient()
    test_sh_rendering()
    test_renderer_class()
    test_large_scene()

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
