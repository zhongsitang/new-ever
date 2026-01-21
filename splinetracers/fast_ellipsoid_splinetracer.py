# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fast Ellipsoid Spline Tracer - No slangtorch dependency

This module provides differentiable ray tracing for ellipsoid primitives
using OptiX for acceleration and native CUDA for backward pass.
"""

from pathlib import Path
from typing import Optional

import torch

import sys
sys.path.append(str(Path(__file__).parent))

from build.lib import ellipsoid_splinetracer as sp

# Global OptiX context (created on first use)
_optix_context = None


def _get_optix_context(device: torch.device) -> sp.OptixContext:
    """Get or create the global OptiX context."""
    global _optix_context
    if _optix_context is None:
        _optix_context = sp.OptixContext(device)
    return _optix_context


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000.0,
    max_prim_size: float = 3.0,
    dL_dmeans2D: Optional[torch.Tensor] = None,
    wcts: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    return_extras: bool = False,
) -> torch.Tensor:
    """
    Trace rays through ellipsoid primitives with automatic differentiation.

    This is the simplified interface that replaces the old SplineTracer class.
    Gradients are computed automatically via PyTorch autograd.

    Args:
        mean: Primitive centers [N, 3]
        scale: Primitive scales [N, 3]
        quat: Primitive rotations as quaternions [N, 4]
        density: Primitive densities [N]
        features: Spherical harmonic features [N, feature_size, 3]
        rayo: Ray origins [num_rays, 3]
        rayd: Ray directions [num_rays, 3]
        tmin: Minimum ray parameter
        tmax: Maximum ray parameter
        max_prim_size: Maximum primitive size for traversal
        dL_dmeans2D: (Unused, kept for API compatibility)
        wcts: World-to-camera transforms [num_rays, 4, 4] or [1, 4, 4]
        max_iters: Maximum iterations per ray
        return_extras: If True, return additional debug info (not implemented)

    Returns:
        color_and_loss: [num_rays, 5] - RGB color (3), depth (1), distortion loss (1)
    """
    device = rayo.device

    # Ensure inputs are contiguous
    mean = mean.contiguous()
    scale = scale.contiguous()
    quat = quat.contiguous()
    density = density.contiguous()
    features = features.contiguous()
    rayo = rayo.contiguous()
    rayd = rayd.contiguous()

    # Handle wcts
    if wcts is None:
        wcts = torch.ones((1, 4, 4), device=device, dtype=torch.float32)
    else:
        wcts = wcts.contiguous()

    # Get OptiX context
    ctx = _get_optix_context(device)

    # Call the C++ implementation with autograd support
    color_and_loss = sp.trace_rays(
        ctx,
        mean,
        scale,
        quat,
        density,
        features,
        rayo,
        rayd,
        tmin,
        tmax,
        max_prim_size,
        wcts,
        max_iters,
    )

    if return_extras:
        # For now, return_extras returns minimal info since full extras require
        # additional forward pass data that's not exposed
        return color_and_loss, {}
    else:
        return color_and_loss
