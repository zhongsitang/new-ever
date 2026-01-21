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
Simplified spline tracer interface without slangtorch dependency.
Autograd is fully implemented in C++.
"""

from typing import Optional
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from build.lib import ellipsoid_splinetracer as sp

# Global OptiX context (lazily initialized)
_optix_context = None


def _get_optix_context(device: torch.device):
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
    wcts: Optional[torch.Tensor] = None,
    max_iters: int = 500,
) -> torch.Tensor:
    """
    Trace rays through ellipsoid primitives with full autograd support.

    Args:
        mean: Primitive centers (N, 3)
        scale: Primitive scales (N, 3)
        quat: Primitive quaternions (N, 4)
        density: Primitive densities (N,)
        features: Primitive features/colors (N, D, 3) where D is SH degree squared
        rayo: Ray origins (R, 3)
        rayd: Ray directions (R, 3)
        tmin: Minimum ray parameter
        tmax: Maximum ray parameter
        max_prim_size: Maximum primitive size for acceleration
        wcts: Optional world-to-clip transforms (R, 4, 4) or (1, 4, 4)
        max_iters: Maximum iterations per ray

    Returns:
        Tensor of shape (R, 5) containing [R, G, B, opacity, distortion_loss]
    """
    device = rayo.device

    # Ensure contiguous tensors
    mean = mean.contiguous()
    scale = scale.contiguous()
    quat = quat.contiguous()
    density = density.contiguous()
    features = features.contiguous()
    rayo = rayo.contiguous()
    rayd = rayd.contiguous()

    # Handle optional wcts
    if wcts is None:
        wcts = torch.empty(0, dtype=torch.float32, device=device)
    else:
        wcts = wcts.contiguous()

    # Get or create OptiX context
    otx = _get_optix_context(device)

    # Create primitives and GAS
    prims = sp.Primitives(device)
    prims.add_primitives(mean, scale, quat, density, features)
    gas = sp.GAS(otx, device, prims, True, False, True)

    # Call the autograd-enabled trace_rays
    result = sp.trace_rays(
        mean, scale, quat, density, features,
        rayo, rayd, wcts,
        tmin, tmax, max_prim_size, max_iters,
        otx.context_ptr(), gas.handle()
    )

    return result[0]  # Return the color tensor
