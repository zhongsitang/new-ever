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

from pathlib import Path
from typing import *

import torch

import sys
sys.path.append(str(Path(__file__).parent))

from build.lib import ellipsoid_splinetracer as sp


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000,
    max_prim_size: float = 3,
    wcts: Optional[torch.Tensor] = None,
    max_iters: int = 500,
) -> torch.Tensor:
    """
    Trace rays through ellipsoid spline primitives.

    Args:
        mean: Primitive centers (N, 3)
        scale: Primitive scales (N, 3)
        quat: Primitive rotations as quaternions (N, 4)
        density: Primitive densities (N,)
        features: Spherical harmonics features (N, d, 3)
        rayo: Ray origins (R, 3)
        rayd: Ray directions (R, 3)
        tmin: Minimum ray distance
        tmax: Maximum ray distance
        max_prim_size: Maximum primitive size for acceleration
        wcts: World-to-camera transforms for 2D gradients (optional)
        max_iters: Maximum ray marching iterations

    Returns:
        output: Tensor of shape (R, 5) containing [RGBA (4), distortion_loss (1)]

    Note:
        Gradients are computed automatically via PyTorch autograd.
    """
    wcts_tensor = wcts if wcts is not None else torch.empty(0, device=rayo.device)

    return sp.trace_rays(
        mean, scale, quat, density, features,
        rayo, rayd,
        tmin, tmax, max_prim_size,
        wcts_tensor, max_iters
    )
