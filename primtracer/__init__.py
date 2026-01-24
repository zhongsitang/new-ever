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

from typing import Any

import torch
from torch.autograd import Function

from . import ellipsoid_tracer as _tracer
from . import backwards_kernel
from . import sh_kernel


# =============================================================================
# PrimTracer - Volume Rendering for Primitives
# =============================================================================

class PrimTracer(Function):
    """Differentiable volume rendering for ellipsoid primitives."""

    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: torch.Tensor,
        max_iters: int,
    ):
        # Ensure contiguous tensors
        mean = mean.contiguous()
        scale = scale.contiguous()
        quat = quat.contiguous()
        density = density.contiguous()
        features = features.contiguous()
        rayo = rayo.contiguous()
        rayd = rayd.contiguous()
        tmax = tmax.contiguous()

        # Call the C++ trace_rays function
        out = _tracer.trace_rays(
            mean, scale, quat, density, features,
            rayo, rayd, tmin, tmax, max_iters
        )

        # Extract outputs
        color_rgba = out["color"]
        depth = out["depth"]
        saved = out["saved"]

        # Store for backward
        ctx.tmin = tmin
        ctx.max_iters = max_iters
        ctx.saved = saved

        ctx.save_for_backward(mean, scale, quat, density, features, rayo, rayd, tmax)

        extras = dict(
            hit_collection=saved.hit_collection,
            iters=saved.iters,
            prim_hits=saved.prim_hits,
        )

        return color_rgba, depth, extras

    @staticmethod
    def backward(ctx, grad_color: torch.Tensor, grad_depth: torch.Tensor, grad_extras: dict = None):
        mean, scale, quat, density, features, rayo, rayd, tmax = ctx.saved_tensors
        saved = ctx.saved

        device = mean.device
        num_prims = mean.shape[0]
        num_rays = rayo.shape[0]

        # Allocate gradient tensors
        dL_dmeans = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dscales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dquats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        dL_dfeatures = torch.zeros_like(features)
        dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        prim_hits = torch.zeros((num_prims), dtype=torch.int32, device=device)
        dL_dinitial_contrib = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        # Combine color and depth gradients
        grad_combined = torch.cat([grad_color, grad_depth.reshape(-1, 1)], dim=1)

        block_size = 16
        if saved.iters.sum() > 0:
            dual_model = (
                mean, scale, quat, density, features,
                dL_dmeans, dL_dscales, dL_dquats, dL_ddensities, dL_dfeatures,
                dL_drayo, dL_drayd,
            )

            backwards_kernel.backwards_kernel(
                (block_size, 1, 1),
                (num_rays // block_size + 1, 1, 1),
                saved.states,
                saved.delta_contribs,
                saved.iters,
                saved.hit_collection,
                rayo,
                rayd,
                dual_model,
                saved.initial_contrib,
                dL_dinitial_contrib,
                prim_hits,
                grad_combined.contiguous(),
                ctx.tmin,
                tmax,
                3.0,  # max_prim_size
                ctx.max_iters,
            )

            if saved.initial_prim_count > 0:
                initial_prim_indices = saved.initial_prim_indices[:saved.initial_prim_count]
                ray_block_size = 64
                second_block_size = 16
                backwards_kernel.backwards_initial_contrib_kernel(
                    (ray_block_size, second_block_size, 1),
                    (
                        num_rays // ray_block_size + 1,
                        saved.initial_prim_count // second_block_size + 1,
                        1,
                    ),
                    rayo,
                    rayd,
                    dual_model,
                    saved.initial_contrib,
                    initial_prim_indices,
                    dL_dinitial_contrib,
                    prim_hits,
                    ctx.tmin,
                )

        # Clip gradients for numerical stability
        clip_val = 1e3
        return (
            dL_dmeans.clip(min=-clip_val, max=clip_val),
            dL_dscales.clip(min=-clip_val, max=clip_val),
            dL_dquats.clip(min=-clip_val, max=clip_val),
            dL_ddensities.clip(min=-50, max=50).reshape(density.shape),
            dL_dfeatures.clip(min=-clip_val, max=clip_val),
            dL_drayo.clip(min=-clip_val, max=clip_val),
            dL_drayd.clip(min=-clip_val, max=clip_val),
            None,  # tmin
            None,  # tmax
            None,  # max_iters
        )


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: torch.Tensor | float = 1000.0,
    max_iters: int = 500,
    return_extras: bool = False,
):
    """
    Trace rays through ellipsoid primitives using differentiable volume rendering.

    Args:
        mean: Primitive centers, shape (N, 3)
        scale: Primitive scales (radii along each axis), shape (N, 3)
        quat: Primitive rotations as unit quaternions (w, x, y, z), shape (N, 4)
        density: Primitive densities, shape (N,) or (N, 1)
        features: SH color features, shape (N, C, 3) where C is the number of
            SH coefficients. For degree-0 SH, C=1.
        rayo: Ray origins, shape (M, 3)
        rayd: Ray directions (should be normalized), shape (M, 3)
        tmin: Minimum t value for ray marching
        tmax: Maximum t value. Can be a scalar or per-ray tensor of shape (M,)
        max_iters: Maximum number of hit iterations per ray
        return_extras: If True, return additional info (hit_collection, iters, prim_hits)

    Returns:
        If return_extras=False (default):
            Tuple of (color, depth)
        If return_extras=True:
            Tuple of (color, depth, extras) where extras is a dict containing:
                - hit_collection: Primitive hit indices for each ray
                - iters: Number of iterations per ray
                - prim_hits: Hit count per primitive
    """
    num_rays = rayo.shape[0]

    # Convert tmax to per-ray tensor if needed
    if isinstance(tmax, (int, float)):
        tmax = torch.full((num_rays,), tmax, dtype=torch.float32, device=rayo.device)
    elif isinstance(tmax, torch.Tensor):
        if tmax.numel() == 1:
            tmax = tmax.expand(num_rays).contiguous()
        elif tmax.shape[0] != num_rays:
            raise ValueError(f"tmax must have shape ({num_rays},) or be a scalar, got {tmax.shape}")
        tmax = tmax.to(dtype=torch.float32, device=rayo.device)

    color, depth, extras = PrimTracer.apply(
        mean, scale, quat, density, features,
        rayo, rayd, tmin, tmax, max_iters
    )

    if return_extras:
        return color, depth, extras
    return color, depth


# =============================================================================
# EvalSH - Spherical Harmonics Evaluation
# =============================================================================

class EvalSH(Function):
    """Evaluate spherical harmonics for view-dependent colors."""

    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor,
        sh_degree: int,
    ):
        block_size = 64
        rayo = rayo.reshape(3).contiguous()
        means = means.contiguous()
        features = features.contiguous()
        color = torch.zeros_like(means)
        ctx.sh_degree = sh_degree
        num_prim = means.shape[0]

        sh_kernel.sh_kernel(
            (block_size, 1, 1),
            (num_prim // block_size + 1, 1, 1),
            means,
            features,
            rayo,
            color,
            sh_degree,
        )

        ctx.save_for_backward(means, features, rayo, color)
        return color

    @staticmethod
    def backward(ctx, dL_dcolor: torch.Tensor):
        block_size = 64
        means, features, rayo, color = ctx.saved_tensors
        num_prim = means.shape[0]
        dL_dfeat = torch.zeros_like(features)
        dL_dcolor = dL_dcolor.contiguous()

        sh_kernel.bw_sh_kernel(
            (block_size, 1, 1),
            ((num_prim + block_size - 1) // block_size, 1, 1),
            means,
            features,
            dL_dfeat,
            rayo,
            dL_dcolor,
            ctx.sh_degree,
        )

        return None, dL_dfeat, None, None


def eval_sh(
    means: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """
    Evaluate spherical harmonics for primitives.

    Args:
        means: Primitive centers, shape (N, 3)
        features: SH coefficients, shape (N, C, 3)
        rayo: Ray origin (camera position), shape (3,) or (1, 3)
        sh_degree: Degree of spherical harmonics

    Returns:
        Evaluated colors, shape (N, 3)
    """
    return EvalSH.apply(means, features, rayo, sh_degree)
