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

import time
from typing import Any

import torch
from torch.autograd import Function

from . import ellipsoid_tracer as tracer
from . import backwards_kernel
from . import sh_kernel

otx = tracer.OptixContext(torch.device("cuda:0"))


# =============================================================================
# PrimTracer - Volume Rendering for Primitives
# =============================================================================

class PrimTracer(Function):
    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        color: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: torch.Tensor,  # Now a tensor (per-ray)
        max_prim_size: float,
        max_iters: int,
    ):
        ctx.device = rayo.device
        ctx.prims = tracer.Primitives(ctx.device)
        assert mean.device == ctx.device
        mean = mean.contiguous()
        scale = scale.contiguous()
        density = density.contiguous()
        quat = quat.contiguous()
        color = color.contiguous()
        tmax = tmax.contiguous()
        ctx.prims.add_primitives(mean, scale, quat, density, color)

        ctx.gas = tracer.GAS(otx, ctx.device, ctx.prims, True, False, True)
        ctx.pipeline = tracer.RayPipeline(otx, ctx.device, ctx.prims, True)
        ctx.max_iters = max_iters
        out = ctx.pipeline.trace_rays(ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, max_prim_size)
        ctx.saved = out["saved"]
        ctx.max_prim_size = max_prim_size
        ctx.tmin = tmin
        ctx.tmax = tmax
        hit_collection = out["hit_collection"]

        # Extract distortion loss from integrator state
        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = distortion_pt1 - distortion_pt2

        # Output format: color RGBA (M, 4), depth (M,)
        color_rgba = out["color"]  # (N, 4): R, G, B, A
        depth = out["depth"]       # (N,): depth

        initial_prim_indices = out['initial_hit_inds'][:out['initial_hit_count'][0]]

        ctx.save_for_backward(
            mean, scale, quat, density, color, rayo, rayd, tmax, hit_collection, out['initial_contrib'], initial_prim_indices
        )

        extras = dict(
            distortion_loss=distortion_loss,
            hit_collection=hit_collection,
            iters=ctx.saved.iters,
            primitive_hit_count=ctx.saved.primitive_hit_count,
            saved=ctx.saved,
        )

        return color_rgba, depth, extras

    @staticmethod
    def backward(ctx, grad_color: torch.Tensor, grad_depth: torch.Tensor, grad_extras: dict = None):
        (
            mean,
            scale,
            quat,
            density,
            features,
            rayo,
            rayd,
            tmax,
            hit_collection,
            initial_contrib,
            initial_prim_indices,
        ) = ctx.saved_tensors
        device = ctx.device

        num_prims = mean.shape[0]
        num_rays = rayo.shape[0]
        dL_dmeans = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dscales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dquats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        dL_dfeatures = torch.zeros_like(features)
        dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        primitive_hit_count = torch.zeros((num_prims), dtype=torch.int32, device=device)
        dL_dinitial_contrib = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        # Handle distortion_loss gradient from extras
        dL_ddistortion = torch.zeros((num_rays, 1), dtype=torch.float32, device=device)
        if grad_extras is not None and grad_extras['distortion_loss'] is not None:
            dL_ddistortion = grad_extras['distortion_loss'].reshape(-1, 1)

        # Combine grad_color (R, G, B, A), grad_depth, and distortion_loss gradient
        grad_combined = torch.cat([grad_color, grad_depth.reshape(-1, 1), dL_ddistortion], dim=1)

        block_size = 16
        if ctx.saved.iters.sum() > 0:
            dual_model = (
                mean,
                scale,
                quat,
                density,
                features,
                dL_dmeans,
                dL_dscales,
                dL_dquats,
                dL_ddensities,
                dL_dfeatures,
                dL_drayo,
                dL_drayd,
            )

            backwards_kernel.backwards_kernel(
                (block_size, 1, 1),
                (num_rays // block_size + 1, 1, 1),
                ctx.saved.states,
                ctx.saved.delta_contribs,
                ctx.saved.iters,
                hit_collection,
                rayo,
                rayd,
                dual_model,
                initial_contrib,
                dL_dinitial_contrib,
                primitive_hit_count,
                grad_combined.contiguous(),
                ctx.tmin,
                tmax,
                ctx.max_prim_size,
                ctx.max_iters,
            )

            if initial_prim_indices.shape[0] > 0:
                ray_block_size = 64
                second_block_size = 16
                backwards_kernel.backwards_initial_contrib_kernel(
                    (ray_block_size, second_block_size, 1),
                    (
                        rayo.shape[0] // ray_block_size + 1,
                        initial_prim_indices.shape[0] // second_block_size + 1,
                        1,
                    ),
                    rayo,
                    rayd,
                    dual_model,
                    initial_contrib,
                    initial_prim_indices,
                    dL_dinitial_contrib,
                    primitive_hit_count,
                    ctx.tmin,
                )

        v = 1e3
        mean_v = 1e3
        return (
            dL_dmeans.clip(min=-mean_v, max=mean_v),
            dL_dscales.clip(min=-v, max=v),
            dL_dquats.clip(min=-v, max=v),
            dL_ddensities.clip(min=-50, max=50).reshape(density.shape),
            dL_dfeatures.clip(min=-v, max=v),
            dL_drayo.clip(min=-v, max=v),
            dL_drayd.clip(min=-v, max=v),
            None,  # tmin
            None,  # tmax
            None,  # max_prim_size
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
    tmax: torch.Tensor | float = 1000,
    max_prim_size: float = 3,
    max_iters: int = 500,
    return_extras: bool = False,
):
    """
    Trace rays through ellipsoid primitives using volume rendering integration.

    Parameters
    ----------
    mean : torch.Tensor
        Primitive centers, shape (N, 3).
    scale : torch.Tensor
        Primitive scales (radii along each axis), shape (N, 3).
    quat : torch.Tensor
        Primitive rotations as unit quaternions (w, x, y, z), shape (N, 4).
    density : torch.Tensor
        Primitive densities, shape (N,) or (N, 1).
    features : torch.Tensor
        Primitive SH features/colors, shape (N, C, 3) where C is the number of
        SH coefficients. For degree-0 SH, C=1.
    rayo : torch.Tensor
        Ray origins, shape (M, 3).
    rayd : torch.Tensor
        Ray directions (should be normalized), shape (M, 3).
    tmin : float
        Minimum t value for ray marching (ray starts at origin + tmin * direction).
    tmax : torch.Tensor or float
        Maximum t value for ray marching. Can be a scalar (broadcast to all rays)
        or a tensor of shape (M,) for per-ray tmax values.
    max_prim_size : float
        Maximum primitive size for BVH acceleration structure.
    max_iters : int
        Maximum number of hit iterations per ray.
    return_extras : bool
        If True, return additional debugging information.

    Returns
    -------
    If return_extras=False (default):
        tuple of (color, depth)
            color : torch.Tensor
                RGBA color output, shape (M, 4). Channels are [R, G, B, A] where
                RGB are the accumulated colors and A is the accumulated opacity
                (alpha = 1 - transmittance).
            depth : torch.Tensor
                Expected depth along each ray, shape (M,). Computed as the
                transmittance-weighted mean of hit distances.

    If return_extras=True:
        tuple of (color, depth, extras)
            color : torch.Tensor
                Same as above.
            depth : torch.Tensor
                Same as above.
            extras : dict
                Dictionary containing:
                - 'distortion_loss': torch.Tensor of shape (M,), per-ray distortion
                  loss for regularization.
                - 'hit_collection': torch.Tensor, primitive hit IDs for backward pass.
                - 'iters': torch.Tensor, number of iterations per ray.
                - 'primitive_hit_count': torch.Tensor, hit count per primitive.
                - 'saved': internal state object for backward pass.
    """
    num_rays = rayo.shape[0]

    # Handle tmax: convert scalar to tensor and broadcast if needed
    if isinstance(tmax, (int, float)):
        tmax = torch.full((num_rays,), tmax, dtype=torch.float32, device=rayo.device)
    elif isinstance(tmax, torch.Tensor):
        if tmax.numel() == 1:
            # Scalar tensor, broadcast to all rays
            tmax = tmax.expand(num_rays).contiguous()
        elif tmax.shape[0] != num_rays:
            raise ValueError(f"tmax must have shape ({num_rays},) or be a scalar, got {tmax.shape}")
        tmax = tmax.to(dtype=torch.float32, device=rayo.device)

    color_rgba, depth, extras = PrimTracer.apply(
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
        max_iters,
    )
    if return_extras:
        return color_rgba, depth, extras
    else:
        return color_rgba, depth


# =============================================================================
# EvalSH - Spherical Harmonics Evaluation
# =============================================================================

class EvalSH(Function):
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
):
    """
    Evaluate spherical harmonics for primitives.

    Parameters
    ----------
    means : torch.Tensor
        Primitive centers, shape (N, 3)
    features : torch.Tensor
        SH coefficients, shape (N, C, 3)
    rayo : torch.Tensor
        Ray origin (camera position), shape (3,) or (1, 3)
    sh_degree : int
        Degree of spherical harmonics

    Returns
    -------
    torch.Tensor
        Evaluated colors, shape (N, 3)
    """
    return EvalSH.apply(means, features, rayo, sh_degree)
