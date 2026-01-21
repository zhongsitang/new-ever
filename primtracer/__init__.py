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
PrimTracer - Primitive-based Volume Rendering Library

This module provides GPU-accelerated volume rendering for ellipsoid primitives
using OptiX ray tracing. It supports differentiable rendering with automatic
gradient computation for optimization-based reconstruction.
"""

from typing import Any, Optional

import torch
from torch.autograd import Function

from . import primtracer_core as core
from . import backwards_kernel
from . import sh_kernel

# Global OptiX context (initialized once per process)
_optix_context = core.OptixContext(torch.device("cuda:0"))


# =============================================================================
# VolumeRenderer - Main differentiable volume rendering function
# =============================================================================

class VolumeRenderer(Function):
    """
    Differentiable volume renderer for ellipsoid primitives.

    This PyTorch autograd function performs ray tracing through a collection
    of ellipsoid primitives with Gaussian density profiles, computing both
    the rendered color and gradients for optimization.
    """

    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        sh_coeffs: torch.Tensor,
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        t_near: float,
        t_far: float,
        max_prim_size: float,
        mean2d: torch.Tensor,
        world_to_clip: torch.Tensor,
        max_samples: int,
        return_extras: bool = False,
    ):
        ctx.device = ray_origin.device
        ctx.primitives = core.Primitives(ctx.device)

        assert mean.device == ctx.device
        mean = mean.contiguous()
        scale = scale.contiguous()
        density = density.contiguous()
        quat = quat.contiguous()
        sh_coeffs = sh_coeffs.contiguous()

        ctx.primitives.add_primitives(mean, scale, quat, density, sh_coeffs)

        ctx.accel = core.GAS(_optix_context, ctx.device, ctx.primitives, True, False, True)
        ctx.tracer = core.Forward(_optix_context, ctx.device, ctx.primitives, True)
        ctx.max_samples = max_samples

        out = ctx.tracer.trace_rays(
            ctx.accel, ray_origin, ray_direction, t_near, t_far, ctx.max_samples, max_prim_size
        )

        ctx.saved_buffer = out["saved"]
        ctx.max_prim_size = max_prim_size
        ctx.t_near = t_near
        ctx.t_far = t_far
        prim_sequence = out["tri_collection"]

        # Extract distortion loss from integration state
        states = ctx.saved_buffer.states.reshape(ray_origin.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = distortion_pt1 - distortion_pt2
        color_and_loss = torch.cat([out["color"], distortion_loss.reshape(-1, 1)], dim=1)

        initial_inds = out['initial_touch_inds'][:out['initial_touch_count'][0]]

        ctx.save_for_backward(
            mean, scale, quat, density, sh_coeffs, ray_origin, ray_direction,
            prim_sequence, world_to_clip, out['initial_drgb'], initial_inds
        )

        if return_extras:
            return color_and_loss, dict(
                tri_collection=prim_sequence,
                iters=ctx.saved_buffer.iters,
                opacity=out["color"][:, 3],
                touch_count=ctx.saved_buffer.touch_count,
                distortion_loss=distortion_loss,
                saved=ctx.saved_buffer,
            )
        else:
            return color_and_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, return_extras=False):
        (
            mean,
            scale,
            quat,
            density,
            sh_coeffs,
            ray_origin,
            ray_direction,
            prim_sequence,
            world_to_clip,
            initial_sample,
            initial_inds,
        ) = ctx.saved_tensors
        device = ctx.device

        num_prims = mean.shape[0]
        num_rays = ray_origin.shape[0]

        # Initialize gradient tensors
        grad_means = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        grad_scales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        grad_quats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
        grad_densities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)
        grad_ray_origin = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        grad_ray_direction = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        grad_means2d = torch.zeros((num_prims, 2), dtype=torch.float32, device=device)
        hit_count = torch.zeros((num_prims), dtype=torch.int32, device=device)
        grad_initial_sample = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        block_size = 16
        if ctx.saved_buffer.iters.sum() > 0:
            dual_model = (
                mean,
                scale,
                quat,
                density,
                sh_coeffs,
                grad_means,
                grad_scales,
                grad_quats,
                grad_densities,
                grad_sh_coeffs,
                grad_ray_origin,
                grad_ray_direction,
                grad_means2d,
            )

            backwards_kernel.backwards_kernel(
                (block_size, 1, 1),
                (num_rays // block_size + 1, 1, 1),
                ctx.saved_buffer.states,
                ctx.saved_buffer.diracs,
                ctx.saved_buffer.iters,
                prim_sequence,
                ray_origin,
                ray_direction,
                dual_model,
                initial_sample,
                grad_initial_sample,
                hit_count,
                grad_output.contiguous(),
                world_to_clip if world_to_clip is not None else torch.ones((1, 4, 4), device=device, dtype=torch.float32),
                ctx.t_near,
                ctx.t_far,
                ctx.max_prim_size,
                ctx.max_samples,
            )

            if initial_inds.shape[0] > 0:
                ray_block_size = 64
                prim_block_size = 16
                backwards_kernel.backwards_initial_drgb_kernel(
                    (ray_block_size, prim_block_size, 1),
                    (
                        ray_origin.shape[0] // ray_block_size + 1,
                        initial_inds.shape[0] // prim_block_size + 1,
                        1,
                    ),
                    ray_origin,
                    ray_direction,
                    dual_model,
                    initial_sample,
                    initial_inds,
                    grad_initial_sample,
                    hit_count,
                    ctx.t_near,
                )

        # Gradient clipping for numerical stability
        grad_clip = 1e3
        mean_clip = 1e3
        grad_means2d = None if world_to_clip is None else grad_means2d

        return (
            grad_means.clip(min=-mean_clip, max=mean_clip),
            grad_scales.clip(min=-grad_clip, max=grad_clip),
            grad_quats.clip(min=-grad_clip, max=grad_clip),
            grad_densities.clip(min=-50, max=50).reshape(density.shape),
            grad_sh_coeffs.clip(min=-grad_clip, max=grad_clip),
            grad_ray_origin.clip(min=-grad_clip, max=grad_clip),
            grad_ray_direction.clip(min=-grad_clip, max=grad_clip),
            None,  # t_near
            None,  # t_far
            None,  # max_prim_size
            grad_means2d,
            None,  # world_to_clip
            None,  # max_samples
            None,  # return_extras
        )


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    sh_coeffs: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    t_near: float = 0.0,
    t_far: float = 1000,
    max_prim_size: float = 3,
    grad_means2d: Optional[torch.Tensor] = None,
    world_to_clip: Optional[torch.Tensor] = None,
    max_samples: int = 500,
    return_extras: bool = False,
):
    """
    Trace rays through ellipsoid primitives using differentiable volume rendering.

    Parameters
    ----------
    mean : torch.Tensor
        Primitive centers, shape (N, 3)
    scale : torch.Tensor
        Primitive axis scales, shape (N, 3)
    quat : torch.Tensor
        Primitive rotations as quaternions (w, x, y, z), shape (N, 4)
    density : torch.Tensor
        Primitive peak densities, shape (N,) or (N, 1)
    sh_coeffs : torch.Tensor
        Spherical harmonics coefficients for view-dependent color, shape (N, D, 3)
        where D is the number of SH coefficients
    ray_origin : torch.Tensor
        Ray origins, shape (M, 3)
    ray_direction : torch.Tensor
        Ray directions (should be normalized), shape (M, 3)
    t_near : float
        Near clipping distance along ray
    t_far : float
        Far clipping distance along ray
    max_prim_size : float
        Maximum primitive size for acceleration structure
    grad_means2d : torch.Tensor, optional
        Output tensor for 2D mean gradients (for screen-space loss)
    world_to_clip : torch.Tensor, optional
        World-to-clip transformation matrices for 2D gradient computation
    max_samples : int
        Maximum number of primitive intersections per ray
    return_extras : bool
        If True, returns additional rendering information

    Returns
    -------
    torch.Tensor or tuple
        If return_extras is False: rendered colors with distortion loss, shape (M, 5)
        If return_extras is True: (colors, extras_dict)
    """
    return VolumeRenderer.apply(
        mean,
        scale,
        quat,
        density,
        sh_coeffs,
        ray_origin,
        ray_direction,
        t_near,
        t_far,
        max_prim_size,
        grad_means2d,
        world_to_clip,
        max_samples,
        return_extras,
    )


# =============================================================================
# SHEvaluator - Spherical Harmonics Evaluation
# =============================================================================

class SHEvaluator(Function):
    """
    Differentiable spherical harmonics evaluator.

    Evaluates view-dependent color from SH coefficients for a given view direction.
    """

    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        sh_coeffs: torch.Tensor,
        view_origin: torch.Tensor,
        sh_degree: int,
    ):
        block_size = 64
        view_origin = view_origin.reshape(3).contiguous()
        means = means.contiguous()
        sh_coeffs = sh_coeffs.contiguous()
        color = torch.zeros_like(means)
        ctx.sh_degree = sh_degree
        num_prims = means.shape[0]

        sh_kernel.sh_kernel(
            (block_size, 1, 1),
            (num_prims // block_size + 1, 1, 1),
            means,
            sh_coeffs,
            view_origin,
            color,
            sh_degree,
        )

        ctx.save_for_backward(means, sh_coeffs, view_origin, color)
        return color

    @staticmethod
    def backward(ctx, grad_color: torch.Tensor):
        block_size = 64
        means, sh_coeffs, view_origin, color = ctx.saved_tensors
        num_prims = means.shape[0]
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)
        grad_color = grad_color.contiguous()

        sh_kernel.bw_sh_kernel(
            (block_size, 1, 1),
            ((num_prims + block_size - 1) // block_size, 1, 1),
            means,
            sh_coeffs,
            grad_sh_coeffs,
            view_origin,
            grad_color,
            ctx.sh_degree,
        )

        return None, grad_sh_coeffs, None, None


def eval_sh(
    means: torch.Tensor,
    sh_coeffs: torch.Tensor,
    view_origin: torch.Tensor,
    sh_degree: int,
):
    """
    Evaluate spherical harmonics to compute view-dependent colors.

    Parameters
    ----------
    means : torch.Tensor
        Primitive centers, shape (N, 3)
    sh_coeffs : torch.Tensor
        SH coefficients, shape (N, D, 3) where D = (degree+1)^2
    view_origin : torch.Tensor
        Camera/view position, shape (3,) or (1, 3)
    sh_degree : int
        Degree of spherical harmonics (0-3)

    Returns
    -------
    torch.Tensor
        Evaluated colors, shape (N, 3)
    """
    return SHEvaluator.apply(means, sh_coeffs, view_origin, sh_degree)


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================
SplineTracer = VolumeRenderer
EvalSH = SHEvaluator
