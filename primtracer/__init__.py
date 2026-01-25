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

"""PrimTracer: Differentiable volume rendering for ellipsoid primitives."""

from typing import Any

import torch
from torch.autograd import Function

from . import ellipsoid_tracer as _C
from . import backwards_kernel as _bw
from . import sh_kernel as _sh

__all__ = ["trace_rays", "eval_sh", "RayTracer"]

# Re-export RayTracer for direct access
RayTracer = _C.RayTracer

# =============================================================================
# Internal: RayTracer cache for pipeline reuse
# =============================================================================

_tracers: dict[int, RayTracer] = {}


def _get_tracer(device: int) -> RayTracer:
    if device not in _tracers:
        _tracers[device] = RayTracer(device)
    return _tracers[device]


def _div_up(n: int, d: int) -> int:
    return (n + d - 1) // d


# =============================================================================
# PrimTracer: Differentiable volume rendering
# =============================================================================

class _PrimTracerFn(Function):
    @staticmethod
    def forward(ctx: Any, mean, scale, quat, density, features, rayo, rayd, tmin, tmax, max_iters):
        tracer = _get_tracer(mean.device.index or 0)
        tracer.update_primitives(
            mean.contiguous(), scale.contiguous(), quat.contiguous(),
            density.contiguous(), features.contiguous()
        )
        out = tracer.trace_rays(
            rayo.contiguous(), rayd.contiguous(), tmin, tmax.contiguous(), max_iters
        )

        ctx.tmin, ctx.max_iters = tmin, max_iters
        ctx.save_for_backward(
            mean, scale, quat, density, features, rayo, rayd, tmax,
            out["states"], out["delta_contribs"], out["iters"],
            out["hit_collection"],
        )
        return out["color"], out["depth"], out["iters"], out["hit_collection"], out["prim_hits"]

    @staticmethod
    def backward(ctx, dL_color, dL_depth, *_):
        mean, scale, quat, density, features, rayo, rayd, tmax, \
            states, delta_contribs, iters, hit_collection = ctx.saved_tensors

        N, M = mean.shape[0], rayo.shape[0]
        dev = mean.device

        # Allocate gradients
        dL = {
            "mean": torch.zeros(N, 3, device=dev),
            "scale": torch.zeros(N, 3, device=dev),
            "quat": torch.zeros(N, 4, device=dev),
            "density": torch.zeros(N, device=dev),
            "features": torch.zeros_like(features),
            "rayo": torch.zeros(M, 3, device=dev),
            "rayd": torch.zeros(M, 3, device=dev),
        }
        prim_hits = torch.zeros(N, dtype=torch.int32, device=dev)
        grad = torch.cat([dL_color, dL_depth.view(-1, 1)], dim=1).contiguous()

        if iters.sum() > 0:
            dual = (mean, scale, quat, density, features,
                    dL["mean"], dL["scale"], dL["quat"], dL["density"], dL["features"],
                    dL["rayo"], dL["rayd"])

            # Unified backward pass - handles both normal hits and virtual entry hits
            # for rays starting inside primitives
            _bw.backwards_kernel(
                (16, 1, 1), (_div_up(M, 16), 1, 1),
                states, delta_contribs, iters, hit_collection, rayo, rayd,
                dual, prim_hits,
                grad, ctx.tmin, tmax, 3.0, ctx.max_iters
            )

        clip = 1e3
        return (
            dL["mean"].clamp(-clip, clip),
            dL["scale"].clamp(-clip, clip),
            dL["quat"].clamp(-clip, clip),
            dL["density"].clamp(-clip, clip),
            dL["features"].clamp(-clip, clip),
            dL["rayo"].clamp(-clip, clip),
            dL["rayd"].clamp(-clip, clip),
            None, None, None  # tmin, tmax, max_iters
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
        scale: Primitive scales, shape (N, 3)
        quat: Primitive rotations (w,x,y,z), shape (N, 4)
        density: Primitive densities, shape (N,)
        features: SH features, shape (N, C, 3)
        rayo: Ray origins, shape (M, 3)
        rayd: Ray directions (normalized), shape (M, 3)
        tmin: Minimum ray t
        tmax: Maximum ray t (scalar or per-ray tensor)
        max_iters: Maximum hit iterations per ray
        return_extras: Return additional hit info

    Returns:
        (color, depth) or (color, depth, extras) if return_extras=True
    """
    M = rayo.shape[0]
    density = density.view(-1)

    if isinstance(tmax, (int, float)):
        tmax = torch.full((M,), tmax, dtype=torch.float32, device=rayo.device)
    else:
        tmax = tmax.to(dtype=torch.float32, device=rayo.device)
        if tmax.numel() == 1:
            tmax = tmax.expand(M).contiguous()

    color, depth, iters, hit_collection, prim_hits = _PrimTracerFn.apply(
        mean, scale, quat, density, features, rayo, rayd, tmin, tmax, max_iters
    )

    if return_extras:
        return color, depth, {"iters": iters, "hit_collection": hit_collection, "prim_hits": prim_hits}
    return color, depth


# =============================================================================
# EvalSH: Spherical harmonics evaluation
# =============================================================================

class _EvalSHFn(Function):
    @staticmethod
    def forward(ctx: Any, means, features, rayo, sh_degree):
        means, features = means.contiguous(), features.contiguous()
        rayo = rayo.view(3).contiguous()
        color = torch.zeros_like(means)
        N = means.shape[0]

        _sh.sh_kernel((64, 1, 1), (_div_up(N, 64), 1, 1), means, features, rayo, color, sh_degree)

        ctx.sh_degree = sh_degree
        ctx.save_for_backward(means, features, rayo)
        return color

    @staticmethod
    def backward(ctx, dL_color):
        means, features, rayo = ctx.saved_tensors
        dL_feat = torch.zeros_like(features)
        N = means.shape[0]

        _sh.bw_sh_kernel(
            (64, 1, 1), (_div_up(N, 64), 1, 1),
            means, features, dL_feat, rayo, dL_color.contiguous(), ctx.sh_degree
        )
        return None, dL_feat, None, None


def eval_sh(means: torch.Tensor, features: torch.Tensor, rayo: torch.Tensor, sh_degree: int) -> torch.Tensor:
    """
    Evaluate spherical harmonics for primitives.

    Args:
        means: Primitive centers, shape (N, 3)
        features: SH coefficients, shape (N, C, 3)
        rayo: Camera position, shape (3,)
        sh_degree: SH degree

    Returns:
        Colors, shape (N, 3)
    """
    return _EvalSHFn.apply(means, features, rayo, sh_degree)
