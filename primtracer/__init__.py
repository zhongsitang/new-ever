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
import math

import torch
from torch.autograd import Function

from . import optix_tracer as _C
from . import slang_backward as _bw
from . import slang_sh as _sh

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
    def forward(ctx: Any, mean, scale, quat, density, features, rayo, rayd, tmin, tmax, min_logT, max_hits):
        tracer = _get_tracer(mean.device.index or 0)
        tracer.update_primitives(mean, scale, quat, density, features)
        out = tracer.trace_rays(rayo, rayd, tmin, tmax, min_logT, max_hits)

        ctx.tmin, ctx.max_hits = tmin, max_hits
        ctx.save_for_backward(
            mean, scale, quat, density, features,
            rayo, rayd, tmax,
            out["last_state"], out["last_contrib"], out["ray_hits"], out["hit_collection"],
        )
        return out["color"], out["depth"], out["ray_hits"], out["hit_collection"], out["prim_hits"]

    @staticmethod
    def backward(ctx, dL_color, dL_depth, *_):
        (mean, scale, quat, density, features,
         rayo, rayd, tmax,
         last_state, last_contrib, ray_hits, hit_collection) = ctx.saved_tensors

        N, M = mean.shape[0], rayo.shape[0]
        dev = mean.device

        # Allocate gradient tensors
        dL_mean = torch.zeros(N, 3, device=dev)
        dL_scale = torch.zeros(N, 3, device=dev)
        dL_quat = torch.zeros(N, 4, device=dev)
        dL_density = torch.zeros(N, device=dev)
        dL_features = torch.zeros_like(features)
        dL_rayo = torch.zeros(M, 3, device=dev)
        dL_rayd = torch.zeros(M, 3, device=dev)

        if ray_hits.sum() > 0:
            dL_out = torch.cat([dL_color, dL_depth.unsqueeze(-1)], dim=-1).contiguous()

            _bw.bw_trace_rays(
                (16, 1, 1), (_div_up(M, 16), 1, 1),
                (mean, scale, quat, density, features,
                 dL_mean, dL_scale, dL_quat, dL_density, dL_features),
                (rayo, rayd, tmax, ctx.tmin, dL_rayo, dL_rayd),
                (last_state, last_contrib, ray_hits, hit_collection),
                dL_out, ctx.max_hits,
            )

        clip = 1e3
        return (
            dL_mean.clamp(-clip, clip),
            dL_scale.clamp(-clip, clip),
            dL_quat.clamp(-clip, clip),
            dL_density.clamp(-clip, clip),
            dL_features.clamp(-clip, clip),
            dL_rayo.clamp(-clip, clip),
            dL_rayd.clamp(-clip, clip),
            None, None, None, None,
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
    tmax: torch.Tensor | float = 1e7,
    min_logT: float = math.log(1e-3),
    max_hits: int = 500,
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
        min_logT: log(T) cutoff (stop when logT <= min_logT)
        max_hits: Maximum number of hits per ray
        return_extras: Return additional hit info

    Returns:
        (color, depth) or (color, depth, extras) if return_extras=True
    """
    M = rayo.shape[0]

    if isinstance(tmax, (int, float)):
        tmax = torch.full((M,), tmax, dtype=torch.float32, device=rayo.device)
    else:
        tmax = tmax.to(dtype=torch.float32, device=rayo.device)
        if tmax.numel() == 1:
            tmax = tmax.expand(M).contiguous()

    color, depth, ray_hits, hit_collection, prim_hits = _PrimTracerFn.apply(
        mean.contiguous(),
        scale.contiguous(),
        quat.contiguous(),
        density.view(-1).contiguous(),
        features.contiguous(),
        rayo.contiguous(),
        rayd.contiguous(),
        tmin,
        tmax,
        min_logT,
        max_hits,
    )

    if return_extras:
        extras = {"ray_hits": ray_hits, "hit_collection": hit_collection, "prim_hits": prim_hits}
        return color, depth, extras
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
