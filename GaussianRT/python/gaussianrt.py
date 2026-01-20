"""
GaussianRT - Hardware Ray Tracing for Gaussian Volume Rendering

Python interface with PyTorch autograd support.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.autograd import Function

# Import the compiled extension
try:
    import gaussianrt_ext
except ImportError:
    raise ImportError(
        "GaussianRT CUDA extension not found. "
        "Please build the extension first with: python setup.py install"
    )


# Global context (created on first use)
_context: Optional[gaussianrt_ext.Context] = None
_current_device: int = -1


def _get_context(device: torch.device) -> gaussianrt_ext.Context:
    """Get or create the GaussianRT context for the given device."""
    global _context, _current_device

    device_index = device.index if device.index is not None else 0

    if _context is None or _current_device != device_index:
        _context = gaussianrt_ext.Context(device_index)
        _current_device = device_index

    return _context


def _get_shader_path() -> str:
    """Get path to the ray tracer shader."""
    module_dir = Path(__file__).parent.parent
    shader_path = module_dir / "shaders" / "raytracer.slang"
    if not shader_path.exists():
        # Try build directory
        shader_path = module_dir / "build" / "shaders" / "raytracer.slang"
    return str(shader_path)


class GaussianRTFunction(Function):
    """Autograd function for differentiable Gaussian ray tracing."""

    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        densities: torch.Tensor,
        features: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        tmin: float,
        tmax: float,
        max_prim_size: float,
        means2D: Optional[torch.Tensor],
        wcts: Optional[torch.Tensor],
        max_iters: int,
        return_extras: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass: render Gaussians using ray tracing."""

        device = means.device
        ctx.device = device

        # Ensure contiguous tensors
        means = means.contiguous()
        scales = scales.contiguous()
        quats = quats.contiguous()
        densities = densities.contiguous()
        features = features.contiguous()
        ray_origins = ray_origins.contiguous()
        ray_directions = ray_directions.contiguous()

        # Get context and set up
        context = _get_context(device)
        context.build_accel(means, scales, quats, densities)
        context.create_tracer(_get_shader_path())
        context.set_primitives(means, scales, quats, densities, features)

        # Compute SH degree from feature size
        feature_count = features.size(1)
        sh_degree = int((feature_count ** 0.5)) - 1

        # Trace rays
        result = context.trace_rays(
            ray_origins,
            ray_directions,
            tmin,
            tmax,
            max_iters,
            max_prim_size,
            sh_degree,
            save_for_backward=True
        )

        # Extract outputs
        color = result["color"]
        states = result["states"]

        # Compute distortion loss from states
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = distortion_pt1 - distortion_pt2

        # Combine color and distortion loss
        color_and_loss = torch.cat([color, distortion_loss.unsqueeze(1)], dim=1)

        # Save for backward
        ctx.save_for_backward(
            means, scales, quats, densities, features,
            ray_origins, ray_directions,
            result.get("tri_collection", torch.tensor([])),
            wcts if wcts is not None else torch.ones((1, 4, 4), device=device),
            result["initial_drgb"]
        )
        ctx.saved_result = result
        ctx.max_prim_size = max_prim_size
        ctx.tmin = tmin
        ctx.tmax = tmax
        ctx.max_iters = max_iters
        ctx.sh_degree = sh_degree

        if return_extras:
            extras = {
                "tri_collection": result.get("tri_collection"),
                "iters": result["iters"],
                "opacity": color[:, 3],
                "touch_count": result["touch_count"],
                "distortion_loss": distortion_loss,
                "states": states,
            }
            return color_and_loss, extras
        else:
            return (color_and_loss,)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        *args
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass: compute gradients."""

        (
            means, scales, quats, densities, features,
            ray_origins, ray_directions,
            tri_collection, wcts, initial_drgb
        ) = ctx.saved_tensors

        device = ctx.device
        result = ctx.saved_result

        # Get context
        context = _get_context(device)

        # Run backward pass
        grads = context.backward(
            grad_output.contiguous(),
            means, scales, quats, densities, features,
            ray_origins, ray_directions,
            result["states"], result["diracs"], result["iters"],
            tri_collection, initial_drgb, wcts,
            ctx.tmin, ctx.tmax, ctx.max_prim_size,
            ctx.max_iters, ctx.sh_degree
        )

        # Clip gradients for stability
        v = 1e3
        mean_v = 1e3

        dL_dmeans = grads["dL_dmeans"].clamp(min=-mean_v, max=mean_v)
        dL_dscales = grads["dL_dscales"].clamp(min=-v, max=v)
        dL_dquats = grads["dL_dquats"].clamp(min=-v, max=v)
        dL_ddensities = grads["dL_ddensities"].clamp(min=-50, max=50).reshape(densities.shape)
        dL_dfeatures = grads["dL_dfeatures"].clamp(min=-v, max=v)
        dL_dray_origins = grads["dL_dray_origins"].clamp(min=-v, max=v)
        dL_dray_dirs = grads["dL_dray_dirs"].clamp(min=-v, max=v)
        dL_dmeans2D = grads.get("dL_dmeans2D")

        return (
            dL_dmeans,
            dL_dscales,
            dL_dquats,
            dL_ddensities,
            dL_dfeatures,
            dL_dray_origins,
            dL_dray_dirs,
            None,  # tmin
            None,  # tmax
            None,  # max_prim_size
            dL_dmeans2D,  # means2D
            None,  # wcts
            None,  # max_iters
            None,  # return_extras
        )


def trace_rays(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    densities: torch.Tensor,
    features: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000.0,
    max_prim_size: float = 3.0,
    means2D: Optional[torch.Tensor] = None,
    wcts: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    return_extras: bool = False,
) -> torch.Tensor:
    """
    Trace rays through a scene of Gaussian primitives.

    Args:
        means: Primitive centers [N, 3]
        scales: Primitive scales [N, 3]
        quats: Primitive rotations as quaternions [N, 4]
        densities: Primitive densities [N]
        features: Spherical harmonics features [N, (sh_degree+1)^2, 3]
        ray_origins: Ray origins [R, 3]
        ray_directions: Ray directions [R, 3]
        tmin: Near plane distance
        tmax: Far plane distance
        max_prim_size: Maximum primitive size
        means2D: Optional 2D means for splatting gradient [N, 2]
        wcts: Optional world-to-clip transforms [R, 4, 4] or [1, 4, 4]
        max_iters: Maximum ray marching iterations
        return_extras: If True, return extra debug information

    Returns:
        Color and distortion loss tensor [R, 5] (RGBA + distortion)
        If return_extras=True, also returns a dict with debug info
    """
    result = GaussianRTFunction.apply(
        means,
        scales,
        quats,
        densities,
        features,
        ray_origins,
        ray_directions,
        tmin,
        tmax,
        max_prim_size,
        means2D,
        wcts,
        max_iters,
        return_extras,
    )

    if return_extras:
        return result
    else:
        return result[0]


# Convenience class for repeated rendering
class GaussianRenderer:
    """
    High-level renderer for Gaussian primitives.

    Example:
        renderer = GaussianRenderer()
        renderer.set_primitives(means, scales, quats, densities, features)
        color = renderer.render(ray_origins, ray_directions)
    """

    def __init__(self, device: torch.device = None):
        """Initialize renderer on given device."""
        if device is None:
            device = torch.device("cuda:0")
        self.device = device
        self._primitives_set = False

    def set_primitives(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        densities: torch.Tensor,
        features: torch.Tensor,
    ):
        """Set the Gaussian primitives to render."""
        self.means = means.to(self.device).contiguous()
        self.scales = scales.to(self.device).contiguous()
        self.quats = quats.to(self.device).contiguous()
        self.densities = densities.to(self.device).contiguous()
        self.features = features.to(self.device).contiguous()
        self._primitives_set = True

    def render(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        tmin: float = 0.0,
        tmax: float = 1000.0,
        max_prim_size: float = 3.0,
        max_iters: int = 500,
        return_extras: bool = False,
    ) -> torch.Tensor:
        """Render the scene from given rays."""
        if not self._primitives_set:
            raise RuntimeError("Call set_primitives() first")

        ray_origins = ray_origins.to(self.device).contiguous()
        ray_directions = ray_directions.to(self.device).contiguous()

        return trace_rays(
            self.means,
            self.scales,
            self.quats,
            self.densities,
            self.features,
            ray_origins,
            ray_directions,
            tmin=tmin,
            tmax=tmax,
            max_prim_size=max_prim_size,
            max_iters=max_iters,
            return_extras=return_extras,
        )
