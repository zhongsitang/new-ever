"""
PyTorch integration for GaussianRT volume renderer.

This module provides a differentiable volume rendering function that can be
used in PyTorch training pipelines.
"""

import torch
from torch.autograd import Function
from typing import Tuple, Optional, Dict, Any

# Import C++ extension (will be available after build)
try:
    from . import _gaussian_rt
except ImportError:
    _gaussian_rt = None


class VolumeRenderFunction(Function):
    """
    PyTorch autograd function for differentiable volume rendering.

    Forward pass uses Slang-RHI ray tracing to render ellipsoid volume elements.
    Backward pass uses Slang autodiff to compute gradients.
    """

    @staticmethod
    def forward(
        ctx,
        positions: torch.Tensor,      # [N, 3] Element centers
        scales: torch.Tensor,         # [N, 3] Element scales
        rotations: torch.Tensor,      # [N, 4] Element rotations (quaternions)
        opacities: torch.Tensor,      # [N] Element opacities
        features: torch.Tensor,       # [N, F] SH coefficients
        ray_origins: torch.Tensor,    # [R, 3] Ray origins
        ray_directions: torch.Tensor, # [R, 3] Ray directions
        t_min: float,
        t_max: float,
        max_samples: int,
        sh_degree: int,
        renderer: Any,
        compute_depth: bool = True,
        compute_distortion: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: trace rays and compute colors.

        Args:
            positions: [N, 3] Center positions of volume elements
            scales: [N, 3] Scale factors (semi-axes)
            rotations: [N, 4] Rotation quaternions (w, x, y, z)
            opacities: [N] Opacity values
            features: [N, F] Spherical harmonics coefficients
            ray_origins: [R, 3] Ray origin points
            ray_directions: [R, 3] Ray direction vectors (normalized)
            t_min: Minimum ray parameter
            t_max: Maximum ray parameter
            max_samples: Maximum samples per ray
            sh_degree: SH evaluation degree (0-3)
            renderer: VolumeRenderer instance
            compute_depth: Whether to compute depth output
            compute_distortion: Whether to compute distortion loss

        Returns:
            colors: [R, 4] RGBA colors
            depths: [R] Depth values (if compute_depth)
            distortions: [R] Distortion loss values (if compute_distortion)
        """
        # Ensure contiguous
        positions = positions.contiguous()
        scales = scales.contiguous()
        rotations = rotations.contiguous()
        opacities = opacities.contiguous()
        features = features.contiguous()
        ray_origins = ray_origins.contiguous()
        ray_directions = ray_directions.contiguous()

        # Build acceleration structure
        renderer.build_accel(positions, scales, rotations)

        # Forward pass
        colors, states, last_points, sample_counts, sample_indices, touch_counts = \
            renderer.forward(
                positions, scales, rotations, opacities, features,
                ray_origins, ray_directions,
                t_min, t_max, max_samples, sh_degree
            )

        # Extract depth and distortion from states
        # State layout: [log_T, color.xyz, depth, depth_weight, t_current, distortion.xy, weight.xy]
        num_rays = ray_origins.size(0)
        depths = states[:, 4] / torch.clamp(states[:, 5], min=1e-6)
        distortions = states[:, 7]

        # Save for backward
        ctx.save_for_backward(
            states, last_points, sample_counts, sample_indices,
            positions, scales, rotations, opacities, features,
            ray_origins, ray_directions
        )
        ctx.t_min = t_min
        ctx.t_max = t_max
        ctx.max_samples = max_samples
        ctx.sh_degree = sh_degree
        ctx.renderer = renderer
        ctx.compute_depth = compute_depth
        ctx.compute_distortion = compute_distortion

        return colors, depths, distortions

    @staticmethod
    def backward(
        ctx,
        grad_colors: torch.Tensor,
        grad_depths: Optional[torch.Tensor],
        grad_distortions: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: compute gradients.

        Args:
            grad_colors: [R, 4] Gradient w.r.t. output colors
            grad_depths: [R] Gradient w.r.t. depth (or None)
            grad_distortions: [R] Gradient w.r.t. distortion (or None)

        Returns:
            Gradients for all inputs (or None for non-differentiable inputs)
        """
        (states, last_points, sample_counts, sample_indices,
         positions, scales, rotations, opacities, features,
         ray_origins, ray_directions) = ctx.saved_tensors

        num_rays = ray_origins.size(0)

        # Handle None gradients
        if grad_depths is None:
            grad_depths = torch.zeros(num_rays, device=grad_colors.device)
        if grad_distortions is None:
            grad_distortions = torch.zeros(num_rays, device=grad_colors.device)

        # Ensure contiguous
        grad_colors = grad_colors.contiguous()
        grad_depths = grad_depths.contiguous()
        grad_distortions = grad_distortions.contiguous()

        # Backward pass
        (grad_positions, grad_scales, grad_rotations,
         grad_opacities, grad_features,
         grad_ray_origins, grad_ray_dirs) = ctx.renderer.backward(
            states, last_points, sample_counts, sample_indices,
            positions, scales, rotations, opacities, features,
            ray_origins, ray_directions,
            grad_colors, grad_depths, grad_distortions,
            ctx.t_min, ctx.t_max, ctx.max_samples, ctx.sh_degree
        )

        return (
            grad_positions,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_features,
            grad_ray_origins,
            grad_ray_dirs,
            None,  # t_min
            None,  # t_max
            None,  # max_samples
            None,  # sh_degree
            None,  # renderer
            None,  # compute_depth
            None,  # compute_distortion
        )


class VolumeRendererModule(torch.nn.Module):
    """
    PyTorch module wrapper for volume rendering.

    Example usage:
        renderer = VolumeRendererModule(device_index=0)

        # In training loop:
        colors, depths, distortions = renderer(
            positions, scales, rotations, opacities, features,
            ray_origins, ray_directions
        )
        loss = F.mse_loss(colors, target_colors)
        loss.backward()
    """

    def __init__(
        self,
        device_index: int = 0,
        shader_dir: str = "",
        t_min: float = 0.1,
        t_max: float = 100.0,
        max_samples: int = 128,
        sh_degree: int = 3,
    ):
        super().__init__()

        if _gaussian_rt is None:
            raise RuntimeError(
                "GaussianRT C++ extension not loaded. "
                "Please build and install the extension first."
            )

        self._renderer = _gaussian_rt.VolumeRenderer(device_index, shader_dir)
        self.t_min = t_min
        self.t_max = t_max
        self.max_samples = max_samples
        self.sh_degree = sh_degree

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        max_samples: Optional[int] = None,
        sh_degree: Optional[int] = None,
        compute_depth: bool = True,
        compute_distortion: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render volume elements.

        Args:
            positions: [N, 3] Element centers
            scales: [N, 3] Element scales
            rotations: [N, 4] Element rotations
            opacities: [N] Element opacities
            features: [N, F] SH coefficients
            ray_origins: [R, 3] Ray origins
            ray_directions: [R, 3] Ray directions
            t_min: Optional override for minimum t
            t_max: Optional override for maximum t
            max_samples: Optional override for max samples
            sh_degree: Optional override for SH degree
            compute_depth: Whether to compute depth
            compute_distortion: Whether to compute distortion

        Returns:
            colors: [R, 4] RGBA
            depths: [R] Depth values
            distortions: [R] Distortion values
        """
        return VolumeRenderFunction.apply(
            positions,
            scales,
            rotations,
            opacities,
            features,
            ray_origins,
            ray_directions,
            t_min if t_min is not None else self.t_min,
            t_max if t_max is not None else self.t_max,
            max_samples if max_samples is not None else self.max_samples,
            sh_degree if sh_degree is not None else self.sh_degree,
            self._renderer,
            compute_depth,
            compute_distortion,
        )

    def render_image(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        features: torch.Tensor,
        camera: Dict[str, Any],
        width: int,
        height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a full image from camera parameters.

        Args:
            positions, scales, rotations, opacities, features: Element data
            camera: Dict with 'position', 'forward', 'up', 'right', 'focal_length'
            width: Image width
            height: Image height

        Returns:
            colors: [H, W, 4] RGBA image
            depths: [H, W] Depth map
            distortions: [H, W] Distortion map
        """
        device = positions.device

        # Generate rays
        ray_origins, ray_directions = self._generate_rays(
            camera, width, height, device
        )

        # Render
        colors, depths, distortions = self.forward(
            positions, scales, rotations, opacities, features,
            ray_origins, ray_directions
        )

        # Reshape to image
        colors = colors.view(height, width, 4)
        depths = depths.view(height, width)
        distortions = distortions.view(height, width)

        return colors, depths, distortions

    def _generate_rays(
        self,
        camera: Dict[str, Any],
        width: int,
        height: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for a pinhole camera."""
        # Camera parameters
        position = torch.tensor(camera['position'], device=device, dtype=torch.float32)
        forward = torch.tensor(camera['forward'], device=device, dtype=torch.float32)
        up = torch.tensor(camera['up'], device=device, dtype=torch.float32)
        right = torch.tensor(camera['right'], device=device, dtype=torch.float32)
        focal_length = camera.get('focal_length', 1.0)
        sensor_width = camera.get('sensor_width', 1.0)
        sensor_height = camera.get('sensor_height', height / width * sensor_width)

        # Pixel coordinates
        u = torch.linspace(-0.5, 0.5, width, device=device)
        v = torch.linspace(-0.5, 0.5, height, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')

        # Scale to sensor size
        uu = uu * sensor_width
        vv = vv * sensor_height

        # Ray directions
        directions = (
            forward * focal_length +
            right.unsqueeze(0).unsqueeze(0) * uu.unsqueeze(-1) +
            up.unsqueeze(0).unsqueeze(0) * (-vv.unsqueeze(-1))
        )
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        directions = directions.view(-1, 3)

        # Ray origins (all same for pinhole)
        origins = position.unsqueeze(0).expand(width * height, 3).contiguous()

        return origins, directions


def render_volume(
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_min: float = 0.1,
    t_max: float = 100.0,
    max_samples: int = 128,
    sh_degree: int = 3,
    renderer: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional interface for volume rendering.

    Creates a temporary renderer if none provided.
    For repeated calls, prefer VolumeRendererModule.

    Returns:
        colors: [R, 4] RGBA
        depths: [R] Depth
        distortions: [R] Distortion
    """
    if renderer is None:
        if _gaussian_rt is None:
            raise RuntimeError("GaussianRT extension not loaded")
        renderer = _gaussian_rt.VolumeRenderer(0, "")

    return VolumeRenderFunction.apply(
        positions, scales, rotations, opacities, features,
        ray_origins, ray_directions,
        t_min, t_max, max_samples, sh_degree,
        renderer, True, True
    )
