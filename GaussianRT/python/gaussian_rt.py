"""
GaussianRT - PyTorch Integration

Provides PyTorch autograd Function for differentiable Gaussian ray tracing.
"""

import torch
from torch.autograd import Function
from typing import Optional, Tuple, Dict, Any

# Try to import the C++ extension
try:
    import gaussian_rt_ext as _C
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False
    print("Warning: gaussian_rt_ext not built. Using placeholder implementation.")


# Global state for lazy initialization
_global_state = {
    'device': None,
    'forward_renderer': None,
    'backward_pass': None,
    'primitives': None,
    'accel_struct': None,
    'cuda_device_id': -1,
}


def _ensure_initialized(cuda_device_id: int):
    """Ensure the rendering backend is initialized for the given CUDA device."""
    global _global_state

    if not HAS_EXTENSION:
        return

    if _global_state['cuda_device_id'] != cuda_device_id:
        # Need to reinitialize for new device
        _global_state['device'] = _C.Device()
        if not _global_state['device'].initialize(cuda_device_id):
            raise RuntimeError(f"Failed to initialize device on CUDA:{cuda_device_id}")

        _global_state['forward_renderer'] = _C.ForwardRenderer(_global_state['device'])
        if not _global_state['forward_renderer'].initialize(enable_backward=True):
            raise RuntimeError("Failed to initialize forward renderer")

        _global_state['backward_pass'] = _C.BackwardPass(_global_state['device'])
        if not _global_state['backward_pass'].initialize():
            raise RuntimeError("Failed to initialize backward pass")

        _global_state['cuda_device_id'] = cuda_device_id


class GaussianRayTrace(Function):
    """
    PyTorch autograd Function for differentiable Gaussian ray tracing.

    Forward pass traces rays through Gaussian primitives and accumulates
    color using volume rendering.

    Backward pass computes gradients for all primitive parameters.
    """

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        densities: torch.Tensor,
        features: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        tmin: float = 0.01,
        tmax: float = 100.0,
        max_iters: int = 512,
        sh_degree: int = 0,
        max_prim_size: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass: differentiable Gaussian ray tracing.

        Args:
            means: (N, 3) Gaussian centers
            scales: (N, 3) Gaussian scales
            quats: (N, 4) Rotation quaternions (x, y, z, w)
            densities: (N,) Opacity values
            features: (N, F) Color features (RGB or SH coefficients)
            ray_origins: (M, 3) Ray origin points
            ray_directions: (M, 3) Ray direction vectors
            tmin: Minimum ray parameter
            tmax: Maximum ray parameter
            max_iters: Maximum iterations per ray
            sh_degree: Spherical harmonics degree (0-3)
            max_prim_size: Maximum primitive size for culling

        Returns:
            color: (M, 4) RGBA output for each ray
        """
        # Validate inputs
        assert means.dim() == 2 and means.size(1) == 3, "means must be (N, 3)"
        assert scales.dim() == 2 and scales.size(1) == 3, "scales must be (N, 3)"
        assert quats.dim() == 2 and quats.size(1) == 4, "quats must be (N, 4)"
        assert densities.dim() == 1, "densities must be (N,)"
        assert features.dim() == 2, "features must be (N, F)"
        assert ray_origins.dim() == 2 and ray_origins.size(1) == 3
        assert ray_directions.dim() == 2 and ray_directions.size(1) == 3

        device = means.device
        assert device.type == 'cuda', "All tensors must be on CUDA"

        cuda_device_id = device.index if device.index is not None else 0
        _ensure_initialized(cuda_device_id)

        num_rays = ray_origins.size(0)
        num_prims = means.size(0)
        feature_size = features.size(1)

        if HAS_EXTENSION:
            # Use C++ extension
            global _global_state

            # Create primitives
            prims = _C.Primitives(_global_state['device'])
            prims.set_data_from_tensors(means, scales, quats, densities, features)

            # Build acceleration structure
            accel = _C.AccelStruct(_global_state['device'])
            accel.build(prims, allow_update=True, fast_build=True)

            # Trace rays
            output = _global_state['forward_renderer'].trace_rays(
                accel, prims,
                ray_origins.contiguous(),
                ray_directions.contiguous(),
                tmin, tmax, max_iters, sh_degree, max_prim_size
            )

            color = output['color']

            # Save for backward
            ctx.save_for_backward(
                means, scales, quats, densities, features,
                ray_origins, ray_directions,
                output['state'], output['tri_collection'], output['iters']
            )
            ctx.params = {
                'tmin': tmin,
                'tmax': tmax,
                'max_iters': max_iters,
                'sh_degree': sh_degree,
                'max_prim_size': max_prim_size,
            }
            ctx.num_prims = num_prims
            ctx.num_rays = num_rays
            ctx.feature_size = feature_size
            ctx.prims = prims
        else:
            # Fallback: placeholder implementation
            color = torch.zeros(num_rays, 4, device=device, dtype=torch.float32)

            ctx.save_for_backward(
                means, scales, quats, densities, features,
                ray_origins, ray_directions
            )
            ctx.params = {
                'tmin': tmin,
                'tmax': tmax,
                'max_iters': max_iters,
                'sh_degree': sh_degree,
            }
            ctx.num_prims = num_prims
            ctx.num_rays = num_rays
            ctx.feature_size = feature_size

        return color

    @staticmethod
    def backward(
        ctx,
        grad_color: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: compute gradients for all parameters.

        Args:
            grad_color: (M, 4) Gradient of loss w.r.t. output color

        Returns:
            Gradients for all input parameters
        """
        if HAS_EXTENSION:
            (means, scales, quats, densities, features,
             ray_origins, ray_directions,
             last_state, tri_collection, iters) = ctx.saved_tensors

            global _global_state

            # Compute gradients
            grads = _global_state['backward_pass'].compute(
                last_state, tri_collection, iters,
                grad_color.contiguous(),
                ctx.prims,
                ray_origins, ray_directions,
                ctx.params['tmin'],
                ctx.params['tmax'],
                ctx.params['max_iters'],
                ctx.params['sh_degree']
            )

            grad_means = grads['dMeans'].clamp(-1e3, 1e3)
            grad_scales = grads['dScales'].clamp(-1e3, 1e3)
            grad_quats = grads['dQuats'].clamp(-1e3, 1e3)
            grad_densities = grads['dDensities'].clamp(-50, 50)
            grad_features = grads['dFeatures'].clamp(-1e3, 1e3)
        else:
            # Fallback: zero gradients
            (means, scales, quats, densities, features,
             ray_origins, ray_directions) = ctx.saved_tensors

            grad_means = torch.zeros_like(means)
            grad_scales = torch.zeros_like(scales)
            grad_quats = torch.zeros_like(quats)
            grad_densities = torch.zeros_like(densities)
            grad_features = torch.zeros_like(features)

        # Return gradients for all inputs
        # (means, scales, quats, densities, features, ray_origins, ray_directions,
        #  tmin, tmax, max_iters, sh_degree, max_prim_size)
        return (
            grad_means,
            grad_scales,
            grad_quats,
            grad_densities,
            grad_features,
            None,  # ray_origins
            None,  # ray_directions
            None,  # tmin
            None,  # tmax
            None,  # max_iters
            None,  # sh_degree
            None,  # max_prim_size
        )


def render_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    densities: torch.Tensor,
    features: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    tmin: float = 0.01,
    tmax: float = 100.0,
    max_iters: int = 512,
    sh_degree: int = 0,
    max_prim_size: float = 1.0,
) -> torch.Tensor:
    """
    Render 3D Gaussians using differentiable ray tracing.

    This is a convenience wrapper around GaussianRayTrace.apply().

    Args:
        means: (N, 3) Gaussian centers
        scales: (N, 3) Gaussian scales (radii along local axes)
        quats: (N, 4) Rotation quaternions (x, y, z, w format)
        densities: (N,) Opacity/density values
        features: (N, F) Color features (RGB for F=3, or SH coefficients)
        ray_origins: (M, 3) Ray origin points
        ray_directions: (M, 3) Ray direction vectors (need not be normalized)
        tmin: Minimum ray parameter t
        tmax: Maximum ray parameter t
        max_iters: Maximum number of Gaussian intersections per ray
        sh_degree: Spherical harmonics degree (0-3)
        max_prim_size: Maximum primitive size for culling

    Returns:
        color: (M, 4) RGBA output for each ray
            - RGB: Accumulated color from volume rendering
            - A: Remaining transmittance (1 = fully transparent, 0 = fully opaque)

    Example:
        >>> import torch
        >>> from gaussian_rt import render_gaussians
        >>>
        >>> # Create 100 random Gaussians
        >>> means = torch.randn(100, 3, device='cuda')
        >>> scales = torch.ones(100, 3, device='cuda') * 0.1
        >>> quats = torch.tensor([[0, 0, 0, 1]], device='cuda').expand(100, 4)
        >>> densities = torch.ones(100, device='cuda')
        >>> colors = torch.rand(100, 3, device='cuda')
        >>>
        >>> # Generate rays for a 64x64 image
        >>> ray_o = torch.zeros(64*64, 3, device='cuda')
        >>> ray_o[:, 2] = -5  # Camera at z=-5
        >>> ray_d = torch.zeros(64*64, 3, device='cuda')
        >>> ray_d[:, 2] = 1   # Looking towards +z
        >>>
        >>> # Render
        >>> output = render_gaussians(means, scales, quats, densities, colors, ray_o, ray_d)
        >>> image = output[:, :3].reshape(64, 64, 3)
    """
    return GaussianRayTrace.apply(
        means, scales, quats, densities, features,
        ray_origins, ray_directions,
        tmin, tmax, max_iters, sh_degree, max_prim_size
    )


class GaussianRenderer:
    """
    High-level Gaussian renderer with state management.

    Caches acceleration structures and provides efficient updates
    for iterative rendering (e.g., during optimization).
    """

    def __init__(self, cuda_device_id: int = 0, enable_backward: bool = True):
        """
        Initialize the renderer.

        Args:
            cuda_device_id: CUDA device to use
            enable_backward: Enable gradient computation
        """
        self.cuda_device_id = cuda_device_id
        self.enable_backward = enable_backward
        self.initialized = False

        self._device = None
        self._prims = None
        self._accel = None
        self._forward = None
        self._backward = None

    def _ensure_initialized(self):
        """Initialize backend if not already done."""
        if self.initialized or not HAS_EXTENSION:
            return

        self._device = _C.Device()
        if not self._device.initialize(self.cuda_device_id):
            raise RuntimeError("Failed to initialize device")

        self._forward = _C.ForwardRenderer(self._device)
        if not self._forward.initialize(enable_backward=self.enable_backward):
            raise RuntimeError("Failed to initialize forward renderer")

        if self.enable_backward:
            self._backward = _C.BackwardPass(self._device)
            if not self._backward.initialize():
                raise RuntimeError("Failed to initialize backward pass")

        self.initialized = True

    def set_primitives(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        densities: torch.Tensor,
        features: torch.Tensor,
    ):
        """
        Set Gaussian primitive data.

        Args:
            means: (N, 3) Gaussian centers
            scales: (N, 3) Gaussian scales
            quats: (N, 4) Rotation quaternions
            densities: (N,) Opacity values
            features: (N, F) Color features
        """
        self._ensure_initialized()

        if not HAS_EXTENSION:
            self._cached_primitives = (means, scales, quats, densities, features)
            return

        self._prims = _C.Primitives(self._device)
        self._prims.set_data_from_tensors(means, scales, quats, densities, features)

        # Build acceleration structure
        self._accel = _C.AccelStruct(self._device)
        self._accel.build(self._prims, allow_update=True, fast_build=True)

    def update_primitives(
        self,
        means: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ):
        """
        Update primitive data and refit acceleration structure.

        Only updates the provided parameters. Others remain unchanged.
        """
        if not HAS_EXTENSION:
            return

        # Update data
        if self._prims is not None:
            self._prims.update_aabbs()
            if self._accel is not None:
                self._accel.update(self._prims)

    def render(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        tmin: float = 0.01,
        tmax: float = 100.0,
        max_iters: int = 512,
        sh_degree: int = 0,
    ) -> torch.Tensor:
        """
        Render the current primitives.

        Args:
            ray_origins: (M, 3) Ray origins
            ray_directions: (M, 3) Ray directions
            tmin: Minimum ray parameter
            tmax: Maximum ray parameter
            max_iters: Maximum iterations per ray
            sh_degree: Spherical harmonics degree

        Returns:
            color: (M, 4) RGBA output
        """
        if not HAS_EXTENSION:
            num_rays = ray_origins.size(0)
            return torch.zeros(num_rays, 4, device=ray_origins.device)

        self._ensure_initialized()

        output = self._forward.trace_rays(
            self._accel, self._prims,
            ray_origins.contiguous(),
            ray_directions.contiguous(),
            tmin, tmax, max_iters, sh_degree, 1.0
        )

        return output['color']

    def synchronize(self):
        """Synchronize all pending operations."""
        if self._device is not None:
            self._device.synchronize()
