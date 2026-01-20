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
Test suite for GaussianRT ray tracing implementation.

Tests:
1. Forward pass comparison with JAX quadrature reference implementation
2. Backward pass gradient checking using torch.autograd.gradcheck
"""

import sys
from pathlib import Path
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import JAX reference implementation
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from utils import math_util
from jaxutil import safe_math, quadrature
from splinetracers import quad

# Import GaussianRT - conditional import based on availability
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "GaussianRT" / "python"))
    from gaussianrt import trace_rays as gaussianrt_trace_rays
    GAUSSIANRT_AVAILABLE = True
except ImportError as e:
    GAUSSIANRT_AVAILABLE = False
    GAUSSIANRT_IMPORT_ERROR = str(e)

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Strict tolerances for numerical comparison
FORWARD_ATOL = 5e-4
FORWARD_RTOL = 5e-4
GRAD_ATOL = 1e-4
GRAD_RTOL = 1e-4
GRADCHECK_EPS = 1e-5


def SH2RGB(x):
    """Convert SH0 to RGB."""
    return 0.28209479177387814 * x + 0.5


def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )


def jax_trace_rays_reference(
    mean: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    density: np.ndarray,
    features: np.ndarray,
    rayo: np.ndarray,
    rayd: np.ndarray,
    tmin: float,
    tmax: float,
    num_quad: int = 2**17,  # Increased for higher precision
):
    """
    JAX-based reference implementation for ray tracing.
    Uses numerical quadrature for ground truth.
    """
    def query_ellipsoid(tdist, rayo, rayd, params):
        xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
        R = math_util.jquatToMat3(math_util.jl2_normalize(params['quat']))
        sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) /
              jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
        d = jnp.linalg.norm(sc, axis=-1, ord=2).reshape(-1, 1)
        densities = jnp.where(d < 1, params['density'], 0)
        colors = jnp.where(d < 1, params['density'] * params['features'].reshape(1, 3), 0)
        return densities, colors

    vquery_ellipsoid = jax.vmap(query_ellipsoid, in_axes=(None, None, None, {
        'mean': 0,
        'quat': 0,
        'density': 0,
        'scale': 0,
        'features': 0,
    }))

    def sum_vquery_ellipsoid(tdist, rayo, rayd, params):
        densities, colors = vquery_ellipsoid(tdist, rayo, rayd, params)
        density_sum = densities.sum(axis=0)
        colors = safe_math.safe_div(colors.sum(axis=0), density_sum)
        return density_sum.reshape(-1), colors.clip(min=0)

    tdist = jnp.linspace(tmin, tmax, num_quad + 1)
    params = {
        'mean': mean.astype(np.float64).reshape(-1, 3),
        'scale': scale.astype(np.float64).reshape(-1, 3),
        'quat': quat.astype(np.float64).reshape(-1, 4),
        'density': density.astype(np.float64).reshape(-1, 1),
        'features': SH2RGB(features).astype(np.float64).reshape(-1, 3),
    }
    rayo = rayo.astype(np.float64)
    rayd = rayd.astype(np.float64)

    out, extras = quadrature.render_quadrature(
        tdist,
        lambda t: sum_vquery_ellipsoid(t, rayo, rayd, params),
        return_extras=True,
    )
    return out, extras


def jax_trace_rays_batch(
    mean: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    density: np.ndarray,
    features: np.ndarray,
    rayos: np.ndarray,
    rayds: np.ndarray,
    tmin: float,
    tmax: float,
):
    """Batch version of JAX reference for multiple rays."""
    results = []
    for i in range(rayos.shape[0]):
        out, _ = jax_trace_rays_reference(
            mean, scale, quat, density, features,
            rayos[i:i+1], rayds[i:i+1], tmin, tmax
        )
        results.append(np.asarray(out))
    return np.concatenate(results, axis=0)


class GaussianRTForwardTest(parameterized.TestCase):
    """Test forward pass against JAX quadrature reference."""

    def _skip_if_unavailable(self):
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    @parameterized.product(
        N=[1, 2, 5, 10, 20, 50],
        density_multi=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    )
    def test_forward_vs_jax_quadrature(self, N, density_multi):
        """Test forward rendering against JAX quadrature reference."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        # Create test data - ray origin outside primitives
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=torch.float32
        ).to(device)
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 0] *= 0.5
        mean[:, 1] *= 0.5
        mean[:, 2] += 0.5  # Place primitives in front of ray

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = density_multi * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 3

        # JAX reference result
        jax_color, _ = jax_trace_rays_reference(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayo.detach().cpu().numpy(),
            rayd.detach().cpu().numpy(),
            tmin, tmax
        )
        jax_rgb = np.asarray(jax_color)[:, :3].reshape(-1)

        # GaussianRT result
        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL, rtol=FORWARD_RTOL,
            err_msg=f"Forward mismatch for N={N}, density_multi={density_multi}"
        )

    @parameterized.product(
        N=[1, 5, 10, 20],
        density_multi=[0.01, 0.1, 1.0],
    )
    def test_forward_ray_origin_inside(self, N, density_multi):
        """Test forward with ray origin potentially inside primitives."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=torch.float32
        ).to(device)
        # Place some primitives at origin (ray origin inside)
        mean = 1.2 * torch.rand(N, 3, dtype=torch.float32).to(device) - 0.2
        mean[:, 0] *= 0.5
        mean[:, 1] *= 0.5

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = density_multi * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 3

        # JAX reference
        jax_color, _ = jax_trace_rays_reference(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayo.detach().cpu().numpy(),
            rayd.detach().cpu().numpy(),
            tmin, tmax
        )
        jax_rgb = np.asarray(jax_color)[:, :3].reshape(-1)

        # GaussianRT
        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL, rtol=FORWARD_RTOL,
            err_msg=f"Forward (inside) mismatch for N={N}, density_multi={density_multi}"
        )

    @parameterized.product(
        num_rays=[1, 4, 16, 64],
        N=[5, 10],
    )
    def test_forward_multiple_rays(self, num_rays, N):
        """Test forward with multiple rays in different directions."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        # Generate random ray directions on hemisphere
        theta = np.random.rand(num_rays) * np.pi / 4  # Limit to forward hemisphere
        phi = np.random.rand(num_rays) * 2 * np.pi
        rayds = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=-1).astype(np.float32)

        rayos = np.zeros((num_rays, 3), dtype=np.float32)

        rayo = torch.tensor(rayos, dtype=torch.float32).to(device)
        rayd = torch.tensor(rayds, dtype=torch.float32).to(device)

        scale = 0.3 * torch.rand(N, 3, dtype=torch.float32).to(device) + 0.1
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 2] += 1.0  # Place in front

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = 0.5 * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 5

        # JAX reference (batch)
        jax_rgb = jax_trace_rays_batch(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayos, rayds,
            tmin, tmax
        )[:, :3]

        # GaussianRT
        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL, rtol=FORWARD_RTOL,
            err_msg=f"Multi-ray forward mismatch for num_rays={num_rays}, N={N}"
        )

    @parameterized.parameters(
        {'scale_factor': 0.01},  # Very small primitives
        {'scale_factor': 0.05},
        {'scale_factor': 2.0},   # Large primitives
    )
    def test_forward_extreme_scales(self, scale_factor):
        """Test forward with extreme scale values."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        N = 10
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = scale_factor * torch.rand(N, 3, dtype=torch.float32).to(device) + scale_factor * 0.1
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 2] += max(0.5, scale_factor)

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = 0.5 * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, max(5, scale_factor * 5)

        jax_color, _ = jax_trace_rays_reference(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayo.detach().cpu().numpy(),
            rayd.detach().cpu().numpy(),
            tmin, tmax
        )
        jax_rgb = np.asarray(jax_color)[:, :3].reshape(-1)

        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL * 2, rtol=FORWARD_RTOL * 2,  # Slightly relaxed for extreme cases
            err_msg=f"Extreme scale mismatch for scale_factor={scale_factor}"
        )

    def test_forward_overlapping_primitives(self):
        """Test forward with heavily overlapping primitives."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        N = 20
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        # All primitives centered at same location
        scale = 0.3 * torch.rand(N, 3, dtype=torch.float32).to(device) + 0.2
        mean = torch.zeros(N, 3, dtype=torch.float32).to(device)
        mean[:, 2] = 1.0  # All at z=1

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = 0.1 * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 3

        jax_color, _ = jax_trace_rays_reference(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayo.detach().cpu().numpy(),
            rayd.detach().cpu().numpy(),
            tmin, tmax
        )
        jax_rgb = np.asarray(jax_color)[:, :3].reshape(-1)

        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL, rtol=FORWARD_RTOL,
            err_msg="Overlapping primitives forward mismatch"
        )

    def test_forward_anisotropic_primitives(self):
        """Test forward with highly anisotropic (elongated) primitives."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        N = 10
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        # Highly elongated ellipsoids
        scale = torch.zeros(N, 3, dtype=torch.float32).to(device)
        scale[:, 0] = 0.05 + 0.05 * torch.rand(N)  # Thin
        scale[:, 1] = 0.05 + 0.05 * torch.rand(N)  # Thin
        scale[:, 2] = 0.5 + 0.5 * torch.rand(N)    # Long

        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 0] *= 0.3
        mean[:, 1] *= 0.3
        mean[:, 2] += 0.5

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = 0.5 * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 3

        jax_color, _ = jax_trace_rays_reference(
            mean.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            quat.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            rayo.detach().cpu().numpy(),
            rayd.detach().cpu().numpy(),
            tmin, tmax
        )
        jax_rgb = np.asarray(jax_color)[:, :3].reshape(-1)

        gaussianrt_color = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_color[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, jax_rgb,
            atol=FORWARD_ATOL, rtol=FORWARD_RTOL,
            err_msg="Anisotropic primitives forward mismatch"
        )

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic across multiple runs."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        N = 20
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.3 * torch.rand(N, 3, dtype=torch.float32).to(device)
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 2] += 0.5

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = 0.5 * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        results = []
        for _ in range(5):
            result = gaussianrt_trace_rays(
                mean, scale, quat, density.squeeze(-1), features,
                rayo, rayd,
                tmin=0, tmax=3, max_iters=500
            )
            results.append(result[:, :3].detach().cpu().numpy())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Non-deterministic result at iteration {i}"
            )


class GaussianRTGradientTest(parameterized.TestCase):
    """Test backward pass using gradient checking."""

    def _skip_if_unavailable(self):
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def _create_test_inputs(self, N, density_multi=1.0, seed=42):
        """Create test inputs with requires_grad=True for gradient checking."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        mean = torch.rand(N, 3, dtype=torch.float64, device=device, requires_grad=True)
        mean.data[:, 2] += 0.5  # Place in front of ray

        scale = (0.3 * torch.rand(N, 3, dtype=torch.float64, device=device) + 0.1).requires_grad_(True)

        quat_raw = 2 * torch.rand(N, 4, dtype=torch.float64, device=device) - 1
        quat = (quat_raw / torch.norm(quat_raw, dim=-1, keepdim=True)).detach().requires_grad_(True)

        density = (density_multi * torch.rand(N, dtype=torch.float64, device=device) + 0.1).requires_grad_(True)
        features = torch.rand(N, 1, 3, dtype=torch.float64, device=device, requires_grad=True)

        rayo = torch.zeros(1, 3, dtype=torch.float64, device=device, requires_grad=True)
        rayd = torch.tensor([[0., 0., 1.]], dtype=torch.float64, device=device, requires_grad=True)

        return mean, scale, quat, density, features, rayo, rayd

    @parameterized.product(
        N=[1, 3, 5, 10],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_mean(self, N, density_multi):
        """Gradient check for mean parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(mean_input):
            result = gaussianrt_trace_rays(
                mean_input.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (mean,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5, 10],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_scale(self, N, density_multi):
        """Gradient check for scale parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(scale_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale_input.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (scale,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5, 10],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_quat(self, N, density_multi):
        """Gradient check for quaternion parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(quat_input):
            # Normalize quaternion inside function
            quat_normalized = quat_input / torch.norm(quat_input, dim=-1, keepdim=True)
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat_normalized.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (quat,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5, 10],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_density(self, N, density_multi):
        """Gradient check for density parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(density_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density_input.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (density,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5, 10],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_features(self, N, density_multi):
        """Gradient check for features (SH coefficients) parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(features_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features_input.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (features,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_ray_origin(self, N, density_multi):
        """Gradient check for ray origin parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(rayo_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo_input.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (rayo,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.product(
        N=[1, 3, 5],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_gradcheck_ray_direction(self, N, density_multi):
        """Gradient check for ray direction parameter."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, density_multi)

        def func(rayd_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd_input.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (rayd,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
        {'N': 5},
    )
    def test_gradcheck_all_params(self, N):
        """Gradient check for all parameters simultaneously."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(mean_in, scale_in, density_in, features_in):
            result = gaussianrt_trace_rays(
                mean_in.float(), scale_in.float(), quat.float(),
                density_in.float(), features_in.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (mean, scale, density, features),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_per_channel_loss(self, N):
        """Gradient check with per-channel loss instead of sum."""
        self._skip_if_unavailable()

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        for channel in range(3):
            def func(mean_input, ch=channel):
                result = gaussianrt_trace_rays(
                    mean_input.float(), scale.float(), quat.float(),
                    density.float(), features.float(),
                    rayo.float(), rayd.float(),
                    tmin=0.0, tmax=3.0, max_iters=200
                )
                return result[:, ch].sum()

            torch.autograd.gradcheck(
                func, (mean,),
                eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
                raise_exception=True
            )

    def test_gradcheck_multiple_rays(self):
        """Gradient check with multiple rays."""
        self._skip_if_unavailable()

        N = 5
        num_rays = 4

        torch.manual_seed(42)
        mean = torch.rand(N, 3, dtype=torch.float64, device=device, requires_grad=True)
        mean.data[:, 2] += 0.5

        scale = (0.3 * torch.rand(N, 3, dtype=torch.float64, device=device) + 0.1).requires_grad_(True)
        quat_raw = 2 * torch.rand(N, 4, dtype=torch.float64, device=device) - 1
        quat = (quat_raw / torch.norm(quat_raw, dim=-1, keepdim=True)).detach().requires_grad_(True)
        density = (0.5 * torch.rand(N, dtype=torch.float64, device=device) + 0.1).requires_grad_(True)
        features = torch.rand(N, 1, 3, dtype=torch.float64, device=device, requires_grad=True)

        rayo = torch.zeros(num_rays, 3, dtype=torch.float64, device=device, requires_grad=True)
        rayd = torch.randn(num_rays, 3, dtype=torch.float64, device=device)
        rayd[:, 2] = torch.abs(rayd[:, 2]) + 0.5  # Ensure forward direction
        rayd = (rayd / torch.norm(rayd, dim=-1, keepdim=True)).detach().requires_grad_(True)

        def func(mean_input):
            result = gaussianrt_trace_rays(
                mean_input.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (mean,),
            eps=GRADCHECK_EPS, atol=GRAD_ATOL, rtol=GRAD_RTOL,
            raise_exception=True
        )

    def test_gradient_magnitude_reasonable(self):
        """Test that gradient magnitudes are reasonable (not exploding/vanishing)."""
        self._skip_if_unavailable()

        N = 10
        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        # Convert to float32 for actual computation
        mean_f = mean.float().detach().requires_grad_(True)
        scale_f = scale.float().detach().requires_grad_(True)
        quat_f = quat.float().detach().requires_grad_(True)
        density_f = density.float().detach().requires_grad_(True)
        features_f = features.float().detach().requires_grad_(True)

        result = gaussianrt_trace_rays(
            mean_f, scale_f, quat_f, density_f, features_f,
            rayo.float(), rayd.float(),
            tmin=0.0, tmax=3.0, max_iters=200
        )
        loss = result[:, :3].sum()
        loss.backward()

        # Check gradients exist and are finite
        for name, param in [('mean', mean_f), ('scale', scale_f), ('quat', quat_f),
                            ('density', density_f), ('features', features_f)]:
            self.assertIsNotNone(param.grad, f"{name} gradient is None")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} gradient has non-finite values")

            # Check gradient magnitude is reasonable (not exploding)
            grad_norm = param.grad.norm().item()
            self.assertLess(grad_norm, 1e6, f"{name} gradient norm too large: {grad_norm}")

    def test_gradient_consistency_across_seeds(self):
        """Test gradient computation is consistent across different random seeds."""
        self._skip_if_unavailable()

        N = 5
        gradients = []

        for seed in [42, 123, 456]:
            mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N, seed=seed)

            mean_f = mean.float().detach().requires_grad_(True)

            result = gaussianrt_trace_rays(
                mean_f, scale.float(), quat.float(), density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=200
            )
            loss = result[:, :3].sum()
            loss.backward()

            gradients.append(mean_f.grad.clone())

        # Gradients should be different for different inputs
        for i in range(1, len(gradients)):
            self.assertFalse(
                torch.allclose(gradients[0], gradients[i]),
                "Gradients should differ for different random inputs"
            )


class GaussianRTBackwardCompareJAXTest(parameterized.TestCase):
    """Compare GaussianRT backward pass with JAX autodiff."""

    def _skip_if_unavailable(self):
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    @parameterized.product(
        N=[1, 3, 5],
        density_multi=[0.1, 0.5, 1.0],
    )
    def test_backward_vs_jax_autodiff(self, N, density_multi):
        """Compare backward gradients with JAX automatic differentiation."""
        self._skip_if_unavailable()

        torch.manual_seed(42)
        np.random.seed(42)

        # Create inputs
        rayo_np = np.array([[0, 0, 0]], dtype=np.float64)
        rayd_np = np.array([[0, 0, 1]], dtype=np.float64)

        scale_np = (0.3 * np.random.rand(N, 3) + 0.1).astype(np.float64)
        mean_np = np.random.rand(N, 3).astype(np.float64)
        mean_np[:, 2] += 0.5

        quat_np = (2 * np.random.rand(N, 4) - 1).astype(np.float64)
        quat_np = quat_np / np.linalg.norm(quat_np, axis=-1, keepdims=True)
        density_np = (density_multi * np.random.rand(N, 1) + 0.1).astype(np.float64)
        features_np = np.random.rand(N, 1, 3).astype(np.float64)

        tmin, tmax = 0.0, 3.0

        # JAX gradient computation
        def jax_loss(mean_jax):
            out, _ = jax_trace_rays_reference(
                mean_jax, scale_np, quat_np, density_np, features_np,
                rayo_np, rayd_np, tmin, tmax
            )
            return jnp.sum(out[:, :3])

        jax_grad = jax.grad(jax_loss)(mean_np)

        # PyTorch/GaussianRT gradient computation
        mean_th = torch.tensor(mean_np, dtype=torch.float32, device=device, requires_grad=True)
        scale_th = torch.tensor(scale_np, dtype=torch.float32, device=device)
        quat_th = torch.tensor(quat_np, dtype=torch.float32, device=device)
        density_th = torch.tensor(density_np.squeeze(-1), dtype=torch.float32, device=device)
        features_th = torch.tensor(features_np, dtype=torch.float32, device=device)
        rayo_th = torch.tensor(rayo_np, dtype=torch.float32, device=device)
        rayd_th = torch.tensor(rayd_np, dtype=torch.float32, device=device)

        result = gaussianrt_trace_rays(
            mean_th, scale_th, quat_th, density_th, features_th,
            rayo_th, rayd_th,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        loss = result[:, :3].sum()
        loss.backward()

        gaussianrt_grad = mean_th.grad.cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_grad, np.asarray(jax_grad),
            atol=1e-3, rtol=1e-3,
            err_msg=f"Backward vs JAX mismatch for N={N}, density_multi={density_multi}"
        )


if __name__ == "__main__":
    absltest.main()
