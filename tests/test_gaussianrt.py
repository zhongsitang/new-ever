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
    num_quad: int = 2**16,
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


class GaussianRTForwardTest(parameterized.TestCase):
    """Test forward pass against JAX quadrature reference."""

    @parameterized.product(
        N=[1, 5, 10, 20],
        density_multi=[0.01, 0.1, 1.0],
    )
    def test_forward_vs_jax_quadrature(self, N, density_multi):
        """Test forward rendering against JAX quadrature reference."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

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
        # Use simple features (SH degree 0: single coefficient per channel)
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
            atol=1e-3, rtol=1e-3,
            err_msg=f"Forward mismatch for N={N}, density_multi={density_multi}"
        )

    @parameterized.product(
        N=[1, 5, 10],
        density_multi=[0.1, 1.0],
    )
    def test_forward_ray_origin_inside(self, N, density_multi):
        """Test forward with ray origin potentially inside primitives."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

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
            atol=1e-3, rtol=1e-3,
            err_msg=f"Forward (inside) mismatch for N={N}, density_multi={density_multi}"
        )


class GaussianRTGradientTest(parameterized.TestCase):
    """Test backward pass using gradient checking."""

    def _create_test_inputs(self, N, density_multi=1.0):
        """Create test inputs with requires_grad=True for gradient checking."""
        torch.manual_seed(42)
        np.random.seed(42)

        mean = torch.rand(N, 3, dtype=torch.float64, device=device, requires_grad=True)
        mean.data[:, 2] += 0.5  # Place in front of ray

        scale = 0.5 * torch.rand(N, 3, dtype=torch.float64, device=device, requires_grad=True)

        quat_raw = 2 * torch.rand(N, 4, dtype=torch.float64, device=device) - 1
        quat = (quat_raw / torch.norm(quat_raw, dim=-1, keepdim=True)).detach().requires_grad_(True)

        density = (density_multi * torch.rand(N, dtype=torch.float64, device=device)).requires_grad_(True)
        features = torch.rand(N, 1, 3, dtype=torch.float64, device=device, requires_grad=True)

        rayo = torch.zeros(1, 3, dtype=torch.float64, device=device, requires_grad=True)
        rayd = torch.tensor([[0., 0., 1.]], dtype=torch.float64, device=device, requires_grad=True)

        return mean, scale, quat, density, features, rayo, rayd

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
        {'N': 5},
    )
    def test_gradcheck_mean(self, N):
        """Gradient check for mean parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(mean_input):
            result = gaussianrt_trace_rays(
                mean_input.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            # Return sum of RGB channels for scalar output
            return result[:, :3].sum()

        # Use numerical gradient check
        torch.autograd.gradcheck(
            func, (mean,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_scale(self, N):
        """Gradient check for scale parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(scale_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale_input.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (scale,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_density(self, N):
        """Gradient check for density parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(density_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density_input.float(), features.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (density,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_features(self, N):
        """Gradient check for features (SH coefficients) parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(features_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features_input.float(),
                rayo.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (features,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_ray_origin(self, N):
        """Gradient check for ray origin parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(rayo_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo_input.float(), rayd.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (rayo,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )

    @parameterized.parameters(
        {'N': 1},
        {'N': 3},
    )
    def test_gradcheck_ray_direction(self, N):
        """Gradient check for ray direction parameter."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mean, scale, quat, density, features, rayo, rayd = self._create_test_inputs(N)

        def func(rayd_input):
            result = gaussianrt_trace_rays(
                mean.float(), scale.float(), quat.float(),
                density.float(), features.float(),
                rayo.float(), rayd_input.float(),
                tmin=0.0, tmax=3.0, max_iters=100
            )
            return result[:, :3].sum()

        torch.autograd.gradcheck(
            func, (rayd,),
            eps=1e-4, atol=1e-3, rtol=1e-3,
            raise_exception=True
        )


class GaussianRTCompareOriginalTest(parameterized.TestCase):
    """Compare GaussianRT with original splinetracer implementation."""

    @parameterized.product(
        N=[1, 5, 10, 20],
        density_multi=[0.01, 0.1, 1.0],
    )
    def test_compare_with_original_splinetracer(self, N, density_multi):
        """Compare GaussianRT output with original fast_ellipsoid_splinetracer."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Try to import original splinetracer
        try:
            from splinetracers import fast_ellipsoid_splinetracer as original
        except ImportError:
            self.skipTest("Original splinetracer not available")

        torch.manual_seed(42)
        np.random.seed(42)

        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=torch.float32
        ).to(device)
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 0] *= 0.5
        mean[:, 1] *= 0.5
        mean[:, 2] += 0.5

        quat = l2_normalize_th(2 * torch.rand(N, 4, dtype=torch.float32).to(device) - 1)
        density = density_multi * torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin, tmax = 0, 3

        # Original implementation
        original_result = original.trace_rays(
            mean, scale, quat, density, features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500, return_extras=False
        )
        original_rgb = original_result[:, :3].reshape(-1).detach().cpu().numpy()

        # GaussianRT implementation
        gaussianrt_result = gaussianrt_trace_rays(
            mean, scale, quat, density.squeeze(-1), features,
            rayo, rayd,
            tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_rgb = gaussianrt_result[:, :3].reshape(-1).detach().cpu().numpy()

        np.testing.assert_allclose(
            gaussianrt_rgb, original_rgb,
            atol=1e-4, rtol=1e-4,
            err_msg=f"Mismatch with original for N={N}, density_multi={density_multi}"
        )


class GaussianRTBackwardCompareTest(parameterized.TestCase):
    """Compare GaussianRT backward pass with original implementation."""

    @parameterized.product(
        N=[1, 5],
        density_multi=[0.1, 1.0],
    )
    def test_backward_compare_with_original(self, N, density_multi):
        """Compare backward gradients with original splinetracer."""
        if not GAUSSIANRT_AVAILABLE:
            self.skipTest(f"GaussianRT not available: {GAUSSIANRT_IMPORT_ERROR}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            from splinetracers import fast_ellipsoid_splinetracer as original
        except ImportError:
            self.skipTest("Original splinetracer not available")

        torch.manual_seed(42)
        np.random.seed(42)

        # Create inputs with gradients
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)

        scale = 0.5 * torch.rand(N, 3, dtype=torch.float32, device=device)
        mean = torch.rand(N, 3, dtype=torch.float32, device=device)
        mean[:, 2] += 0.5

        quat_raw = 2 * torch.rand(N, 4, dtype=torch.float32, device=device) - 1
        quat = quat_raw / torch.norm(quat_raw, dim=-1, keepdim=True)
        density = density_multi * torch.rand(N, 1, dtype=torch.float32, device=device)
        features = torch.rand(N, 1, 3, dtype=torch.float32, device=device)

        tmin, tmax = 0, 3

        # Clone for independent computation
        mean_orig = mean.clone().requires_grad_(True)
        scale_orig = scale.clone().requires_grad_(True)
        quat_orig = quat.clone().requires_grad_(True)
        density_orig = density.clone().requires_grad_(True)
        features_orig = features.clone().requires_grad_(True)

        mean_new = mean.clone().requires_grad_(True)
        scale_new = scale.clone().requires_grad_(True)
        quat_new = quat.clone().requires_grad_(True)
        density_new = density.squeeze(-1).clone().requires_grad_(True)
        features_new = features.clone().requires_grad_(True)

        # Original backward
        original_result = original.trace_rays(
            mean_orig, scale_orig, quat_orig, density_orig, features_orig,
            rayo, rayd, tmin=tmin, tmax=tmax, max_iters=500
        )
        original_loss = original_result[:, :3].sum()
        original_loss.backward()

        # GaussianRT backward
        gaussianrt_result = gaussianrt_trace_rays(
            mean_new, scale_new, quat_new, density_new, features_new,
            rayo, rayd, tmin=tmin, tmax=tmax, max_iters=500
        )
        gaussianrt_loss = gaussianrt_result[:, :3].sum()
        gaussianrt_loss.backward()

        # Compare gradients
        np.testing.assert_allclose(
            mean_new.grad.cpu().numpy(),
            mean_orig.grad.cpu().numpy(),
            atol=1e-3, rtol=1e-3,
            err_msg="Mean gradient mismatch"
        )

        np.testing.assert_allclose(
            scale_new.grad.cpu().numpy(),
            scale_orig.grad.cpu().numpy(),
            atol=1e-3, rtol=1e-3,
            err_msg="Scale gradient mismatch"
        )

        np.testing.assert_allclose(
            density_new.grad.cpu().numpy(),
            density_orig.grad.squeeze(-1).cpu().numpy(),
            atol=1e-3, rtol=1e-3,
            err_msg="Density gradient mismatch"
        )

        np.testing.assert_allclose(
            features_new.grad.cpu().numpy(),
            features_orig.grad.cpu().numpy(),
            atol=1e-3, rtol=1e-3,
            err_msg="Features gradient mismatch"
        )


if __name__ == "__main__":
    absltest.main()
