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

"""Tests for eval_sh (spherical harmonics) function."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

import primtracer
from tests.test_utils import get_device, l2_normalize, eval_sh_torch

# Suppress gradcheck warning about non-double precision inputs
pytestmark = pytest.mark.filterwarnings('ignore:.*double precision.*:UserWarning')

DEVICE = get_device()


class TestEvalSHCorrectness:
    """Correctness tests comparing against reference implementation."""

    @pytest.mark.parametrize('n', [1, 10, 100])
    @pytest.mark.parametrize('sh_degree', [0, 1, 2, 3])
    def test_sh_evaluation(self, n, sh_degree):
        """Test SH evaluation matches reference."""
        torch.manual_seed(42)
        n_coeffs = (sh_degree + 1) ** 2
        features = torch.rand(n, n_coeffs, 3, device=DEVICE)
        means = l2_normalize(2 * torch.rand(n, 3, device=DEVICE) - 1)
        rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32, device=DEVICE)

        rgb_ref = eval_sh_torch(means, features, rayo, sh_degree)
        rgb = primtracer.eval_sh(means, features, rayo, sh_degree)

        torch.testing.assert_close(rgb, rgb_ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize('sh_degree', [0, 1, 2, 3])
    def test_canonical_directions(self, sh_degree):
        """Test SH at canonical axis directions."""
        torch.manual_seed(42)
        n_coeffs = (sh_degree + 1) ** 2
        means = torch.tensor(
            [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]],
            dtype=torch.float32, device=DEVICE
        )
        features = torch.rand(6, n_coeffs, 3, device=DEVICE)
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=DEVICE)

        rgb_ref = eval_sh_torch(means, features, rayo, sh_degree)
        rgb = primtracer.eval_sh(means, features, rayo, sh_degree)

        torch.testing.assert_close(rgb, rgb_ref, atol=1e-5, rtol=1e-5)


class TestEvalSHGradient:
    """Gradient tests for eval_sh."""

    @pytest.mark.parametrize('n', [1, 10])
    @pytest.mark.parametrize('sh_degree', [0, 1, 2, 3])
    def test_grad_features(self, n, sh_degree):
        """Gradient check on SH features."""
        torch.manual_seed(42)
        n_coeffs = (sh_degree + 1) ** 2
        features = torch.nn.Parameter(torch.rand(n, n_coeffs, 3, device=DEVICE))
        means = l2_normalize(2 * torch.rand(n, 3, device=DEVICE) - 1)
        rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32, device=DEVICE)

        def loss(f):
            return primtracer.eval_sh(means, f, rayo, sh_degree).sum()

        torch.autograd.gradcheck(loss, (features,), eps=1e-4, atol=2e-2)

    @pytest.mark.parametrize('sh_degree', [0, 1, 2, 3])
    def test_grad_matches_reference(self, sh_degree):
        """Gradient matches reference implementation."""
        torch.manual_seed(42)
        n_coeffs = (sh_degree + 1) ** 2
        features = torch.rand(1, n_coeffs, 3, device=DEVICE)
        means = l2_normalize(2 * torch.rand(1, 3, device=DEVICE) - 1)
        rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32, device=DEVICE)

        features_param = torch.nn.Parameter(features.clone())

        for i in range(3):
            grad = torch.nn.functional.one_hot(torch.tensor([i]), 3).float().to(DEVICE)
            _, vjp_impl = torch.autograd.functional.vjp(
                lambda f: primtracer.eval_sh(means, f, rayo, sh_degree), features_param, grad
            )
            _, vjp_ref = torch.autograd.functional.vjp(
                lambda f: eval_sh_torch(means, f, rayo, sh_degree), features_param, grad
            )
            np.testing.assert_allclose(
                vjp_impl.cpu().numpy(), vjp_ref.cpu().numpy(), atol=1e-6, rtol=1e-6
            )
