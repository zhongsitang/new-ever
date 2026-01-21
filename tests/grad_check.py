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

"""Tests for gradient checking of ray tracing methods."""

import numpy as np
import torch
from absl.testing import absltest
from absl.testing import parameterized

from tests.utils.test_utils import METHODS
from tests.utils.math_util import l2_normalize_th

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

device = torch.device('cuda')


class GradCheckTest(parameterized.TestCase):

    @parameterized.product(
        method=METHODS,
        N=[1, 5, 10, 20, 40],
        density_multi=[0.01, 0.1, 1],
    )
    def test_grad_check(self, method, N, density_multi):
        torch.manual_seed(42)
        np.random.seed(42)

        dtype = torch.float32

        rayo = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=dtype).to(device)
        rayd = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=dtype).to(device)

        scales = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=dtype
        ).to(device)
        means = 2 * torch.rand(N, 3, dtype=dtype).to(device) - 1

        quats = l2_normalize_th(torch.rand(N, 4, dtype=dtype).to(device))
        quats = torch.tensor([0, 0, 0, 1],
                             dtype=dtype, device=device).reshape(1, -1).expand(N, -1).contiguous()
        densities = density_multi * torch.rand(N, 1, dtype=dtype).to(device)
        feats = torch.rand(N, 1, 3, dtype=dtype).to(device)

        means = torch.nn.Parameter(means)
        scales = torch.nn.Parameter(scales)
        quats = torch.nn.Parameter(quats)
        feats = torch.nn.Parameter(feats)
        densities = torch.nn.Parameter(densities)

        fixed_random = 0.5

        def l2_loss(means, scales, quats, densities, feats):
            color, depth, extras = method.trace_rays(
                means, scales, quats, densities, feats, rayo, rayd,
                fixed_random, 100)
            # color: [R, G, B, A], depth: scalar per ray
            return color.sum()

        torch.autograd.gradcheck(
            l2_loss,
            (means, scales, quats, densities, feats),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
        )

    @parameterized.product(
        method=METHODS,
        N=[1, 5, 10, 20, 40],
        density_multi=[0.01, 0.1, 1],
    )
    def test_grad_check_depth(self, method, N, density_multi):
        torch.manual_seed(42)
        np.random.seed(42)

        dtype = torch.float32

        rayo = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=dtype).to(device)
        rayd = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=dtype).to(device)

        scales = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=dtype
        ).to(device)
        means = 2 * torch.rand(N, 3, dtype=dtype).to(device) - 1

        quats = l2_normalize_th(torch.rand(N, 4, dtype=dtype).to(device))
        quats = torch.tensor([0, 0, 0, 1],
                             dtype=dtype, device=device).reshape(1, -1).expand(N, -1).contiguous()
        densities = density_multi * torch.rand(N, 1, dtype=dtype).to(device)
        feats = torch.rand(N, 1, 3, dtype=dtype).to(device)

        means = torch.nn.Parameter(means)
        scales = torch.nn.Parameter(scales)
        quats = torch.nn.Parameter(quats)
        feats = torch.nn.Parameter(feats)
        densities = torch.nn.Parameter(densities)

        fixed_random = 0.5

        def depth_loss(means, scales, quats, densities, feats):
            _, depth, _ = method.trace_rays(
                means, scales, quats, densities, feats, rayo, rayd,
                fixed_random, 100)
            return depth.sum()

        torch.autograd.gradcheck(
            depth_loss,
            (means, scales, quats, densities, feats),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
        )

    @parameterized.product(
        method=METHODS,
        N=[1, 5, 10, 20, 40],
        density_multi=[0.01, 0.1, 1],
    )
    def test_grad_check_distortion(self, method, N, density_multi):
        torch.manual_seed(42)
        np.random.seed(42)

        dtype = torch.float32

        rayo = torch.tensor([[0, 0, 0], [0, 0, 0.1]], dtype=dtype).to(device)
        rayd = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=dtype).to(device)

        scales = 0.5 * torch.tensor(
            np.random.rand(N, 3), dtype=dtype
        ).to(device)
        means = 2 * torch.rand(N, 3, dtype=dtype).to(device) - 1

        quats = l2_normalize_th(torch.rand(N, 4, dtype=dtype).to(device))
        quats = torch.tensor([0, 0, 0, 1],
                             dtype=dtype, device=device).reshape(1, -1).expand(N, -1).contiguous()
        densities = density_multi * torch.rand(N, 1, dtype=dtype).to(device)
        feats = torch.rand(N, 1, 3, dtype=dtype).to(device)

        means = torch.nn.Parameter(means)
        scales = torch.nn.Parameter(scales)
        quats = torch.nn.Parameter(quats)
        feats = torch.nn.Parameter(feats)
        densities = torch.nn.Parameter(densities)

        fixed_random = 0.5

        def l2_loss_w_dist(means, scales, quats, densities, feats):
            color, depth, extras = method.trace_rays(
                means, scales, quats, densities, feats, rayo, rayd,
                fixed_random, 100)
            # color: [R, G, B, A], depth: scalar per ray, distortion_loss in extras
            distortion_loss = extras['distortion_loss']
            return color.sum() + 10.0 * distortion_loss.sum()

        torch.autograd.gradcheck(
            l2_loss_w_dist,
            (means, scales, quats, densities, feats),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    absltest.main()