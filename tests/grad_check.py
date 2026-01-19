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

from absl.testing import absltest
from absl.testing import parameterized
from utils.test_utils import METHODS, SYM_METHODS, QUAD_PAIRS
import numpy as np
import torch
from icecream import ic
from utils.math_util import l2_normalize_th
import random
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

import eval_sh

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
            color = method.trace_rays(
                means, scales, quats, densities, feats, rayo, rayd,
                fixed_random, 100)
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=dtype, device=device)
            return (color * weights).sum()

        torch.autograd.gradcheck(
            l2_loss,
            (means, scales, quats, densities, feats),
            eps=1e-3,
            atol=1e-2,
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
            color = method.trace_rays(
                means, scales, quats, densities, feats, rayo, rayd,
                fixed_random, 100)
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0], dtype=dtype, device=device)
            return (color * weights).sum()

        torch.autograd.gradcheck(
            l2_loss_w_dist,
            (means, scales, quats, densities, feats),
            eps=1e-3,
            atol=1e-2,
            rtol=1e-3,
        )


if __name__ == "__main__":
    absltest.main()