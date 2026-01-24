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

"""Tests for trace_rays function."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

import primtracer
from tests.test_utils import (
    get_device,
    to_tensor,
    l2_normalize,
    create_primitives,
    create_rays,
    trace_rays_reference,
)

# Suppress gradcheck warning about non-double precision inputs
pytestmark = pytest.mark.filterwarnings('ignore:.*double precision.*:UserWarning')

DEVICE = get_device()


class TestTraceRaysCorrectness:
    """Correctness tests comparing against reference implementation."""

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    @pytest.mark.parametrize('density_scale', [0.01, 0.1, 1.0])
    def test_ray_outside_primitives(self, n, density_scale):
        """Ray origin outside all primitives."""
        p = create_primitives(n, density_scale, device=DEVICE)
        p['mean'][:, 2] += 0.5  # Move in front of ray
        p['mean'][:, :2] *= 0.5
        rayo, rayd = create_rays(device=DEVICE)

        c_ref, d_ref = trace_rays_reference(
            p['mean'], p['scale'], p['quat'],
            p['density'], p['features'],
            rayo, rayd,
            tmin=0, tmax=3,
        )
        c, d = primtracer.trace_rays(
            p['mean'], p['scale'], p['quat'],
            p['density'], p['features'],
            rayo, rayd,
            tmin=0, tmax=3,
        )

        c_ref_tensor = to_tensor(c_ref, DEVICE).float()
        d_ref_tensor = to_tensor(d_ref, DEVICE).float()
        torch.testing.assert_close(c, c_ref_tensor, atol=1e-4, rtol=1e-4)
        # torch.testing.assert_close(d, d_ref_tensor, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    @pytest.mark.parametrize('density_scale', [0.01, 0.1, 1.0])
    def test_ray_inside_primitives(self, n, density_scale):
        """Ray origin may be inside some primitives."""
        p = create_primitives(n, density_scale, device=DEVICE)
        p['mean'] = 1.2 * p['mean'] - 0.2
        p['mean'][:, :2] *= 0.5
        rayo, rayd = create_rays(device=DEVICE)

        c_ref, d_ref = trace_rays_reference(
            p['mean'], p['scale'], p['quat'],
            p['density'], p['features'],
            rayo, rayd,
            tmin=0, tmax=3,
        )
        c, d = primtracer.trace_rays(
            p['mean'], p['scale'], p['quat'],
            p['density'], p['features'],
            rayo, rayd,
            tmin=0, tmax=3,
        )

        c_ref_tensor = to_tensor(c_ref, DEVICE).float()
        d_ref_tensor = to_tensor(d_ref, DEVICE).float()
        torch.testing.assert_close(c, c_ref_tensor, atol=1e-4, rtol=1e-4)
        # torch.testing.assert_close(d, d_ref_tensor, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('n', [1, 5, 10, 50])
    @pytest.mark.parametrize('density_scale', [0.01, 0.1, 1.0])
    def test_per_ray_tmax(self, n, density_scale):
        """Per-ray tmax as tensor."""
        num_rays = 4
        p = create_primitives(n, density_scale, device=DEVICE)
        p['mean'][:, 2] += 0.5

        rayo = torch.rand(num_rays, 3, device=DEVICE)
        rayd = l2_normalize(torch.rand(num_rays, 3, device=DEVICE) - 0.5)
        tmax = 1.0 + 2.0 * torch.rand(num_rays, device=DEVICE)

        # Reference: per-ray
        colors_ref = []
        depths_ref = []
        for i in range(num_rays):
            c, d = trace_rays_reference(
                p['mean'], p['scale'], p['quat'],
                p['density'], p['features'],
                rayo[i:i+1], rayd[i:i+1],
                tmin=0, tmax=float(tmax[i]),
            )
            colors_ref.append(to_tensor(c))
            depths_ref.append(to_tensor(d))
        color_ref = torch.concat(colors_ref).to(DEVICE).float()
        depth_ref = torch.concat(depths_ref).to(DEVICE).float()

        color, depth = primtracer.trace_rays(
            p['mean'], p['scale'], p['quat'],
            p['density'], p['features'],
            rayo, rayd,
            tmin=0, tmax=tmax,
        )

        torch.testing.assert_close(color, color_ref, atol=1e-4, rtol=1e-4)
        # torch.testing.assert_close(depth, depth_ref, atol=1e-4, rtol=1e-4)


class TestTraceRaysGradient:
    """Gradient tests using torch.autograd.gradcheck."""

    @staticmethod
    def _make_params(n, density_scale):
        p = create_primitives(n, density_scale, device=DEVICE)
        return tuple(
            torch.nn.Parameter(p[k])
            for k in ['mean', 'scale', 'quat', 'density', 'features']
        )

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    @pytest.mark.parametrize('density_scale', [0.01, 0.1, 1.0])
    def test_grad_color(self, n, density_scale):
        """Gradient check on color output."""
        mean, scale, quat, density, features = self._make_params(n, density_scale)
        rayo, rayd = create_rays(n=2, device=DEVICE)

        def loss(m, s, q, d, f):
            c, _ = primtracer.trace_rays(
                m, s, q, d, f,
                rayo, rayd,
                tmin=0.5, tmax=100,
            )
            return c.sum()

        torch.autograd.gradcheck(
            loss,
            (mean, scale, quat, density, features),
            eps=1e-3, atol=1e-3, rtol=1e-3,
        )

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    @pytest.mark.parametrize('density_scale', [0.01, 0.1, 1.0])
    def test_grad_depth(self, n, density_scale):
        """Gradient check on depth output."""
        mean, scale, quat, density, features = self._make_params(n, density_scale)
        rayo, rayd = create_rays(n=2, device=DEVICE)

        def loss(m, s, q, d, f):
            _, depth = primtracer.trace_rays(
                m, s, q, d, f,
                rayo, rayd,
                tmin=0.5, tmax=100,
            )
            return depth.sum()

        torch.autograd.gradcheck(
            loss,
            (mean, scale, quat, density, features),
            eps=1e-3, atol=1e-3, rtol=1e-3,
        )

