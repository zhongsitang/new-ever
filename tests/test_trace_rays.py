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

"""Tests for primtracer.trace_rays."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

import primtracer
from tests.test_utils import (
    create_random_test_scene,
    directional_gradcheck,
    eval_sh_torch,
    get_device,
    export_scene_obj,
    trace_rays_reference,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*double precision.*:UserWarning")

DEVICE = get_device()
SH_C0 = 0.28209479177387814
OUTPUT_DIR = Path(__file__).parent / "output"


def _trace(scene, max_hits=500, tmin=None, tmax=None, min_logT=None,
           return_extras=False):
    """Wrapper for primtracer.trace_rays with scene dict."""
    tmin_val = scene["tmin"] if tmin is None else tmin
    tmax_val = scene["tmax"] if tmax is None else tmax
    min_logT_val = math.log(2e-9) if min_logT is None else min_logT
    return primtracer.trace_rays(
        scene["mean"],
        scene["scale"],
        scene["quat"],
        scene["density"],
        scene["features"],
        scene["rayo"],
        scene["rayd"],
        tmin=tmin_val,
        tmax=tmax_val,
        min_logT=min_logT_val,
        max_hits=max_hits,
        return_extras=return_extras,
    )


def _trace_reference(scene):
    """Reference implementation using PyTorch GPU quadrature."""
    return trace_rays_reference(
        scene["mean"],
        scene["scale"],
        scene["quat"],
        scene["density"],
        scene["features"],
        scene["rayo"],
        scene["rayd"],
        tmin=scene["tmin"],
        tmax=scene["tmax"],
        num_samples=2**16,
        ray_batch_size=64,
    )


def _save_render(name, color, depth, resolution, color_ref=None, depth_ref=None):
    """Save rendered images (horizontally concatenated: RGB | Alpha | Depth)."""
    import numpy as np
    from PIL import Image
    OUTPUT_DIR.mkdir(exist_ok=True)

    def to_images(c, d):
        rgb = c[:, :3].reshape(resolution, resolution, 3).detach().cpu().numpy()
        alpha = c[:, 3:4].reshape(resolution, resolution, 1).detach().cpu().numpy()
        dep = d.reshape(resolution, resolution).detach().cpu().numpy()
        rgb_img = (rgb.clip(0, 1) * 255).astype("uint8")
        alpha_img = np.repeat((alpha.clip(0, 1) * 255).astype("uint8"), 3, axis=2)
        d_norm = dep / (dep.max() + 1e-6)
        depth_img = np.repeat((d_norm * 255).astype("uint8")[:, :, None], 3, axis=2)
        return np.concatenate([rgb_img, alpha_img, depth_img], axis=1)

    row = to_images(color, depth)
    if color_ref is not None and depth_ref is not None:
        row_ref = to_images(color_ref, depth_ref)
        combined = np.concatenate([row_ref, row], axis=0)  # ref on top, actual below
    else:
        combined = row
    Image.fromarray(combined).save(OUTPUT_DIR / f"{name}.png")


def _single_ellipsoid_scene(
    mean=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    rayo=(0.0, 0.0, -2.0),
    rayd=(0.0, 0.0, 1.0),
    density=1.0,
    color=(0.2, 0.7, 0.4),
    device=DEVICE,
):
    """Create a single ellipsoid scene for analytic tests."""
    mean_t = torch.tensor([mean], device=device)
    scale_t = torch.tensor([scale], device=device)
    quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    density_t = torch.tensor([density], device=device)

    color_t = torch.tensor(color, device=device)
    c0 = (color_t - 0.5) / SH_C0
    features_t = c0.view(1, 1, 3).contiguous()

    rayo_t = torch.tensor([rayo], device=device)
    rayd_t = torch.tensor([rayd], device=device)

    return {
        "mean": mean_t,
        "scale": scale_t,
        "quat": quat_t,
        "density": density_t,
        "features": features_t,
        "rayo": rayo_t,
        "rayd": rayd_t,
        "tmin": 0.0,
        "tmax": 10.0,
    }


class TestAnalytic:
    """Analytic tests with known expected values."""

    def test_single_ellipsoid_analytic(self):
        """Single ellipsoid should match analytic formula."""
        scene = _single_ellipsoid_scene(
            scale=(0.5, 0.5, 1.0), density=1.7, device=DEVICE)
        color, depth = _trace(scene, max_hits=8)

        t_entry = 1.0
        length = 2.0
        sigma = 1.7
        tau = sigma * length
        alpha = 1.0 - math.exp(-tau)

        expected_rgb = scene["features"][0, 0] * SH_C0 + 0.5
        expected_rgb = expected_rgb * alpha
        expected_depth = (
            t_entry + (1.0 / sigma)
            - length * math.exp(-tau) / (1.0 - math.exp(-tau))
        )

        torch.testing.assert_close(
            color[0, :3], expected_rgb, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            color[0, 3], torch.tensor(alpha, device=DEVICE), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            depth[0], torch.tensor(expected_depth, device=DEVICE),
            atol=2e-5, rtol=2e-5)

    def test_sh_degree_affects_color(self):
        """Higher SH degree should produce different colors."""
        device = DEVICE
        mean = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        scale = torch.tensor([[0.5, 0.5, 1.0]], device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        density = torch.tensor([1.3], device=device)
        rayo = torch.tensor([[0.0, 0.0, -2.0]], device=device)
        rayd = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        color_const = torch.tensor([0.1, 0.6, 0.3], device=device)
        c0 = (color_const - 0.5) / SH_C0
        features_deg0 = c0.view(1, 1, 3).contiguous()

        scene0 = {
            "mean": mean, "scale": scale, "quat": quat, "density": density,
            "features": features_deg0, "rayo": rayo, "rayd": rayd,
            "tmin": torch.zeros(1, device=device),
            "tmax": torch.full((1,), 10.0, device=device),
        }
        color0, _ = _trace(scene0, max_hits=8)

        features_deg1 = torch.zeros(1, 4, 3, device=device)
        features_deg1[:, 0, :] = c0
        features_deg1[:, 2, :] = torch.tensor([0.2, 0.0, 0.0], device=device)

        scene1 = dict(scene0)
        scene1["features"] = features_deg1
        color1, _ = _trace(scene1, max_hits=8)

        expected_color = eval_sh_torch(mean, features_deg1, rayo,
                                       sh_degree=1, apply_clip=True)
        sigma = float(density.item())
        tau = sigma * 2.0
        alpha = 1.0 - math.exp(-tau)
        expected_rgb = expected_color[0] * alpha

        torch.testing.assert_close(color1[0, :3], expected_rgb, atol=2e-5, rtol=2e-5)
        assert not torch.allclose(color0, color1)

    def test_no_hit_returns_zero(self):
        """Zero density should produce zero alpha and depth."""
        scene = _single_ellipsoid_scene(density=0.0, device=DEVICE)
        color, depth = _trace(scene, max_hits=4)
        torch.testing.assert_close(
            color[0, 3], torch.tensor(0.0, device=DEVICE), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            depth[0], torch.tensor(0.0, device=DEVICE), atol=1e-6, rtol=0)


class TestCoreCorrectness:
    """Tests for correctness against reference implementation."""

    RESOLUTION = 32
    SAVE_IMAGES = True

    def _create_scene(self):
        """Create a test scene with default parameters."""
        return create_random_test_scene(
            num_primitives=32,
            cam_resolution=self.RESOLUTION,
            density_range=(0.1, 1.0),
            seed=1,
            device=DEVICE,
        )

    def test_random_scene_reference(self):
        """Random scene should match JAX reference."""
        scene = self._create_scene()
        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        if self.SAVE_IMAGES:
            _save_render("random_scene", c, d, self.RESOLUTION, c_ref, d_ref)
        torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d, d_ref, atol=1e-3, rtol=1e-3)

    def test_per_ray_tmax(self):
        """Per-ray tmax should work correctly."""
        scene = self._create_scene()
        # Override with random per-ray tmax
        num_rays = scene["rayo"].shape[0]
        scene["tmax"] = 2.0 + 3.0 * torch.rand(num_rays, device=DEVICE)

        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        if self.SAVE_IMAGES:
            _save_render("per_ray_tmax", c, d, self.RESOLUTION, c_ref, d_ref)
        torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d, d_ref, atol=1e-3, rtol=1e-3)

    def test_ray_inside_ellipsoid(self):
        """Ray originating inside ellipsoid should match reference."""
        scene = self._create_scene()
        # Move first two ellipsoids to camera to ensure inside-ellipsoid cases
        cam_pos = scene["rayo"][0]  # Camera position (all rays share same origin)
        scene["mean"][0] = cam_pos.clone()
        scene["mean"][1] = cam_pos + torch.tensor([0.1, 0.1, 0.2], device=DEVICE)
        # Make them more transparent so we can see through
        scene["density"][0] = torch.tensor([1], device=DEVICE)
        scene["density"][1] = torch.tensor([1], device=DEVICE)
        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        if self.SAVE_IMAGES:
            _save_render("ray_inside_ellipsoid", c, d, self.RESOLUTION, c_ref, d_ref)
        torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d, d_ref, atol=1e-3, rtol=1e-3)


class TestRobustness:
    """Tests for robustness and determinism."""

    def test_extreme_values_finite(self):
        """Extreme values should produce finite outputs."""
        scene = create_random_test_scene(
            num_primitives=16,
            cam_resolution=32,
            density_range=(10.0, 100.0),
            scale_range=(1e-3, 5.0),
            seed=31,
            device=DEVICE,
        )
        color, depth = _trace(scene)

        assert torch.isfinite(color).all()
        assert torch.isfinite(depth).all()
        assert torch.all(color[:, 3] >= -1e-5)
        assert torch.all(color[:, 3] <= 1.0 + 1e-5)

    def test_determinism(self):
        """Repeated calls should produce identical results."""
        scene = create_random_test_scene(
            num_primitives=16, cam_resolution=32, seed=37, device=DEVICE)

        color1, depth1 = _trace(scene, max_hits=32)
        color2, depth2 = _trace(scene, max_hits=32)

        torch.testing.assert_close(color1, color2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth1, depth2, atol=1e-6, rtol=1e-6)

    def test_batch_split_consistency(self):
        """Splitting rays should produce same results as full batch."""
        scene = create_random_test_scene(
            num_primitives=24, cam_resolution=32, seed=41, device=DEVICE)
        color_full, depth_full = _trace(scene)

        rayo = scene["rayo"]
        rayd = scene["rayd"]
        mid = rayo.shape[0] // 2

        scene_a = dict(scene)
        scene_a["rayo"] = rayo[:mid]
        scene_a["rayd"] = rayd[:mid]

        scene_b = dict(scene)
        scene_b["rayo"] = rayo[mid:]
        scene_b["rayd"] = rayd[mid:]

        color_a, depth_a = _trace(scene_a)
        color_b, depth_b = _trace(scene_b)

        color_cat = torch.cat([color_a, color_b], dim=0)
        depth_cat = torch.cat([depth_a, depth_b], dim=0)

        torch.testing.assert_close(color_cat, color_full, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth_cat, depth_full, atol=1e-6, rtol=1e-6)


class TestGradients:
    """Gradient correctness tests."""

    def _create_scene(self):
        """Create a test scene for gradient checking."""
        return create_random_test_scene(
            num_primitives=32,
            cam_resolution=32,
            seed=42,
            device=DEVICE,
        )

    def _make_params(self, scene):
        """Create parameter dict for gradient checking."""
        return {
            "mean": torch.nn.Parameter(scene["mean"]),
            "scale": torch.nn.Parameter(scene["scale"]),
            "quat": torch.nn.Parameter(scene["quat"]),
            "density": torch.nn.Parameter(scene["density"]),
            "features": torch.nn.Parameter(scene["features"]),
        }

    def test_grad_color(self):
        """Color gradients should be correct."""
        scene = self._create_scene()

        def loss_fn(mean, scale, quat, density, features):
            color, _ = primtracer.trace_rays(
                mean, scale, quat, density, features,
                scene["rayo"], scene["rayd"],
                tmin=scene["tmin"], tmax=scene["tmax"],
            )
            return color.sum()

        directional_gradcheck(loss_fn, self._make_params(scene))


class TestPerformance:
    """Performance benchmark tests."""

    RESOLUTION = 1024
    SAVE_IMAGES = True
    NUM_ITERS = 5
    MAX_HITS = 128

    def _create_scene(self):
        """Create a scene for performance benchmarks."""
        return create_random_test_scene(
            num_primitives=4096,
            cam_resolution=self.RESOLUTION,
            scale_range=(0.001, 0.1),
            density_range=(0.1, 5.0),
            seed=42,
            device=DEVICE,
        )

    def test_perf(self):
        """Forward and backward pass performance."""
        max_fwd_ms = float(os.getenv("PERF_FWD_MAX_MS", "200.0"))
        max_bwd_ms = float(os.getenv("PERF_BWD_MAX_MS", "300.0"))
        scene = self._create_scene()
        grad_keys = ["mean", "scale", "quat", "density", "features"]
        grad_scene = dict(scene)

        def clone_params():
            for key in grad_keys:
                grad_scene[key] = scene[key].clone().requires_grad_(True)

        # Warmup
        clone_params()
        color, depth = _trace(grad_scene, max_hits=self.MAX_HITS)
        (color.sum() + depth.sum()).backward()
        torch.cuda.synchronize()

        # Benchmark
        total_fwd_ms = 0.0
        total_bwd_ms = 0.0
        for _ in range(self.NUM_ITERS):
            clone_params()
            torch.cuda.synchronize()

            start = time.perf_counter()
            color, depth = _trace(grad_scene, max_hits=self.MAX_HITS)
            torch.cuda.synchronize()
            total_fwd_ms += (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            (color.sum() + depth.sum()).backward()
            torch.cuda.synchronize()
            total_bwd_ms += (time.perf_counter() - start) * 1000.0

        fwd_ms = total_fwd_ms / self.NUM_ITERS
        bwd_ms = total_bwd_ms / self.NUM_ITERS

        print(f"perf_fwd={fwd_ms:.2f}ms, perf_bwd={bwd_ms:.2f}ms")
        assert fwd_ms <= max_fwd_ms, \
            f"perf_fwd={fwd_ms:.2f}ms > max={max_fwd_ms:.2f}ms"
        assert bwd_ms <= max_bwd_ms, \
            f"perf_bwd={bwd_ms:.2f}ms > max={max_bwd_ms:.2f}ms"

        if self.SAVE_IMAGES:
            _save_render("performance", color, depth, self.RESOLUTION)
