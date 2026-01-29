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

"""Extended tests for primtracer.trace_rays covering preprocessing, hits, and stability."""

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
    l2_normalize,
    create_random_test_scene,
    eval_sh_torch,
    get_device,
    to_tensor,
    trace_rays_reference,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*double precision.*:UserWarning")

DEVICE = get_device()
SH_C0 = 0.28209479177387814


def _trace(scene, max_hits=64, tmin=None, tmax=None, min_logT=None, return_extras=False):
    tmin_val = scene.get("tmin", 0.0) if tmin is None else tmin
    tmax_val = scene.get("tmax", 1e7) if tmax is None else tmax
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


def _trace_reference(scene, tmin=None, tmax=None):
    colors, depths = [], []
    tmin_val = float(scene["tmin"]) if tmin is None else tmin
    tmax_val = scene["tmax"] if tmax is None else tmax

    for i in range(scene["rayo"].shape[0]):
        tmax_i = float(tmax_val[i]) if isinstance(tmax_val, torch.Tensor) else float(tmax_val)
        c, d = trace_rays_reference(
            scene["mean"],
            scene["scale"],
            scene["quat"],
            scene["density"],
            scene["features"],
            scene["rayo"][i : i + 1],
            scene["rayd"][i : i + 1],
            tmin=tmin_val if not isinstance(tmin_val, torch.Tensor) else float(tmin_val),
            tmax=tmax_i,
        )
        colors.append(to_tensor(c))
        depths.append(to_tensor(d))

    device = scene["rayo"].device
    return (
        torch.cat(colors).to(device).float(),
        torch.cat(depths).to(device).float(),
    )


def _single_ellipsoid_scene(
    mean=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    rayo=(0.0, 0.0, -2.0),
    rayd=(0.0, 0.0, 1.0),
    density=1.0,
    color=(0.2, 0.7, 0.4),
    device=DEVICE,
):
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


def _create_simple_scene(device: torch.device | None = None):
    """Return a small deterministic scene (rays + primitives) for analysis."""
    device = device or get_device()
    rayo = torch.zeros(3, 3, device=device)
    rayd = l2_normalize(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.05, 0.0, 1.0],
                [-0.05, 0.02, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    mean = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 2.0],
            [-0.05, 0.05, 3.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor(
        [
            [0.25, 0.2, 0.25],
            [0.2, 0.2, 0.2],
            [0.3, 0.2, 0.25],
        ],
        dtype=torch.float32,
        device=device,
    )
    quat = l2_normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9239, 0.0, 0.3827, 0.0],
                [0.9808, 0.0, 0.0, 0.1951],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    density = torch.tensor([[1.0], [0.7], [0.9]], dtype=torch.float32, device=device)
    features = torch.tensor(
        [
            [[0.8, 0.6, 0.4]],
            [[0.6, 0.7, 0.9]],
            [[0.3, 0.4, 0.5]],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {
        "mean": mean,
        "scale": scale,
        "quat": quat,
        "density": density,
        "features": features,
        "rayo": rayo,
        "rayd": rayd,
        "tmin": 0.0,
        "tmax": 4.0,
    }


class TestCoreCorrectness:
    def test_simple_scene_reference(self):
        scene = _create_simple_scene(device=DEVICE)
        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        torch.testing.assert_close(c, c_ref, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(d, d_ref, atol=5e-5, rtol=5e-5)

    def test_random_scene_reference_outside(self):
        scene = create_random_test_scene(
            n=10,
            num_rays=2,
            tmin=0.0,
            tmax=3.0,
            density_scale=0.1,
            hit_prob=0.8,
            overlap_prob=0.1,
            seed=1,
            device=DEVICE,
        )
        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        torch.testing.assert_close(c, c_ref, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(d, d_ref, atol=5e-5, rtol=5e-5)

    def test_random_scene_reference_inside(self):
        scene = create_random_test_scene(
            n=8,
            num_rays=2,
            tmin=0.0,
            tmax=3.0,
            density_scale=0.1,
            hit_prob=0.8,
            overlap_prob=0.1,
            seed=3,
            device=DEVICE,
        )
        scene["mean"][0] = torch.zeros(3, device=DEVICE)
        scene["scale"][0] = 0.3
        c_ref, d_ref = _trace_reference(scene)
        c, d = _trace(scene)
        torch.testing.assert_close(c, c_ref, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(d, d_ref, atol=5e-5, rtol=5e-5)

    def test_per_ray_tmax_reference(self):
        num_rays = 5
        scene = create_random_test_scene(
            n=8,
            num_rays=num_rays,
            tmin=0.0,
            tmax=3.0,
            density_scale=0.1,
            hit_prob=0.8,
            overlap_prob=0.1,
            seed=5,
            device=DEVICE,
        )
        tmax = 1.0 + 2.0 * torch.rand(num_rays, device=DEVICE)

        c_ref, d_ref = _trace_reference(scene, tmin=0, tmax=tmax)
        c, d = _trace(scene, tmin=0, tmax=tmax)
        torch.testing.assert_close(c, c_ref, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(d, d_ref, atol=5e-5, rtol=5e-5)


class TestInvariances:
    def test_quat_scale_invariance(self):
        scene = _create_simple_scene(device=DEVICE)
        scene_scaled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}
        scene_scaled["quat"][2] = scene_scaled["quat"][2] * 1.1

        c_ref, d_ref = _trace(scene)
        c, d = _trace(scene_scaled)
        torch.testing.assert_close(c, c_ref, atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(d, d_ref, atol=5e-6, rtol=5e-6)

    def test_quat_rotation_invariance_for_sphere(self):
        scene = _create_simple_scene(device=DEVICE)
        scene_rot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}

        q_alt = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=DEVICE)
        q_alt = q_alt / q_alt.norm()
        scene_rot["quat"][1] = q_alt

        c_ref, d_ref = _trace(scene)
        c, d = _trace(scene_rot)
        torch.testing.assert_close(c, c_ref, atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(d, d_ref, atol=5e-6, rtol=5e-6)

    def test_primitive_permutation_invariance(self):
        scene = create_random_test_scene(
            n=32,
            num_rays=16,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.7,
            overlap_prob=0.1,
            seed=23,
            device=DEVICE,
        )
        color_ref, depth_ref = _trace(scene, max_hits=64, return_extras=False)

        perm = torch.randperm(scene["mean"].shape[0], device=DEVICE)
        scene_perm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}
        for key in ["mean", "scale", "quat", "density", "features"]:
            scene_perm[key] = scene_perm[key][perm]

        color, depth = _trace(scene_perm, max_hits=64, return_extras=False)
        torch.testing.assert_close(color, color_ref, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(depth, depth_ref, atol=5e-5, rtol=5e-5)

    def test_rayd_normalization_in_shader(self):
        scene = _create_simple_scene(device=DEVICE)

        color_ref, depth_ref = _trace(scene, max_hits=64, return_extras=False)

        scene_scaled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}
        scene_scaled["rayd"] = scene_scaled["rayd"] * 3.7
        color, depth = _trace(scene_scaled, max_hits=64, return_extras=False)

        torch.testing.assert_close(color, color_ref, atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(depth, depth_ref, atol=5e-6, rtol=5e-6)


class TestAnalyticAndShading:
    def test_single_ellipsoid_analytic(self):
        scene = _single_ellipsoid_scene(scale=(0.5, 0.5, 1.0), density=1.7, device=DEVICE)
        color, depth = _trace(scene, max_hits=8, return_extras=False)

        t_entry = 1.0
        length = 2.0
        sigma = 1.7
        tau = sigma * length
        alpha = 1.0 - math.exp(-tau)

        expected_rgb = scene["features"][0, 0] * SH_C0 + 0.5
        expected_rgb = expected_rgb * alpha
        expected_depth = t_entry + (1.0 / sigma) - length * math.exp(-tau) / (1.0 - math.exp(-tau))

        torch.testing.assert_close(color[0, :3], expected_rgb, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(color[0, 3], torch.tensor(alpha, device=DEVICE), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(depth[0], torch.tensor(expected_depth, device=DEVICE), atol=2e-5, rtol=2e-5)

    def test_sh_degree_affects_color(self):
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
            "mean": mean,
            "scale": scale,
            "quat": quat,
            "density": density,
            "features": features_deg0,
            "rayo": rayo,
            "rayd": rayd,
            "tmin": 0.0,
            "tmax": 10.0,
        }
        color0, _ = _trace(scene0, max_hits=8, return_extras=False)

        features_deg1 = torch.zeros(1, 4, 3, device=device)
        features_deg1[:, 0, :] = c0
        features_deg1[:, 2, :] = torch.tensor([0.2, 0.0, 0.0], device=device)

        scene1 = dict(scene0)
        scene1["features"] = features_deg1
        color1, _ = _trace(scene1, max_hits=8, return_extras=False)

        expected_color = eval_sh_torch(mean, features_deg1, rayo, sh_degree=1, apply_clip=True)
        sigma = float(density.item())
        tau = sigma * 2.0
        alpha = 1.0 - math.exp(-tau)
        expected_rgb = expected_color[0] * alpha

        torch.testing.assert_close(color1[0, :3], expected_rgb, atol=2e-5, rtol=2e-5)
        assert not torch.allclose(color0, color1)

    def test_no_hit_alpha_depth_zero(self):
        scene = _single_ellipsoid_scene(density=0.0, device=DEVICE)
        color, depth = _trace(scene, max_hits=4, return_extras=False)
        torch.testing.assert_close(color[0, 3], torch.tensor(0.0, device=DEVICE), atol=1e-6, rtol=0)
        torch.testing.assert_close(depth[0], torch.tensor(0.0, device=DEVICE), atol=1e-6, rtol=0)


class TestInputsAndLifecycle:
    def test_density_shape_compatibility(self):
        scene = _create_simple_scene(device=DEVICE)
        scene_flat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}
        scene_flat["density"] = scene_flat["density"].view(-1)

        color_ref, depth_ref = _trace(scene, max_hits=64, return_extras=False)
        color, depth = _trace(scene_flat, max_hits=64, return_extras=False)

        torch.testing.assert_close(color, color_ref, atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(depth, depth_ref, atol=5e-6, rtol=5e-6)

    def test_tmax_broadcast_scalar_vs_tensor(self):
        scene = _create_simple_scene(device=DEVICE)
        num_rays = scene["rayo"].shape[0]
        tmax_scalar = scene["tmax"]
        tmax_tensor = torch.full((num_rays,), float(tmax_scalar), device=DEVICE)

        color_ref, depth_ref = _trace(scene, tmax=tmax_scalar, max_hits=64, return_extras=False)
        color, depth = _trace(scene, tmax=tmax_tensor, max_hits=64, return_extras=False)

        torch.testing.assert_close(color, color_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth, depth_ref, atol=1e-6, rtol=1e-6)

    def test_noncontiguous_inputs_equivalent(self):
        scene = _create_simple_scene(device=DEVICE)
        base_rayo = scene["rayo"]
        base_rayd = scene["rayd"]

        num_rays = base_rayo.shape[0]
        rayo_big = torch.zeros((num_rays * 2, 3), device=DEVICE)
        rayd_big = torch.zeros((num_rays * 2, 3), device=DEVICE)
        rayo_big[::2] = base_rayo
        rayd_big[::2] = base_rayd
        rayo_nc = rayo_big[::2]
        rayd_nc = rayd_big[::2]
        assert not rayo_nc.is_contiguous()
        assert not rayd_nc.is_contiguous()

        scene_nc = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene.items()}
        scene_nc["rayo"] = rayo_nc
        scene_nc["rayd"] = rayd_nc

        color_ref, depth_ref = _trace(scene, max_hits=64, return_extras=False)
        color, depth = _trace(scene_nc, max_hits=64, return_extras=False)

        torch.testing.assert_close(color, color_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth, depth_ref, atol=1e-6, rtol=1e-6)

    def test_invalid_dtype_raises(self):
        scene = _create_simple_scene(device=DEVICE)
        mean_bad = scene["mean"].double()
        with pytest.raises(RuntimeError):
            primtracer.trace_rays(
                mean_bad,
                scene["scale"],
                scene["quat"],
                scene["density"],
                scene["features"],
                scene["rayo"],
                scene["rayd"],
                tmin=scene["tmin"],
                tmax=scene["tmax"],
            )

    def test_invalid_shape_raises(self):
        scene = _create_simple_scene(device=DEVICE)
        mean_bad = torch.zeros(scene["mean"].shape[0], 4, device=DEVICE)
        with pytest.raises(RuntimeError):
            primtracer.trace_rays(
                mean_bad,
                scene["scale"],
                scene["quat"],
                scene["density"],
                scene["features"],
                scene["rayo"],
                scene["rayd"],
                tmin=scene["tmin"],
                tmax=scene["tmax"],
            )

    def test_update_primitives_refresh(self):
        device_index = DEVICE.index or 0
        tracer = primtracer.RayTracer(device_index)

        scene_a = _single_ellipsoid_scene(density=0.0, device=DEVICE)
        scene_b = _single_ellipsoid_scene(density=1.2, device=DEVICE)
        tmax_a = torch.full((scene_a["rayo"].shape[0],), float(scene_a["tmax"]), device=DEVICE)
        tmax_b = torch.full((scene_b["rayo"].shape[0],), float(scene_b["tmax"]), device=DEVICE)

        tracer.update_primitives(
            scene_a["mean"], scene_a["scale"], scene_a["quat"], scene_a["density"], scene_a["features"]
        )
        out_a = tracer.trace_rays(scene_a["rayo"], scene_a["rayd"], scene_a["tmin"], tmax_a, -20.0, 8)

        tracer.update_primitives(
            scene_b["mean"], scene_b["scale"], scene_b["quat"], scene_b["density"], scene_b["features"]
        )
        out_b = tracer.trace_rays(scene_b["rayo"], scene_b["rayd"], scene_b["tmin"], tmax_b, -20.0, 8)

        assert out_a["color"].abs().sum().item() < out_b["color"].abs().sum().item()


class TestHitCollectionAndLimits:
    def test_hit_collection_bounds(self):
        scene = create_random_test_scene(
            n=128,
            num_rays=64,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.9,
            overlap_prob=0.3,
            seed=11,
            device=DEVICE,
        )
        max_hits = 32
        _, _, extras = _trace(scene, max_hits=max_hits, return_extras=True)

        ray_hits = extras["ray_hits"].cpu()
        hit_collection = extras["hit_collection"].cpu()

        num_rays = scene["rayo"].shape[0]
        num_prims = scene["mean"].shape[0]

        assert hit_collection.numel() == num_rays * max_hits
        assert ray_hits.min().item() >= 0
        assert ray_hits.max().item() <= max_hits

        for r in range(num_rays):
            n = int(ray_hits[r].item())
            for i in range(n):
                hit_id = int(hit_collection[r + i * num_rays].item())
                assert 0 <= hit_id < num_prims * 2

    def test_prim_hits_matches_hit_collection(self):
        scene = create_random_test_scene(
            n=256,
            num_rays=128,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.95,
            overlap_prob=0.6,
            seed=17,
            device=DEVICE,
        )
        max_hits = 64
        _, _, extras = _trace(scene, max_hits=max_hits, return_extras=True)

        ray_hits = extras["ray_hits"].cpu()
        hit_collection = extras["hit_collection"].cpu()
        prim_hits = extras["prim_hits"].cpu()

        num_rays = ray_hits.numel()
        num_prims = prim_hits.numel()

        hits = hit_collection.view(max_hits, num_rays).transpose(0, 1)
        mask = torch.arange(max_hits).unsqueeze(0) < ray_hits.unsqueeze(1)
        selected = hits[mask]
        prim_idx = (selected >> 1).to(torch.int64)
        counts = torch.bincount(prim_idx, minlength=num_prims).to(torch.int32)

        assert torch.equal(counts, prim_hits)

    def test_entry_exit_order_single_prim(self):
        scene = _single_ellipsoid_scene(scale=(1.0, 1.0, 1.0), density=1.0, device=DEVICE)
        _, _, extras = _trace(scene, max_hits=8, return_extras=True)

        ray_hits = int(extras["ray_hits"][0].item())
        hit_collection = extras["hit_collection"].cpu()
        assert ray_hits >= 2
        assert hit_collection[0].item() == 1
        assert hit_collection[1].item() == 0

    def test_min_logT_early_stop(self):
        scene = create_random_test_scene(
            n=64,
            num_rays=32,
            tmin=0.0,
            tmax=3.0,
            density_scale=2.0,
            hit_prob=0.9,
            overlap_prob=0.4,
            seed=19,
            device=DEVICE,
        )
        color_full, _, extras_full = _trace(scene, min_logT=-10.0, max_hits=64, return_extras=True)
        color_early, _, extras_early = _trace(scene, min_logT=-0.05, max_hits=64, return_extras=True)

        alpha_full = color_full[:, 3]
        alpha_early = color_early[:, 3]
        assert torch.all(alpha_early <= alpha_full + 1e-5)
        assert extras_early["ray_hits"].max() <= extras_full["ray_hits"].max()

    def test_max_hits_monotonic(self):
        scene = create_random_test_scene(
            n=64,
            num_rays=32,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.9,
            overlap_prob=0.4,
            seed=29,
            device=DEVICE,
        )
        _, _, extras_small = _trace(scene, max_hits=4, return_extras=True)
        _, _, extras_large = _trace(scene, max_hits=32, return_extras=True)
        assert torch.all(extras_small["ray_hits"] <= extras_large["ray_hits"])


class TestRobustnessAndDeterminism:
    def test_extreme_values_finite(self):
        scene = create_random_test_scene(
            n=16,
            num_rays=16,
            tmin=0.0,
            tmax=5.0,
            density_scale=100.0,
            scale_range=(1e-3, 5.0),
            hit_prob=0.8,
            overlap_prob=0.2,
            seed=31,
            device=DEVICE,
        )
        color, depth = _trace(scene, max_hits=64, return_extras=False)
        assert torch.isfinite(color).all()
        assert torch.isfinite(depth).all()
        assert torch.all(color[:, 3] >= -1e-5)
        assert torch.all(color[:, 3] <= 1.0 + 1e-5)

    def test_repeat_determinism(self):
        scene = create_random_test_scene(
            n=16,
            num_rays=8,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.8,
            overlap_prob=0.2,
            seed=37,
            device=DEVICE,
        )
        color1, depth1 = _trace(scene, max_hits=32, return_extras=False)
        color2, depth2 = _trace(scene, max_hits=32, return_extras=False)
        torch.testing.assert_close(color1, color2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth1, depth2, atol=1e-6, rtol=1e-6)

    def test_batch_split_consistency(self):
        scene = create_random_test_scene(
            n=32,
            num_rays=12,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.8,
            overlap_prob=0.2,
            seed=41,
            device=DEVICE,
        )
        color_full, depth_full = _trace(scene, max_hits=64, return_extras=False)

        rayo = scene["rayo"]
        rayd = scene["rayd"]
        mid = rayo.shape[0] // 2

        scene_a = dict(scene)
        scene_a["rayo"] = rayo[:mid]
        scene_a["rayd"] = rayd[:mid]

        scene_b = dict(scene)
        scene_b["rayo"] = rayo[mid:]
        scene_b["rayd"] = rayd[mid:]

        color_a, depth_a = _trace(scene_a, max_hits=64, return_extras=False)
        color_b, depth_b = _trace(scene_b, max_hits=64, return_extras=False)

        color_cat = torch.cat([color_a, color_b], dim=0)
        depth_cat = torch.cat([depth_a, depth_b], dim=0)

        torch.testing.assert_close(color_cat, color_full, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(depth_cat, depth_full, atol=1e-6, rtol=1e-6)


class TestGradients:
    _EPS = 1e-3
    _NUM_DIR = 3
    _REL_ERR_STABLE = 0.2
    _MAX_TRIES = 10

    def _run_gradcheck(self, scene, output="color", check_names=None, tol_mult=1.0):
        if check_names is None:
            check_names = ["mean", "scale", "quat", "density", "features"]
        params = {
            "mean": torch.nn.Parameter(scene["mean"]),
            "scale": torch.nn.Parameter(scene["scale"]),
            "quat": torch.nn.Parameter(scene["quat"]),
            "density": torch.nn.Parameter(scene["density"]),
            "features": torch.nn.Parameter(scene["features"]),
        }

        def loss_fn(mean, scale, quat, density, features):
            color, depth = primtracer.trace_rays(
                mean, scale, quat, density, features,
                scene["rayo"], scene["rayd"],
                tmin=scene.get("tmin", 0),
                tmax=scene.get("tmax", 100),
            )
            return color.sum() if output == "color" else depth.sum()

        self._directional_check(output, params, loss_fn, check_names, tol_mult=tol_mult)

    def _directional_check(self, output, params, loss_fn, names, eps=None, num_dir=None, seed=0, tol_mult=1.0):
        if eps is None:
            eps = self._EPS
        if num_dir is None:
            num_dir = self._NUM_DIR
        grads = torch.autograd.grad(
            loss_fn(**params),
            [params[n] for n in names],
            allow_unused=True,
        )
        static = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in params.items()}

        for name, grad in zip(names, grads):
            if grad is None:
                continue
            base = static[name]
            flat = base.reshape(-1)
            gflat = grad.detach().reshape(-1)
            step_scale = max(1.0, float(base.abs().max().item()))

            gen = torch.Generator(device=base.device)
            gen.manual_seed(seed + abs(hash(name)) % 1000)

            stable = 0
            tries = 0
            max_tries = num_dir * self._MAX_TRIES

            while stable < num_dir and tries < max_tries:
                tries += 1
                v = torch.randn(flat.shape, device=flat.device, generator=gen)
                v = v / (v.abs().max() + 1e-12)

                grad_dir = float((gflat * v).sum().item())
                v_shaped = v.view_as(base)

                def eval_loss(active):
                    active_params = dict(static)
                    active_params[name] = active
                    return loss_fn(**active_params)

                step = eps * step_scale
                loss_pos = eval_loss(base + step * v_shaped)
                loss_neg = eval_loss(base - step * v_shaped)
                fd1 = float((loss_pos - loss_neg).item() / (2.0 * step))

                step2 = 0.5 * step
                loss_pos2 = eval_loss(base + step2 * v_shaped)
                loss_neg2 = eval_loss(base - step2 * v_shaped)
                fd2 = float((loss_pos2 - loss_neg2).item() / (2.0 * step2))

                step3 = 0.5 * step2
                loss_pos3 = eval_loss(base + step3 * v_shaped)
                loss_neg3 = eval_loss(base - step3 * v_shaped)
                fd3 = float((loss_pos3 - loss_neg3).item() / (2.0 * step3))

                err_est = max(abs(fd1 - fd2), abs(fd2 - fd3))
                rel_err = err_est / max(1e-6, abs(fd1))

                if rel_err > self._REL_ERR_STABLE:
                    continue

                err_scale = 2.0
                tol_scale = 1e-3
                if output == "depth":
                    err_scale = 2.0
                    tol_scale = 1e-3
                tol = max(1e-5, err_scale * err_est, tol_scale * abs(fd1)) * tol_mult
                assert abs(grad_dir - fd1) <= tol, (
                    f"{output} {name} directional check failed: "
                    f"|grad_dir - fd|={abs(grad_dir - fd1):.3e} > tol={tol:.3e}"
                )
                stable += 1

            assert stable == num_dir, (
                f"{output} {name} directional check unstable: "
                f"only {stable}/{num_dir} stable directions (tries={tries})"
            )

    def test_grad_color_simple_scene(self):
        scene = _create_simple_scene(device=DEVICE)
        self._run_gradcheck(scene, output="color", tol_mult=1.0)

    def test_grad_depth_simple_scene(self):
        scene = _create_simple_scene(device=DEVICE)
        self._run_gradcheck(scene, output="depth", tol_mult=1.0)

    def test_grad_quat_sphere_near_zero(self):
        scene = _create_simple_scene(device=DEVICE)
        mean = scene["mean"].clone().detach().requires_grad_(True)
        scale = scene["scale"].clone().detach().requires_grad_(True)
        quat = scene["quat"].clone().detach().requires_grad_(True)
        density = scene["density"].clone().detach().requires_grad_(True)
        features = scene["features"].clone().detach().requires_grad_(True)

        color, depth = primtracer.trace_rays(
            mean, scale, quat, density, features,
            scene["rayo"], scene["rayd"],
            tmin=scene.get("tmin", 0),
            tmax=scene.get("tmax", 100),
        )

        grad_c = torch.autograd.grad(color.sum(), quat, retain_graph=True)[0]
        grad_d = torch.autograd.grad(depth.sum(), quat)[0]

        torch.testing.assert_close(grad_c[1], torch.zeros_like(grad_c[1]), atol=5e-5, rtol=0)
        torch.testing.assert_close(grad_d[1], torch.zeros_like(grad_d[1]), atol=5e-5, rtol=0)


class TestPerformance:
    def test_perf_baseline(self):
        max_ms = float(os.getenv("PERF_MAX_MS", "200.0"))
        scene = create_random_test_scene(
            n=1024,
            num_rays=1024,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.8,
            overlap_prob=0.2,
            seed=43,
            device=DEVICE,
        )

        _trace(scene, max_hits=64, return_extras=False)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            _trace(scene, max_hits=64, return_extras=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5.0

        print(f"perf_avg_ms={elapsed_ms:.2f}")
        assert elapsed_ms <= max_ms, f"perf_avg_ms={elapsed_ms:.2f} > max_ms={max_ms:.2f}"

    def test_perf_backward_baseline(self):
        max_ms = float(os.getenv("PERF_BWD_MAX_MS", os.getenv("PERF_MAX_MS", "300.0")))
        scene = create_random_test_scene(
            n=1024,
            num_rays=1024,
            tmin=0.0,
            tmax=3.0,
            density_scale=1.0,
            hit_prob=0.8,
            overlap_prob=0.2,
            seed=47,
            device=DEVICE,
        )

        # Warmup backward
        mean = scene["mean"].clone().detach().requires_grad_(True)
        scale = scene["scale"].clone().detach().requires_grad_(True)
        quat = scene["quat"].clone().detach().requires_grad_(True)
        density = scene["density"].clone().detach().requires_grad_(True)
        features = scene["features"].clone().detach().requires_grad_(True)
        color, depth = primtracer.trace_rays(
            mean, scale, quat, density, features,
            scene["rayo"], scene["rayd"],
            tmin=scene["tmin"],
            tmax=scene["tmax"],
        )
        loss = color.sum() + depth.sum()
        loss.backward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            mean = scene["mean"].clone().detach().requires_grad_(True)
            scale = scene["scale"].clone().detach().requires_grad_(True)
            quat = scene["quat"].clone().detach().requires_grad_(True)
            density = scene["density"].clone().detach().requires_grad_(True)
            features = scene["features"].clone().detach().requires_grad_(True)

            color, depth = primtracer.trace_rays(
                mean, scale, quat, density, features,
                scene["rayo"], scene["rayd"],
                tmin=scene["tmin"],
                tmax=scene["tmax"],
            )
            loss = color.sum() + depth.sum()
            loss.backward()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5.0

        print(f"perf_bwd_avg_ms={elapsed_ms:.2f}")
        assert elapsed_ms <= max_ms, f"perf_bwd_avg_ms={elapsed_ms:.2f} > max_ms={max_ms:.2f}"
