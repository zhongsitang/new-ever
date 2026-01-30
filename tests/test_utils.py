"""
Utilities and reference implementations for:
- Quadrature-based ray marching rendering (PyTorch GPU)
- Spherical harmonics evaluation (PyTorch)
- Test data generators
- Gradient checking utilities
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch

# =============================================================================
# Constants
# =============================================================================
# Threshold for alpha below which depth is considered undefined
ALPHA_THRESHOLD = 1e-6

# Spherical harmonics coefficients
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


# =============================================================================
# Device utilities
# =============================================================================
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Math utilities
# =============================================================================
def l2_normalize(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Normalize to unit length along last axis."""
    if eps is None:
        eps = torch.finfo(x.dtype if x.is_floating_point() else torch.float32).eps
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    return x / torch.sqrt(torch.clamp(norm_sq, min=eps))


def quat_to_mat3(q: torch.Tensor) -> torch.Tensor:
    """Quaternion to rotation matrix.

    Args:
        q: Quaternion(s) in (w,x,y,z) format. Shape (..., 4).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    mat = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return mat.transpose(-1, -2)


# =============================================================================
# Spherical harmonics evaluation
# =============================================================================
def eval_sh(
    deg: int,
    sh: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate spherical harmonics.

    Args:
        deg: SH degree (0-3).
        sh: SH coefficients with shape (..., 3, C) where C = (deg+1)^2.
            Layout: 3 color channels, C coefficients each.
        dirs: Unit directions with shape (..., 3).

    Returns:
        RGB colors with shape (..., 3), values in [0, 1] after +0.5 offset.
    """
    result = SH_C0 * sh[..., 0]  # (..., 3)

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result
                  - SH_C1 * y * sh[..., 1]
                  + SH_C1 * z * sh[..., 2]
                  - SH_C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            result = (result
                      + SH_C2[0] * xy * sh[..., 4]
                      + SH_C2[1] * yz * sh[..., 5]
                      + SH_C2[2] * (2*zz - xx - yy) * sh[..., 6]
                      + SH_C2[3] * xz * sh[..., 7]
                      + SH_C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result
                          + SH_C3[0] * y * (3*xx - yy) * sh[..., 9]
                          + SH_C3[1] * xy * z * sh[..., 10]
                          + SH_C3[2] * y * (4*zz - xx - yy) * sh[..., 11]
                          + SH_C3[3] * z * (2*zz - 3*xx - 3*yy) * sh[..., 12]
                          + SH_C3[4] * x * (4*zz - xx - yy) * sh[..., 13]
                          + SH_C3[5] * z * (xx - yy) * sh[..., 14]
                          + SH_C3[6] * x * (xx - 3*yy) * sh[..., 15])

                if deg > 3:
                    result = (result
                              + SH_C4[0] * xy * (xx - yy) * sh[..., 16]
                              + SH_C4[1] * yz * (3*xx - yy) * sh[..., 17]
                              + SH_C4[2] * xy * (7*zz - 1) * sh[..., 18]
                              + SH_C4[3] * yz * (7*zz - 3) * sh[..., 19]
                              + SH_C4[4] * (zz * (35*zz - 30) + 3) * sh[..., 20]
                              + SH_C4[5] * xz * (7*zz - 3) * sh[..., 21]
                              + SH_C4[6] * (xx - yy) * (7*zz - 1) * sh[..., 22]
                              + SH_C4[7] * xz * (xx - 3*yy) * sh[..., 23]
                              + SH_C4[8] * (xx*(xx - 3*yy) - yy*(3*xx - yy)) * sh[..., 24])

    return result + 0.5


def eval_sh_torch(
    means: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    sh_degree: int,
    apply_clip: bool = True,
) -> torch.Tensor:
    """Evaluate SH for primitives using direction from camera to primitive.

    Args:
        means: Primitive centers (N, 3).
        features: SH coefficients (N, C, 3) where C = (sh_degree+1)^2.
        rayo: Ray origin (1, 3) or (N, 3).
        sh_degree: SH degree.
        apply_clip: Whether to clamp result to [0, inf).

    Returns:
        RGB colors (N, 3).
    """
    n = means.shape[0]
    rayo = rayo.reshape(1, 3) if rayo.dim() == 1 else rayo[:1]

    # Direction from camera to primitive
    dir_pp = means - rayo.expand(n, -1)
    dir_pp = l2_normalize(dir_pp)

    # Transpose features from (N, C, 3) to (N, 3, C) for eval_sh
    sh = features.transpose(1, 2)
    result = eval_sh(sh_degree, sh, dir_pp)

    return result.clamp(min=0.0) if apply_clip else result


# =============================================================================
# Test data generators
# =============================================================================
def create_random_test_scene(
    num_primitives: int,
    # Primitive parameters
    sh_degree: int = 3,
    scale_range: Tuple[float, float] = (0.05, 0.2),
    density_range: Tuple[float, float] = (0.0, 1.0),
    bbox_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bbox_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    # Camera parameters
    cam_eye: Tuple[float, float, float] = (0.0, 0.0, -3.0),
    cam_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    cam_fov: float = 60.0,
    cam_resolution: int = 64,
    # Ray parameters
    tmin: float = 0.0,
    tmax: float = 5.0,
    # Other
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Create a test scene with uniformly distributed ellipsoids and pinhole camera rays.

    Args:
        num_primitives: Number of ellipsoid primitives to generate.
        sh_degree: Spherical harmonics degree for color features.
        scale_range: (min, max) range for ellipsoid scales.
        density_range: (min, max) range for ellipsoid densities.
        bbox_min: Bounding box minimum corner for ellipsoid centers.
        bbox_max: Bounding box maximum corner for ellipsoid centers.
        cam_eye: Camera position in world space.
        cam_dir: Camera look direction (will be normalized).
        cam_fov: Field of view in degrees.
        cam_resolution: Image resolution (generates resolution^2 rays).
        tmin: Ray start parameter.
        tmax: Ray end parameter.
        seed: Random seed for reproducibility.
        device: Torch device (defaults to CUDA if available).

    Returns:
        Scene dictionary with keys: mean, scale, quat, density, features,
        rayo, rayd, tmin, tmax.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or get_device()
    N = num_primitives
    R = cam_resolution * cam_resolution
    num_sh_coeff = (sh_degree + 1) ** 2

    # Camera rays (pinhole model)
    forward = torch.tensor(cam_dir, dtype=torch.float32, device=device)
    forward = l2_normalize(forward.unsqueeze(0)).squeeze(0)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    if torch.abs(torch.dot(forward, world_up)) > 0.95:
        world_up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    right = l2_normalize(torch.cross(forward, world_up, dim=0).unsqueeze(0)).squeeze(0)
    up = l2_normalize(torch.cross(right, forward, dim=0).unsqueeze(0)).squeeze(0)

    # Pixel centers in NDC [-1, 1]
    ndc = torch.arange(cam_resolution, dtype=torch.float32, device=device)
    ndc = (ndc + 0.5) / cam_resolution * 2.0 - 1.0
    px, py = torch.meshgrid(ndc, ndc, indexing='xy')
    px, py = px.reshape(-1), py.reshape(-1)

    # Project to image plane
    tan_half_fov = float(np.tan(np.deg2rad(cam_fov * 0.5)))
    ray_dir = (forward
               + (px * tan_half_fov)[:, None] * right
               - (py * tan_half_fov)[:, None] * up)
    ray_dir = l2_normalize(ray_dir)
    ray_origin = torch.tensor(cam_eye, dtype=torch.float32, device=device)
    ray_origin = ray_origin[None, :].expand(R, 3).contiguous()

    # Ellipsoid primitives
    bbox_lo = torch.tensor(bbox_min, dtype=torch.float32, device=device)
    bbox_hi = torch.tensor(bbox_max, dtype=torch.float32, device=device)

    centers = bbox_lo + torch.rand(N, 3, device=device) * (bbox_hi - bbox_lo)
    scales = (scale_range[0]
              + (scale_range[1] - scale_range[0]) * torch.rand(N, 3, device=device))
    rotations = l2_normalize(torch.rand(N, 4, device=device) * 2 - 1)
    densities = (density_range[0]
                 + (density_range[1] - density_range[0]) * torch.rand(N, 1, device=device))
    sh_features = torch.rand(N, num_sh_coeff, 3, device=device)

    return {
        "mean": centers,
        "scale": scales,
        "quat": rotations,
        "density": densities,
        "features": sh_features,
        "rayo": ray_origin,
        "rayd": ray_dir,
        "tmin": tmin,
        "tmax": tmax,
    }


def export_scene_obj(path: str, scene: Dict[str, torch.Tensor]) -> None:
    """Export a scene as OBJ: ellipsoids -> meshes, rays -> thin quads."""
    means = scene["mean"]
    scales = scene["scale"]
    quats = scene["quat"]
    densities = scene["density"].reshape(-1)
    rayo = scene["rayo"]
    rayd = scene["rayd"]

    vertices = []
    faces = []
    ray_triangles = []
    vert_colors = []

    # Unit sphere mesh
    lat_segments = 25
    lon_segments = 50
    sphere_vertices = []
    sphere_faces = []
    for i in range(lat_segments + 1):
        v = i / lat_segments
        phi = np.pi * v
        y = np.cos(phi)
        r = np.sin(phi)
        for j in range(lon_segments):
            u = j / lon_segments
            theta = 2.0 * np.pi * u
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            sphere_vertices.append([x, y, z])
    for i in range(lat_segments):
        for j in range(lon_segments):
            a = i * lon_segments + j
            b = i * lon_segments + (j + 1) % lon_segments
            c = (i + 1) * lon_segments + (j + 1) % lon_segments
            d = (i + 1) * lon_segments + j
            if i != 0:
                sphere_faces.append([a, b, c])
            if i != lat_segments - 1:
                sphere_faces.append([a, c, d])

    sphere_vertices_t = torch.tensor(
        sphere_vertices, dtype=means.dtype, device=means.device)

    for i in range(means.shape[0]):
        scale = scales[i].reshape(1, 3)
        local = sphere_vertices_t * scale
        R = quat_to_mat3(quats[i:i+1]).squeeze(0)
        world = (local @ R.T) + means[i].reshape(1, 3)

        v_offset = len(vertices) + 1
        vertices.extend(world.tolist())
        vert_colors.extend([float(densities[i])] * len(sphere_vertices))
        for f in sphere_faces:
            faces.append([v_offset + idx for idx in f])

    tmin = scene["tmin"]
    tmax = scene["tmax"]
    for i in range(rayo.shape[0]):
        tmin_i = float(tmin[i]) if isinstance(tmin, torch.Tensor) else float(tmin)
        tmax_i = float(tmax[i]) if isinstance(tmax, torch.Tensor) else float(tmax)
        start = rayo[i] + rayd[i] * tmin_i
        end = rayo[i] + rayd[i] * tmax_i
        ray = l2_normalize(rayd[i:i+1]).squeeze(0)

        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=ray.dtype, device=ray.device)
        if torch.abs((ray * world_up).sum()) > 0.95:
            world_up = torch.tensor([1.0, 0.0, 0.0], dtype=ray.dtype, device=ray.device)
        right = l2_normalize(torch.cross(ray, world_up, dim=0).unsqueeze(0)).squeeze(0)
        offset = right * 0.005

        v_offset = len(vertices) + 1
        vertices.append((start + offset).tolist())
        vertices.append((start - offset).tolist())
        vertices.append((end - offset).tolist())
        vertices.append((end + offset).tolist())
        vert_colors.extend([0.0, 0.0, 0.0, 0.0])
        ray_triangles.append([v_offset, v_offset + 1, v_offset + 2])
        ray_triangles.append([v_offset, v_offset + 2, v_offset + 3])

    with open(path, "w", encoding="ascii") as f:
        f.write("# scene export\n")
        for v, c in zip(vertices, vert_colors):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c} {c} {c}\n")
        for face in faces:
            f.write(f"f {' '.join(str(idx) for idx in face)}\n")
        for tri in ray_triangles:
            f.write(f"f {' '.join(str(idx) for idx in tri)}\n")


# =============================================================================
# Reference ray tracing (PyTorch GPU)
# =============================================================================

def ellipsoid_intersect(
    points: torch.Tensor,
    mean: torch.Tensor,
    R: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Check if points are inside an ellipsoid.

    Args:
        points: Query points (..., 3).
        mean: Ellipsoid center (3,).
        R: Rotation matrix (3, 3).
        inv_scale: Inverse scales (3,).

    Returns:
        Boolean mask (...,) indicating inside status.
    """
    local = (points - mean) @ R.T
    local_scaled = local * inv_scale
    dist_sq = (local_scaled ** 2).sum(dim=-1)
    return dist_sq < 1.0


def trace_rays_reference(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float | torch.Tensor = 0.0,
    tmax: float | torch.Tensor = 1000.0,
    num_samples: int = 2**16,
    ray_batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference ray tracing using dense quadrature (PyTorch GPU).

    Args:
        mean, scale, quat, density, features: Primitive parameters.
        rayo, rayd: Ray origins/directions (M, 3).
        tmin, tmax: Scalar or per-ray (M,) tensor.
        num_samples: Quadrature samples per ray.
        ray_batch_size: Rays per batch for memory management.

    Returns:
        color_rgba: (M, 4) RGBA colors.
        depth: (M,) expected depths.
    """
    device = rayo.device
    dtype = torch.float64
    M = rayo.shape[0]
    N = mean.shape[0]
    S = num_samples

    # Convert to float64 for precision
    mean = mean.to(dtype)
    scale = scale.to(dtype)
    quat = quat.to(dtype)
    dens_vals = density.to(dtype).reshape(N)
    rayo = rayo.to(dtype)
    rayd = rayd.to(dtype)

    # Transpose features from (N, C, 3) to (N, 3, C)
    sh_degree = int(np.sqrt(features.shape[1])) - 1
    features = features.to(dtype).transpose(1, 2)

    # Precompute rotation matrices and inverse scales
    R_mats = quat_to_mat3(quat)  # (N, 3, 3)
    inv_scale = 1.0 / scale.clamp(min=1e-8)  # (N, 3)

    # Handle tmin/tmax
    if isinstance(tmin, (int, float)):
        tmin = torch.full((M,), tmin, device=device, dtype=dtype)
    else:
        tmin = tmin.to(dtype)
    if isinstance(tmax, (int, float)):
        tmax = torch.full((M,), tmax, device=device, dtype=dtype)
    else:
        tmax = tmax.to(dtype)

    colors_out = []
    depths_out = []

    for batch_start in range(0, M, ray_batch_size):
        batch_end = min(batch_start + ray_batch_size, M)
        B = batch_end - batch_start

        ray_o = rayo[batch_start:batch_end]
        ray_d = rayd[batch_start:batch_end]
        t_lo = tmin[batch_start:batch_end]
        t_hi = tmax[batch_start:batch_end]

        # Sample points along rays
        t_lin = torch.linspace(0, 1, S + 1, device=device, dtype=dtype)
        tdist = t_lo[:, None] + (t_hi - t_lo)[:, None] * t_lin[None, :]
        t_avg = 0.5 * (tdist[:, 1:] + tdist[:, :-1])
        t_delta = tdist[:, 1:] - tdist[:, :-1]

        # Sample positions: (B, S, 3)
        xs = ray_o[:, None, :] + t_avg[:, :, None] * ray_d[:, None, :]

        # Accumulate density and color over primitives
        total_dens = torch.zeros(B, S, device=device, dtype=dtype)
        total_color = torch.zeros(B, S, 3, device=device, dtype=dtype)

        for n in range(N):
            # Check intersection with ellipsoid
            inside = ellipsoid_intersect(xs, mean[n], R_mats[n], inv_scale[n])

            # Density contribution
            d_n = dens_vals[n]
            dens_n = torch.where(inside, d_n, torch.zeros((), device=device, dtype=dtype))

            # SH color for this primitive
            sh_color_n = eval_sh(sh_degree, features[n], ray_d).clamp(min=0.0)

            # Accumulate
            total_dens += dens_n
            total_color += dens_n[:, :, None] * sh_color_n[:, None, :]

        # Density-weighted average color
        avg_color = total_color / (total_dens[:, :, None] + 1e-10)
        avg_color = torch.where(total_dens[:, :, None] > 1e-10,
                                avg_color, torch.zeros_like(avg_color))

        # Alpha compositing
        tau = total_dens * t_delta
        log_trans = -torch.cat([
            torch.zeros(B, 1, device=device, dtype=dtype),
            tau[:, :-1].cumsum(dim=-1)
        ], dim=-1)
        alpha_i = 1.0 - torch.exp(-tau)
        weights = alpha_i * torch.exp(log_trans)

        # Render
        rendered_color = (weights[:, :, None] * avg_color).sum(dim=1)
        alpha = weights.sum(dim=1)
        depth_num = (weights * t_avg).sum(dim=1)
        expected_depth = torch.where(
            alpha > ALPHA_THRESHOLD,
            depth_num / (alpha + 1e-10),
            torch.zeros_like(alpha)
        )

        colors_out.append(torch.cat([rendered_color, alpha[:, None]], dim=-1))
        depths_out.append(expected_depth)

    color_rgba = torch.cat(colors_out, dim=0).float()
    depth = torch.cat(depths_out, dim=0).float()

    return color_rgba, depth


# =============================================================================
# Gradient checking utilities
# =============================================================================
def directional_gradcheck(
    loss_fn: Callable[..., torch.Tensor],
    params: Dict[str, torch.Tensor],
    eps: float = 1e-3,
    num_directions: int = 3,
    err_scale: float = 3.0,
) -> None:
    """Verify gradients using directional finite differences.

    Args:
        loss_fn: Function that takes **params and returns a scalar loss.
        params: Dict of parameter name -> tensor (with requires_grad=True).
        eps: Base step size for finite differences.
        num_directions: Number of stable directions required per parameter.
        err_scale: Multiplier for estimated numerical error in tolerance.

    Raises:
        AssertionError: If gradient check fails for any parameter.
    """
    param_names = list(params.keys())

    # Compute analytic gradients
    loss = loss_fn(**params)
    grads = torch.autograd.grad(
        loss, [params[n] for n in param_names], allow_unused=True)
    static = {k: v.detach() if torch.is_tensor(v) else v
              for k, v in params.items()}

    for name, grad in zip(param_names, grads):
        if grad is None:
            continue

        base = static[name]
        flat = base.reshape(-1)
        gflat = grad.detach().reshape(-1)
        step_scale = max(1.0, float(base.abs().max().item()))

        gen = torch.Generator(device=base.device)
        gen.manual_seed(abs(hash(name)) % 1000)

        stable = 0
        tries = 0
        max_tries = num_directions * 10

        while stable < num_directions and tries < max_tries:
            tries += 1
            v = torch.randn(flat.shape, device=flat.device, generator=gen)
            v = v / (v.abs().max() + 1e-12)

            grad_dir = float((gflat * v).sum().item())
            v_shaped = v.view_as(base)

            def eval_loss(active):
                active_params = dict(static)
                active_params[name] = active
                return loss_fn(**active_params)

            # Three-point Richardson extrapolation
            step = eps * step_scale
            fd1 = float((eval_loss(base + step * v_shaped)
                         - eval_loss(base - step * v_shaped)).item()
                        / (2.0 * step))

            step2 = 0.5 * step
            fd2 = float((eval_loss(base + step2 * v_shaped)
                         - eval_loss(base - step2 * v_shaped)).item()
                        / (2.0 * step2))

            step3 = 0.25 * step
            fd3 = float((eval_loss(base + step3 * v_shaped)
                         - eval_loss(base - step3 * v_shaped)).item()
                        / (2.0 * step3))

            err_est = max(abs(fd1 - fd2), abs(fd2 - fd3))
            rel_err = err_est / max(1e-6, abs(fd1))

            if rel_err > 0.2:
                continue

            tol = max(1e-5, err_scale * err_est, 1e-3 * abs(fd1))
            assert abs(grad_dir - fd1) <= tol, (
                f"{name} gradient check failed: "
                f"|grad - fd|={abs(grad_dir - fd1):.3e} > tol={tol:.3e}"
            )
            stable += 1

        assert stable == num_directions, (
            f"{name} gradient check unstable: "
            f"only {stable}/{num_directions} stable directions (tries={tries})"
        )
