"""
Utilities and reference implementations for:
- Safe math in JAX (no NaNs in forward/backward)
- Quadrature-based ray marching rendering (JAX)
- Spherical harmonics evaluation (Torch)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp
from jax import config

# -----------------------------------------------------------------------------
# JAX config
# -----------------------------------------------------------------------------
config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TINY = np.float32(np.finfo(np.float32).tiny)
MIN_VAL = np.float32(np.finfo(np.float32).min)
MAX_VAL = np.float32(np.finfo(np.float32).max)

# Threshold for alpha below which depth is considered undefined (no absorption)
ALPHA_THRESHOLD = 1e-6

# Spherical harmonics coefficients (kept for completeness / potential use)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
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
# Torch utilities
# =============================================================================
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x, device: torch.device | None = None) -> torch.Tensor:
    """Convert input to torch tensor (optionally move to device)."""
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    return torch.as_tensor(x, device=device)


def l2_normalize(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Normalize to unit length along last axis (torch)."""
    if eps is None:
        eps = torch.finfo(x.dtype if x.is_floating_point() else torch.float32).eps
    denom = torch.sqrt(torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), min=eps))
    return x / denom


def quat_to_mat3(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (r,x,y,z) -> rotation matrix (3x3) in torch."""
    r, x, y, z = q[0], q[1], q[2], q[3]
    mat = torch.tensor(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - r * z),
            2.0 * (x * z + r * y),
            2.0 * (x * y + r * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - r * x),
            2.0 * (x * z - r * y),
            2.0 * (y * z + r * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=q.dtype,
        device=q.device,
    ).reshape(3, 3)
    return mat.T


# =============================================================================
# JAX utilities
# =============================================================================
def l2_normalize_jax(x: jnp.ndarray, eps: float | None = None) -> jnp.ndarray:
    """Normalize to unit length along last axis (jax)."""
    if eps is None:
        eps = jnp.finfo(jnp.float32).eps
    # NOTE: use jnp.sum (not np.sum) to stay in JAX.
    denom = jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), eps))
    return x / denom


def quat_to_mat3_jax(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion (r,x,y,z) -> rotation matrix (3x3)."""
    r, x, y, z = q[0], q[1], q[2], q[3]
    mat = jnp.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - r * z),
            2.0 * (x * z + r * y),
            2.0 * (x * y + r * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - r * x),
            2.0 * (x * z - r * y),
            2.0 * (y * z + r * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    ).reshape(3, 3)
    # Keep behavior consistent with your original code (transpose at end)
    return mat.T


# =============================================================================
# Test data generators (Torch)
# =============================================================================
def create_primitives(
    n: int,
    density_scale: float = 1.0,
    sh_degree: int = 3,
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Create random primitive parameters (Torch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or get_device()
    num_sh_coeff = (sh_degree + 1) ** 2

    return {
        "mean": torch.rand(n, 3, device=device) * 2 - 1,
        "scale": 0.1 + 0.4 * torch.rand(n, 3, device=device),
        "quat": l2_normalize(2 * torch.rand(n, 4, device=device) - 1),
        "density": density_scale * torch.rand(n, 1, device=device),
        "features": torch.rand(n, num_sh_coeff, 3, device=device),
    }


def create_rays(
    n: int = 1,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create rays (Torch) with given origin and direction."""
    device = device or get_device()

    rayo = torch.tensor([origin], dtype=torch.float32, device=device).expand(n, -1)
    rayo = rayo.contiguous()

    rayd = torch.tensor([direction], dtype=torch.float32, device=device)
    rayd = l2_normalize(rayd).expand(n, -1).contiguous()

    return rayo, rayd


def create_frustum_rays(
    n: int,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    look_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    fov_deg: float = 6.0,
    aspect: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create rays within a small frustum around a look direction."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or get_device()
    rayo = torch.tensor([origin], dtype=torch.float32, device=device).expand(n, -1)

    look = l2_normalize(torch.tensor(look_dir, dtype=torch.float32, device=device))
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    if torch.abs((look * world_up).sum()) > 0.95:
        world_up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

    right = l2_normalize(torch.cross(look, world_up, dim=0))
    up = l2_normalize(torch.cross(right, look, dim=0))

    tan_half = float(np.tan(np.deg2rad(fov_deg * 0.5)))
    xy = (torch.rand(n, 2, device=device) * 2.0 - 1.0)
    xy[:, 0] *= tan_half * aspect
    xy[:, 1] *= tan_half

    rayd = l2_normalize(
        look.reshape(1, 3) + xy[:, 0:1] * right.reshape(1, 3) + xy[:, 1:2] * up.reshape(1, 3)
    )
    return rayo.contiguous(), rayd.contiguous()


def create_primitives_near_rays(
    n: int,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 3.0,
    density_scale: float = 1.0,
    sh_degree: int = 3,
    hit_prob: float = 0.5,
    overlap_prob: float = 0.1,
    scale_range: Tuple[float, float] = (0.05, 0.2),
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Create primitives distributed near ray paths with controllable hit/overlap rates."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or get_device()
    rayo = rayo.to(device)
    rayd = l2_normalize(rayd.to(device))

    num_sh_coeff = (sh_degree + 1) ** 2
    num_rays = rayd.shape[0]

    base_scale = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(
        n, 1, device=device
    )
    scale = base_scale * (0.8 + 0.4 * torch.rand(n, 3, device=device))

    ray_idx = torch.randint(0, num_rays, (n,), device=device)
    t = tmin + (tmax - tmin) * torch.rand(n, 1, device=device)
    base = rayo[ray_idx] + t * rayd[ray_idx]

    ray = rayd[ray_idx]
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    world_up = world_up.expand_as(ray)
    alt_up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).expand_as(ray)
    near_parallel = torch.abs((ray * world_up).sum(dim=1)) > 0.95
    up = torch.where(near_parallel.unsqueeze(1), alt_up, world_up)
    right = l2_normalize(torch.cross(ray, up, dim=1))
    up2 = l2_normalize(torch.cross(right, ray, dim=1))

    angles = 2.0 * np.pi * torch.rand(n, 1, device=device)
    dir_perp = torch.cos(angles) * right + torch.sin(angles) * up2

    scale_ref = scale.mean(dim=1, keepdim=True)
    hit_mask = (torch.rand(n, 1, device=device) < hit_prob)
    radius_hit = scale_ref * (0.2 + 0.6 * torch.rand(n, 1, device=device))
    radius_miss = scale_ref * (2.0 + 2.0 * torch.rand(n, 1, device=device))
    radius = torch.where(hit_mask, radius_hit, radius_miss)

    mean = base + dir_perp * radius

    num_overlap = int(round(n * overlap_prob))
    if n > 1 and num_overlap > 0:
        src_idx = torch.randint(0, n, (num_overlap,), device=device)
        dst_idx = torch.randint(0, n, (num_overlap,), device=device)
        jitter = (torch.rand(num_overlap, 3, device=device) - 0.5) * scale[src_idx] * 0.5
        mean[dst_idx] = mean[src_idx] + jitter

    return {
        "mean": mean,
        "scale": scale,
        "quat": l2_normalize(2 * torch.rand(n, 4, device=device) - 1),
        "density": density_scale * torch.rand(n, 1, device=device),
        "features": torch.rand(n, num_sh_coeff, 3, device=device),
    }


def create_random_test_scene(
    n: int,
    num_rays: int = 8,
    frustum_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    frustum_look_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    frustum_fov_deg: float = 15.0,
    tmin: float = 0.0,
    tmax: float = 3.0,
    density_scale: float = 1.0,
    sh_degree: int = 3,
    scale_range: Tuple[float, float] = (0.05, 0.2),
    hit_prob: float = 0.5,
    overlap_prob: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Create a full scene: frustum rays + primitives near rays."""
    rayo, rayd = create_frustum_rays(
        n=num_rays,
        origin=frustum_origin,
        look_dir=frustum_look_dir,
        fov_deg=frustum_fov_deg,
        seed=seed,
        device=device,
    )
    prims = create_primitives_near_rays(
        n,
        rayo,
        rayd,
        tmin=tmin,
        tmax=tmax,
        density_scale=density_scale,
        sh_degree=sh_degree,
        hit_prob=hit_prob,
        overlap_prob=overlap_prob,
        scale_range=scale_range,
        seed=seed,
        device=device,
    )
    return {**prims, "rayo": rayo, "rayd": rayd, "tmin": tmin, "tmax": tmax}


def export_scene_obj(
    path: str,
    scene: Dict[str, torch.Tensor],
) -> None:
    """Export a scene as OBJ: ellipsoids -> low-res meshes, rays -> thin quads.

    Density is stored as per-vertex grayscale in the OBJ "v" lines.
    """
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

    # Low-res unit sphere (lat-long), used as ellipsoid via scale+rotation.
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
        sphere_vertices, dtype=means.dtype, device=means.device
    )
    for i in range(means.shape[0]):
        scale = scales[i].reshape(1, 3)
        local = sphere_vertices_t * scale
        R = quat_to_mat3(l2_normalize(quats[i]))
        world = (local @ R.T) + means[i].reshape(1, 3)

        v_offset = len(vertices) + 1
        vertices.extend(world.tolist())
        vert_colors.extend([float(densities[i])] * len(sphere_vertices))
        for f in sphere_faces:
            faces.append([v_offset + idx for idx in f])

    ray_length = float(scene["tmax"]) - float(scene["tmin"])
    for i in range(rayo.shape[0]):
        start = rayo[i]
        end = rayo[i] + rayd[i] * ray_length
        ray = l2_normalize(rayd[i])

        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=ray.dtype, device=ray.device)
        if torch.abs((ray * world_up).sum()) > 0.95:
            world_up = torch.tensor([1.0, 0.0, 0.0], dtype=ray.dtype, device=ray.device)
        right = l2_normalize(torch.cross(ray, world_up, dim=0))
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
        f.write("# simple scene export\n")
        for v, c in zip(vertices, vert_colors):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c} {c} {c}\n")
        for face in faces:
            f.write(f"f {' '.join(str(idx) for idx in face)}\n")
        if ray_triangles:
            for tri in ray_triangles:
                f.write(f"f {' '.join(str(idx) for idx in tri)}\n")

# =============================================================================
# JAX safe math (no NaNs in forward/backward)
# =============================================================================
def _remove_zero(x: jnp.ndarray) -> jnp.ndarray:
    """Shift values away from 0 to prevent division blow-ups."""
    return jnp.where(jnp.abs(x) < TINY, TINY, x)


@jax.custom_vjp
def safe_div(n: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Divide n/d with clipping; value and gradients avoid NaNs."""
    return _safe_div_fwd(n, d)[0]


def _safe_div_fwd(n: jnp.ndarray, d: jnp.ndarray):
    r = jnp.clip(n / _remove_zero(d), MIN_VAL, MAX_VAL)
    y = jnp.where(jnp.abs(d) < TINY, 0.0, r)
    return y, (d, r)


def _safe_div_bwd(res, g):
    d, r = res
    dn = jnp.clip(g / _remove_zero(d), MIN_VAL, MAX_VAL)
    dd = jnp.clip(-g * r / _remove_zero(d), MIN_VAL, MAX_VAL)
    return dn, dd


safe_div.defvjp(_safe_div_fwd, _safe_div_bwd)


def generate_safe_fn(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    grad_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x_range: Tuple[float, float],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a safe version of fn(x): clip x both in forward and backward."""

    @jax.custom_jvp
    def safe_fn(x: jnp.ndarray) -> jnp.ndarray:
        return fn(jnp.clip(x, *x_range))

    @safe_fn.defjvp
    def safe_fn_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        y = safe_fn(x)
        y_dot = grad_fn(jnp.clip(x, *x_range), y, x_dot)
        return y, y_dot

    return safe_fn


def safe_log(x: jnp.ndarray) -> jnp.ndarray:
    return generate_safe_fn(
        jnp.log,
        lambda x, _y, x_dot: x_dot / x,
        (TINY, MAX_VAL),
    )(x)


def log1mexp(x: jnp.ndarray) -> jnp.ndarray:
    """Accurate computation of log(1 - exp(-x)) for x > 0."""
    return safe_log(1.0 - jnp.exp(-x))


# =============================================================================
# Quadrature ray tracing (JAX reference)
# =============================================================================
def assert_valid_stepfun(t: jnp.ndarray, y: jnp.ndarray) -> None:
    """Assert step function (t, y) has valid shapes: t[..., M+1], y[..., M]."""
    if t.shape[-1] != y.shape[-1] + 1:
        raise ValueError(f"Invalid shapes ({t.shape}, {y.shape}) for a step function.")


def compute_alpha_weights(density_delta: jnp.ndarray) -> jnp.ndarray:
    """Compute alpha-compositing weights from density * delta."""
    log_trans = -jnp.concatenate(
        [
            jnp.zeros_like(density_delta[..., :1]),
            jnp.cumsum(density_delta[..., :-1], axis=-1),
        ],
        axis=-1,
    )
    log_weights = log1mexp(density_delta) + log_trans
    return jnp.exp(log_weights)


def render_quadrature(
    tdist: jnp.ndarray,
    query_fn: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    return_extras: bool = False,
):
    """
    Numerical quadrature rendering of a set of colored Gaussians.

    Returns:
        If return_extras=False: (color_rgba, depth)
        If return_extras=True: (color_rgba, depth, extras)
    """
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)

    total_density, avg_colors = query_fn(t_avg)  # (S,), (S,3) or broadcastable
    weights = compute_alpha_weights(total_density * t_delta)

    rendered_color = jnp.sum(weights[..., None] * avg_colors, axis=-2)
    alpha = jnp.sum(weights, axis=-1).reshape(-1, 1)
    depth_num = jnp.sum(weights * t_avg, axis=-1)
    # When alpha < ALPHA_THRESHOLD, depth is meaningless (no absorption occurred)
    expected_depth = jnp.where(
        alpha.reshape(-1) < ALPHA_THRESHOLD,
        0.0,
        safe_div(depth_num, alpha.reshape(-1))
    )

    color_rgba = jnp.concatenate([rendered_color.reshape(-1, 3), alpha], axis=1)
    depth = expected_depth.reshape(-1)

    if return_extras:
        extras: Dict[str, jnp.ndarray] = {
            "tdist": tdist,
            "avg_colors": avg_colors,
            "weights": weights,
            "total_density": jnp.sum(total_density * t_delta, axis=-1),
        }
        return color_rgba, depth, extras
    else:
        return color_rgba, depth


def eval_sh_jax(deg: int, sh: jnp.ndarray, dirs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate spherical harmonics (JAX reference)."""
    result = C0 * sh[..., 0]

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

    return result + 0.5


def query_ellipsoid(
    tdist: jnp.ndarray,
    rayo: jnp.ndarray,
    rayd: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Query density/color for a single primitive along points on a ray.
    """
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)

    R = quat_to_mat3_jax(l2_normalize_jax(params["quat"]))
    scale = jnp.maximum(params["scale"].reshape(1, 3), 1e-8)

    sc = ((xs - params["mean"].reshape(1, 3)) @ R.T) / scale
    d = jnp.linalg.norm(sc, axis=-1, ord=2).reshape(-1, 1)

    inside = d < 1.0
    densities = jnp.where(inside, params["density"], 0.0)
    # Evaluate SH color using ray direction
    sh_deg = params["sh_degree"]
    color = eval_sh_jax(sh_deg, params["features"], rayd.reshape(1, 3))
    colors = jnp.where(inside, params["density"] * color, 0.0)
    return densities, colors


def trace_rays_reference(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000.0,
    num_samples: int = 2**18,
    return_extras: bool = False,
):
    """
    Reference ray tracing using dense quadrature (JAX).
    Inputs are torch tensors; converted to numpy->jax arrays inside.

    Returns:
        If return_extras=False: (color_rgba, depth)
        If return_extras=True: (color_rgba, depth, extras)
    """
    # Infer SH degree from features shape: (N, num_coeff, 3) -> deg = sqrt(num_coeff) - 1
    num_coeff = features.shape[1]
    sh_degree = int(np.sqrt(num_coeff)) - 1

    vquery_ellipsoid = jax.vmap(
        query_ellipsoid,
        in_axes=(
            None,
            None,
            None,
            {
                "mean": 0,
                "quat": 0,
                "density": 0,
                "scale": 0,
                "features": 0,
                "sh_degree": None,
            },
        ),
    )

    def sum_vquery_ellipsoid(t: jnp.ndarray, ray_o: jnp.ndarray, ray_d: jnp.ndarray, p):
        densities, colors = vquery_ellipsoid(t, ray_o, ray_d, p)  # (P,S,1), (P,S,3)
        dens = densities.sum(axis=0)  # (S,1) or (S,)
        col = safe_div(colors.sum(axis=0), dens)  # safe average
        return dens.reshape(-1), col.clip(min=0.0)

    tdist = jnp.linspace(tmin, tmax, num_samples + 1)

    # features shape: (N, num_coeff, 3) -> transpose to (N, 3, num_coeff) for SH eval
    features_np = features.detach().cpu().numpy().astype(np.float64)
    features_transposed = features_np.transpose(0, 2, 1)  # (N, 3, num_coeff)

    params = {
        "mean": mean.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        "scale": scale.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        "quat": quat.detach().cpu().numpy().astype(np.float64).reshape(-1, 4),
        "density": density.detach().cpu().numpy().astype(np.float64).reshape(-1, 1),
        "features": features_transposed,
        "sh_degree": sh_degree,
    }
    ray_o = rayo.detach().cpu().numpy().astype(np.float64)
    ray_d = rayd.detach().cpu().numpy().astype(np.float64)

    return render_quadrature(
        tdist,
        lambda t: sum_vquery_ellipsoid(t, ray_o, ray_d, params),
        return_extras=return_extras,
    )


# =============================================================================
# Spherical harmonics (Torch reference)
# =============================================================================
@torch.jit.script
def eval_sh_reference(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics (torch reference)."""
    result = 0.28209479177387814 * sh[..., 0]

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result
            - 0.4886025119029199 * y * sh[..., 1]
            + 0.4886025119029199 * z * sh[..., 2]
            - 0.4886025119029199 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + 1.0925484305920792 * xy * sh[..., 4]
                - 1.0925484305920792 * yz * sh[..., 5]
                + 0.31539156525252005 * (2 * zz - xx - yy) * sh[..., 6]
                - 1.0925484305920792 * xz * sh[..., 7]
                + 0.5462742152960396 * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    - 0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9]
                    + 2.890611442640554 * xy * z * sh[..., 10]
                    - 0.4570457994644658 * y * (4 * zz - xx - yy) * sh[..., 11]
                    + 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    - 0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13]
                    + 1.445305721320277 * z * (xx - yy) * sh[..., 14]
                    - 0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + 2.5033429417967046 * xy * (xx - yy) * sh[..., 16]
                        - 1.7701307697799304 * yz * (3 * xx - yy) * sh[..., 17]
                        + 0.9461746957575601 * xy * (7 * zz - 1) * sh[..., 18]
                        - 0.6690465435572892 * yz * (7 * zz - 3) * sh[..., 19]
                        + 0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        - 0.6690465435572892 * xz * (7 * zz - 3) * sh[..., 21]
                        + 0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        - 1.7701307697799304 * xz * (xx - 3 * yy) * sh[..., 23]
                        + 0.6258357354491761
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )

    return result


def eval_sh_torch(
    means: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    sh_degree: int,
    apply_clip: bool = True,
) -> torch.Tensor:
    """
    Evaluate SH for primitives (torch reference).

    Note: direction convention is from camera (rayo) toward primitive (means).
    """
    n = means.shape[0]
    rayo = rayo.reshape(1, 3)

    dir_pp = means - rayo.expand(n, -1)
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    shs_view = features.transpose(1, 2).reshape(-1, 3, (sh_degree + 1) ** 2)
    result = eval_sh_reference(sh_degree, shs_view, dir_pp) + 0.5

    return result.clamp(min=0.0) if apply_clip else result
