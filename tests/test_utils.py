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
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Create random primitive parameters (Torch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or get_device()

    return {
        "mean": torch.rand(n, 3, device=device) * 2 - 1,
        "scale": 0.1 + 0.4 * torch.rand(n, 3, device=device),
        "quat": l2_normalize(2 * torch.rand(n, 4, device=device) - 1),
        "density": density_scale * torch.rand(n, 1, device=device),
        "features": torch.rand(n, 1, 3, device=device),
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


@jax.jit
def lossfun_distortion(t: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Compute ∬ w[i] w[j] |t[i] - t[j]| di dj."""
    assert_valid_stepfun(t, w)

    ut = (t[..., 1:] + t[..., :-1]) / 2.0
    dut = jnp.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = jnp.sum(w * jnp.sum(w[..., None, :] * dut, axis=-1), axis=-1)

    loss_intra = jnp.sum(w**2 * jnp.diff(t), axis=-1) / 3.0
    return loss_inter + loss_intra


def render_quadrature(
    tdist: jnp.ndarray,
    query_fn: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    return_extras: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Numerical quadrature rendering of a set of colored Gaussians.

    Returns:
        color_rgba: (N, 4) RGBA
        depth: (N,) expected termination depth
        extras: dict (includes distortion_loss, and optionally debugging fields)
    """
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)

    total_density, avg_colors = query_fn(t_avg)  # (S,), (S,3) or broadcastable
    weights = compute_alpha_weights(total_density * t_delta)

    dist_loss = lossfun_distortion(tdist, weights)

    rendered_color = jnp.sum(weights[..., None] * avg_colors, axis=-2)
    alpha = jnp.sum(weights, axis=-1).reshape(-1, 1)
    expected_termination = jnp.sum(weights * t_avg, axis=-1)

    color_rgba = jnp.concatenate([rendered_color.reshape(-1, 3), alpha], axis=1)
    depth = expected_termination.reshape(-1)

    extras: Dict[str, jnp.ndarray] = {"distortion_loss": dist_loss.reshape(-1)}
    if return_extras:
        extras.update(
            {
                "tdist": tdist,
                "avg_colors": avg_colors,
                "weights": weights,
                "total_density": jnp.sum(total_density * t_delta, axis=-1),
            }
        )

    return color_rgba, depth, extras


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert SH(0th) to rgb-like values (kept as your original behavior)."""
    return sh * C0 + 0.5


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
    colors = jnp.where(inside, params["density"] * params["features"].reshape(1, 3), 0.0)
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
    num_samples: int = 2**16,
    return_extras: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Reference ray tracing using dense quadrature (JAX).
    Inputs are torch tensors; converted to numpy->jax arrays inside.
    """

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
            },
        ),
    )

    def sum_vquery_ellipsoid(t: jnp.ndarray, ray_o: jnp.ndarray, ray_d: jnp.ndarray, p):
        densities, colors = vquery_ellipsoid(t, ray_o, ray_d, p)  # (P,S,1), (P,S,3)
        dens = densities.sum(axis=0)  # (S,1) or (S,)
        col = safe_div(colors.sum(axis=0), dens)  # safe average
        return dens.reshape(-1), col.clip(min=0.0)

    tdist = jnp.linspace(tmin, tmax, num_samples + 1)

    params = {
        "mean": mean.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        "scale": scale.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        "quat": quat.detach().cpu().numpy().astype(np.float64).reshape(-1, 4),
        "density": density.detach().cpu().numpy().astype(np.float64).reshape(-1, 1),
        "features": sh_to_rgb(features).detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
    }
    ray_o = rayo.detach().cpu().numpy().astype(np.float64)
    ray_d = rayd.detach().cpu().numpy().astype(np.float64)

    color_rgba, depth, extras = render_quadrature(
        tdist,
        lambda t: sum_vquery_ellipsoid(t, ray_o, ray_d, params),
        return_extras=return_extras,
    )
    return color_rgba, depth, extras


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
    apply_clip: bool = False,
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
