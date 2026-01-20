import torch
import numpy as np

from splinetracers import fast_ellipsoid_splinetracer

device = torch.device('cuda')


def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )


def test_grad_check(N, density_multi):
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
        color = fast_ellipsoid_splinetracer.trace_rays(
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
    

if __name__ == "__main__":
    test_grad_check(1, 1)