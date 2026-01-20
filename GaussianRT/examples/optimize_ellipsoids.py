"""
Optimization example: Fit ellipsoids to a target image.

This example demonstrates:
1. Differentiable rendering for optimization
2. Using Adam optimizer to fit ellipsoid parameters
3. Visualizing the optimization progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from gaussian_rt import VolumeRendererModule


class EllipsoidModel(nn.Module):
    """Learnable ellipsoid volume elements."""

    def __init__(self, num_elements: int, device: torch.device):
        super().__init__()

        # Initialize parameters
        self.positions = nn.Parameter(
            (torch.rand(num_elements, 3, device=device) - 0.5) * 2.0
        )

        # Log-scale for numerical stability
        self.log_scales = nn.Parameter(
            torch.log(torch.rand(num_elements, 3, device=device) * 0.1 + 0.05)
        )

        # Rotation as quaternion (will be normalized during forward)
        self.rotations_raw = nn.Parameter(
            torch.randn(num_elements, 4, device=device)
        )

        # Logit-opacity for sigmoid activation
        self.logit_opacities = nn.Parameter(
            torch.zeros(num_elements, device=device)
        )

        # SH features
        self.features = nn.Parameter(
            torch.zeros(num_elements, 16 * 3, device=device)
        )
        # Initialize DC component with random colors
        self.features.data[:, 0:3] = torch.rand(num_elements, 3, device=device)

    def forward(self):
        """Return processed parameters."""
        scales = torch.exp(self.log_scales)
        rotations = self.rotations_raw / torch.norm(
            self.rotations_raw, dim=1, keepdim=True
        )
        opacities = torch.sigmoid(self.logit_opacities)

        return self.positions, scales, rotations, opacities, self.features


def create_target_image(width: int, height: int, device: torch.device):
    """Create a simple target image (colored gradient)."""
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=device),
        torch.linspace(0, 1, width, device=device),
        indexing='ij'
    )

    # Create a colorful pattern
    r = 0.5 + 0.5 * torch.sin(x * 6.28 * 2)
    g = 0.5 + 0.5 * torch.sin(y * 6.28 * 2)
    b = 0.5 + 0.5 * torch.sin((x + y) * 6.28)

    target = torch.stack([r, g, b, torch.ones_like(r)], dim=-1)
    return target


def main():
    # Configuration
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    num_elements = 500
    width, height = 128, 128
    num_iterations = 500
    lr = 0.01

    # Create renderer
    print("Initializing renderer...")
    renderer = VolumeRendererModule(
        device_index=0,
        t_min=0.1,
        t_max=10.0,
        max_samples=32,
        sh_degree=0,
    )

    # Create learnable model
    print(f"Creating {num_elements} ellipsoids...")
    model = EllipsoidModel(num_elements, device)

    # Create target image
    target = create_target_image(width, height, device)

    # Camera
    camera = {
        'position': [0, 0, 3],
        'forward': [0, 0, -1],
        'up': [0, 1, 0],
        'right': [1, 0, 0],
        'focal_length': 1.0,
        'sensor_width': 1.0,
    }

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

    # Training loop
    print(f"\nOptimizing for {num_iterations} iterations...")
    losses = []

    pbar = tqdm(range(num_iterations))
    for i in pbar:
        optimizer.zero_grad()

        # Get current parameters
        positions, scales, rotations, opacities, features = model()

        # Render
        colors, depths, distortions = renderer.render_image(
            positions, scales, rotations, opacities, features,
            camera, width, height
        )

        # Compute losses
        color_loss = nn.functional.mse_loss(colors, target)
        distortion_loss = distortions.mean() * 0.01  # Regularization

        loss = color_loss + distortion_loss

        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")

    # Save results
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Target
        axes[0, 0].imshow(target[:, :, :3].cpu().numpy())
        axes[0, 0].set_title('Target')
        axes[0, 0].axis('off')

        # Final render
        with torch.no_grad():
            positions, scales, rotations, opacities, features = model()
            final_colors, final_depths, _ = renderer.render_image(
                positions, scales, rotations, opacities, features,
                camera, width, height
            )

        axes[0, 1].imshow(np.clip(final_colors[:, :, :3].cpu().numpy(), 0, 1))
        axes[0, 1].set_title('Final Render')
        axes[0, 1].axis('off')

        # Difference
        diff = torch.abs(final_colors - target)[:, :, :3].mean(dim=-1)
        axes[0, 2].imshow(diff.cpu().numpy(), cmap='hot')
        axes[0, 2].set_title('Absolute Difference')
        axes[0, 2].axis('off')

        # Alpha
        axes[1, 0].imshow(final_colors[:, :, 3].cpu().numpy(), cmap='gray')
        axes[1, 0].set_title('Alpha')
        axes[1, 0].axis('off')

        # Depth
        axes[1, 1].imshow(final_depths.cpu().numpy(), cmap='viridis')
        axes[1, 1].set_title('Depth')
        axes[1, 1].axis('off')

        # Loss curve
        axes[1, 2].plot(losses)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Training Loss')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig('optimization_result.png', dpi=150)
        print("Saved optimization_result.png")

    except ImportError:
        print("Matplotlib not available, skipping visualization")

    print("\nDone!")


if __name__ == '__main__':
    main()
