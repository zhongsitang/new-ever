"""
Basic rendering example for GaussianRT.

This example shows how to:
1. Create ellipsoid volume elements
2. Set up a camera
3. Render an image
4. Compute gradients for optimization
"""

import torch
import numpy as np
from gaussian_rt import VolumeRendererModule


def create_random_ellipsoids(num_elements: int, device: torch.device):
    """Create random ellipsoid volume elements."""
    # Random positions in a unit cube
    positions = (torch.rand(num_elements, 3, device=device) - 0.5) * 2.0

    # Random scales (small ellipsoids)
    scales = torch.rand(num_elements, 3, device=device) * 0.1 + 0.05

    # Random rotations (unit quaternions)
    rotations = torch.randn(num_elements, 4, device=device)
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)

    # Random opacities
    opacities = torch.rand(num_elements, device=device) * 0.5 + 0.5

    # Random SH coefficients (16 RGB triplets)
    # DC component (first 3) controls base color
    features = torch.zeros(num_elements, 16 * 3, device=device)
    features[:, 0:3] = torch.rand(num_elements, 3, device=device)  # Base color

    return positions, scales, rotations, opacities, features


def create_camera(position, look_at, up=[0, 1, 0]):
    """Create camera parameters."""
    position = np.array(position, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    return {
        'position': position.tolist(),
        'forward': forward.tolist(),
        'up': up.tolist(),
        'right': right.tolist(),
        'focal_length': 1.0,
        'sensor_width': 1.0,
    }


def main():
    # Setup
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    # Create renderer
    print("Initializing renderer...")
    renderer = VolumeRendererModule(
        device_index=0,
        t_min=0.1,
        t_max=10.0,
        max_samples=64,
        sh_degree=0,  # Use only DC component for simplicity
    )

    # Create volume elements
    print("Creating ellipsoids...")
    num_elements = 1000
    positions, scales, rotations, opacities, features = create_random_ellipsoids(
        num_elements, device
    )

    # Enable gradients for optimization
    positions.requires_grad_(True)
    scales.requires_grad_(True)
    rotations.requires_grad_(True)
    opacities.requires_grad_(True)
    features.requires_grad_(True)

    # Create camera
    camera = create_camera(
        position=[0, 0, 3],
        look_at=[0, 0, 0],
    )

    # Render settings
    width, height = 256, 256

    # Render image
    print(f"Rendering {width}x{height} image...")
    colors, depths, distortions = renderer.render_image(
        positions, scales, rotations, opacities, features,
        camera, width, height
    )

    print(f"  Color shape: {colors.shape}")
    print(f"  Depth shape: {depths.shape}")
    print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"  Depth range: [{depths.min():.3f}, {depths.max():.3f}]")

    # Compute a simple loss and gradients
    print("\nComputing gradients...")
    target = torch.ones_like(colors) * 0.5  # Target: gray image
    loss = torch.nn.functional.mse_loss(colors, target)
    loss.backward()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Position gradient norm: {positions.grad.norm():.6f}")
    print(f"  Scale gradient norm: {scales.grad.norm():.6f}")
    print(f"  Opacity gradient norm: {opacities.grad.norm():.6f}")

    # Save image (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Color image
        img = colors[:, :, :3].detach().cpu().numpy()
        axes[0].imshow(np.clip(img, 0, 1))
        axes[0].set_title('Rendered Color')
        axes[0].axis('off')

        # Alpha channel
        alpha = colors[:, :, 3].detach().cpu().numpy()
        axes[1].imshow(alpha, cmap='gray')
        axes[1].set_title('Alpha')
        axes[1].axis('off')

        # Depth
        depth = depths.detach().cpu().numpy()
        axes[2].imshow(depth, cmap='viridis')
        axes[2].set_title('Depth')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('render_output.png', dpi=150)
        print("\nSaved render_output.png")

    except ImportError:
        print("\nMatplotlib not available, skipping image save")

    print("\nDone!")


if __name__ == '__main__':
    main()
