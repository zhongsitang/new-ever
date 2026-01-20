"""
GaussianRT: Differentiable Volume Renderer for Ellipsoid Primitives

A modern implementation of differentiable ray tracing for volume rendering,
using Slang-RHI for the forward pass and Slang autodiff for gradients.
"""

from .renderer import VolumeRenderFunction, render_volume, VolumeRendererModule

__version__ = "0.1.0"
__all__ = ["VolumeRenderFunction", "render_volume", "VolumeRendererModule"]
