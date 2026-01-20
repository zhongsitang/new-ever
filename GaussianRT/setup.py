"""
Setup script for GaussianRT

Build:
    pip install .

Development install:
    pip install -e .

Requirements:
    - PyTorch with CUDA
    - pybind11
    - Slang SDK (set SLANG_ROOT environment variable)
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get Slang root from environment
SLANG_ROOT = os.environ.get('SLANG_ROOT', '')
if not SLANG_ROOT:
    print("Warning: SLANG_ROOT not set. Build may fail.")
    print("Please set SLANG_ROOT to your Slang SDK installation directory.")

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Source files
cpp_sources = [
    'src/device.cpp',
    'src/acceleration_structure.cpp',
    'src/renderer.cpp',
    'src/volume_renderer.cpp',
    'python/bindings.cpp',
]

cuda_sources = [
    'src/cuda/aabb_builder.cu',
    'src/cuda/initialize_state.cu',
]

# Absolute paths
sources = [str(PROJECT_ROOT / s) for s in cpp_sources + cuda_sources]

# Include directories
include_dirs = [
    str(PROJECT_ROOT / 'src'),
    str(PROJECT_ROOT / 'slang'),
]

if SLANG_ROOT:
    include_dirs.append(os.path.join(SLANG_ROOT, 'include'))

# Library directories
library_dirs = []
if SLANG_ROOT:
    library_dirs.append(os.path.join(SLANG_ROOT, 'lib'))

# Libraries to link
libraries = ['slang', 'slang-rhi']

# Compile flags
extra_compile_args = {
    'cxx': ['-std=c++17', '-O3'],
    'nvcc': [
        '-std=c++17',
        '-O3',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
    ]
}

# Extension module
ext_modules = [
    CUDAExtension(
        name='gaussian_rt._gaussian_rt',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )
]

# Setup
setup(
    name='gaussian_rt',
    version='0.1.0',
    author='GaussianRT Authors',
    description='Differentiable Volume Renderer for Ellipsoid Primitives',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
