#!/usr/bin/env python3
"""
GaussianRT - Setup script for Python package

This script builds and installs the GaussianRT Python extension.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake-based extension."""

    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake."""

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Ensure output directory exists
        os.makedirs(extdir, exist_ok=True)

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DGAUSSIANRT_BUILD_TESTS=OFF",
            "-DGAUSSIANRT_BUILD_PYTHON=ON",
        ]

        # Build type
        cfg = "Debug" if self.debug else "Release"
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        # Slang SDK path
        slang_path = os.environ.get("SLANG_SDK_PATH")
        if slang_path:
            cmake_args.append(f"-DSLANG_SDK_PATH={slang_path}")

        # Build arguments
        build_args = ["--config", cfg, "-j"]

        # Create build directory
        build_dir = os.path.join(ext.sourcedir, "build_pip")
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_dir
        )


# Read long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")


setup(
    name="gaussian_rt",
    version="1.0.0",
    author="GaussianRT Authors",
    description="Differentiable 3D Gaussian Ray Tracing Renderer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/gaussian_rt",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("gaussian_rt_ext")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    keywords="gaussian splatting, ray tracing, differentiable rendering, 3d reconstruction",
)
