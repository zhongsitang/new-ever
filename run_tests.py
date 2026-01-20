#!/usr/bin/env python3
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

"""Test runner for splinetracers.

Usage:
    # Run all tests
    python run_tests.py

    # Run specific test file
    python run_tests.py tests/grad_check.py

    # Run with verbose output
    python run_tests.py -v

    # Run with pytest (if installed)
    python run_tests.py --pytest
"""

import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
TESTS_DIR = PROJECT_ROOT / "tests"

# Add project root and tests to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))

# Set working directory to project root
os.chdir(PROJECT_ROOT)


def run_with_absltest(test_files=None, verbose=False):
    """Run tests using absltest (default)."""
    from absl.testing import absltest

    # Change to tests directory for relative imports
    os.chdir(TESTS_DIR)

    if test_files:
        # Run specific test files
        for test_file in test_files:
            test_path = Path(test_file)
            if test_path.exists():
                print(f"\n{'='*60}")
                print(f"Running: {test_file}")
                print('='*60)
                # Import and run the test module
                module_name = test_path.stem
                exec(open(test_path).read())
    else:
        # Run all tests
        test_files = list(TESTS_DIR.glob("*.py"))
        test_files = [f for f in test_files if f.name not in ("__init__.py", "conftest.py")]

        for test_file in sorted(test_files):
            if test_file.name.startswith("test_") or test_file.name.endswith("_test.py"):
                print(f"\n{'='*60}")
                print(f"Running: {test_file.name}")
                print('='*60)
                try:
                    exec(open(test_file).read())
                except Exception as e:
                    print(f"Error in {test_file.name}: {e}")


def run_with_pytest(test_files=None, verbose=False):
    """Run tests using pytest."""
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        return 1

    args = [str(TESTS_DIR)]
    if test_files:
        args = [str(f) for f in test_files]
    if verbose:
        args.append("-v")

    return pytest.main(args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run splinetracers tests")
    parser.add_argument("test_files", nargs="*", help="Specific test files to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--pytest", action="store_true", help="Use pytest instead of absltest")
    parser.add_argument("--list", action="store_true", help="List available tests")

    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for f in sorted(TESTS_DIR.glob("*.py")):
            if f.name not in ("__init__.py", "conftest.py"):
                print(f"  {f.name}")
        return 0

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tests dir: {TESTS_DIR}")
    print(f"Python path: {sys.path[:3]}...")

    if args.pytest:
        return run_with_pytest(args.test_files, args.verbose)
    else:
        run_with_absltest(args.test_files, args.verbose)
        return 0


if __name__ == "__main__":
    sys.exit(main())
