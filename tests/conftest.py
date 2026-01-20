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

"""pytest configuration file for setting up import paths."""

import sys
from pathlib import Path

# Get the project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent

# Add project root to path so 'splinetracers' can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add tests directory to path so 'utils', 'jaxutil', 'eval_sh' can be imported
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Configure JAX for float64 precision (needed by some tests)
try:
    from jax import config
    config.update("jax_enable_x64", True)
except ImportError:
    pass
