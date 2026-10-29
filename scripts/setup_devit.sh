#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
if [ ! -d vendor/devit/.git ]; then
  git clone https://github.com/mlzxy/devit.git vendor/devit
  git -C vendor/devit checkout 9cab390fbe78b3bc8d62c224411dce1c71056cf9
fi
test "$(git -C vendor/devit rev-parse HEAD)" = 9cab390fbe78b3bc8d62c224411dce1c71056cf9
python3 -m venv --system-site-packages .venv-devit
.venv-devit/bin/python -m pip install -r requirements-devit.txt
.venv-devit/bin/python scripts/patch_devit_compat.py
# Build CPU-only auxiliary extension. Standard box ROIAlign/NMS still use
# the existing CUDA torchvision operators. No unsupported rotated/deform CUDA ops.
(cd vendor/devit && PATH="$PWD/../../.venv-devit/bin:$PATH" CUDA_VISIBLE_DEVICES='' MAX_JOBS=4 ../../.venv-devit/bin/python setup.py build_ext --inplace)
.venv-devit/bin/python scripts/download_devit_weights.py
