#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Tools may be installed in the user's shared home, e.g. uv tool install cmake/ninja.
export PATH="$HOME/.local/bin:$PATH"
cmake -S . -B cmake-build-release -G Ninja \
    -DLONGMARCH_WIND3D_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${WIND3D_CUDA_ARCHITECTURES:-native}"
cmake --build cmake-build-release --target wind3d_task -j "${WIND3D_BUILD_JOBS:-8}"
exec cmake-build-release/demo/wind3d/wind3d_task "$@"
