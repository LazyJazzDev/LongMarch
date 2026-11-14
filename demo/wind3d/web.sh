#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p out/wind3d-web
(
flock 9
cmake -S . -B cmake-build-release -G Ninja -DLONGMARCH_WIND3D_STANDALONE=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${WIND3D_CUDA_ARCHITECTURES:-native}"
cmake --build cmake-build-release --target wind3d_web -j 8
) 9>out/wind3d-web/build.lock
exec cmake-build-release/demo/wind3d/wind3d_web "$@"
