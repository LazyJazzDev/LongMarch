#!/bin/bash
# Run the headless Wind3D sweep on one GPU and keep its output with the run.
# Usage: run_sweep.sh OUTPUT_DIR [demo_wind3d_sweep options...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
binary="${WIND3D_SWEEP:-$root/cmake-build-release/demo/wind3d/wind3d_sweep}"
output="$1"
shift
mkdir -p "$output"
{
  echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "args=$*"
} | tee -a "$output/environment.txt"
mkdir -p "$output/checkpoint"
# A preempted job reruns this script; the sweep resumes from the checkpoint.
"$binary" --output "$output/results.csv" --checkpoint "$output/checkpoint" "$@" 2>&1 | tee -a "$output/log.txt"
