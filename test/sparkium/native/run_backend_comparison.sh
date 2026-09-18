#!/usr/bin/env bash
# Renders every JSON scene with the Vulkan reference pipeline and with both
# native backends, then compares the images. Artifacts land in $OUT_DIR.
#
# Usage: run_backend_comparison.sh [output-directory] [frames] [scene...]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CLI="${SPARKIUM_CLI:-$ROOT/build/demo/sparkium_cli/demo_sparkium_cli}"
OUT_DIR="${1:-$ROOT/build/native_comparison}"
FRAMES="${2:-1}"
shift 2 2>/dev/null || shift $#
SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  mapfile -t SCENES < <("$CLI" --list)
fi

mkdir -p "$OUT_DIR/images" "$OUT_DIR/logs" "$OUT_DIR/diffs"
SUMMARY="$OUT_DIR/summary.csv"
echo "scene,pair,mean_abs,psnr_db,fraction_gt_4,block16_mean,block16_max,mean_signed,status" > "$SUMMARY"
echo "scene,backend,frames,wall_seconds,exit_status" > "$OUT_DIR/timings.csv"

render() {
  local scene="$1" name="$2" backend="$3" image="$4"
  local log="$OUT_DIR/logs/${name}_${backend}.log"
  local start=$SECONDS
  if [ "$backend" = "reference" ]; then
    "$CLI" "$scene" -o "$image" --frames "$FRAMES" > "$log" 2>&1
  else
    "$CLI" "$scene" -o "$image" --frames "$FRAMES" --backend "$backend" > "$log" 2>&1
  fi
  local status=$?
  local elapsed=$((SECONDS - start))
  echo "elapsed_seconds=$elapsed" >> "$log"
  echo "exit_status=$status" >> "$log"
  # Wall clock for reference only: the four containers share one GPU.
  echo "$name,$backend,$FRAMES,$elapsed,$status" >> "$OUT_DIR/timings.csv"
  return $status
}

compare() {
  local name="$1" pair="$2" reference="$3" candidate="$4"
  local json="$OUT_DIR/diffs/${name}_${pair}.json"
  python3 "$ROOT/test/sparkium/native/compare_images.py" "$reference" "$candidate" \
    --diff "$OUT_DIR/diffs/${name}_${pair}.png" --json "$json" \
    > "$OUT_DIR/logs/compare_${name}_${pair}.log" 2>&1
  if [ ! -s "$json" ]; then
    echo "$name,$pair,,,,,,,compare-failed" >> "$SUMMARY"
    return 1
  fi
  python3 - "$json" "$name" "$pair" >> "$SUMMARY" <<'PYTHON'
import json, sys
stats = json.load(open(sys.argv[1]))
print("{},{},{:.4f},{:.2f},{:.8f},{:.4f},{:.3f},{:+.4f},ok".format(
    sys.argv[2], sys.argv[3], stats["mean_abs"], stats["psnr_db"], stats["fraction_gt_4"],
    stats["block_mean_abs"], stats["block_max_abs"], stats["mean_signed"]))
PYTHON
}

overall=0
for scene in "${SCENES[@]}"; do
  name="$(basename "$(dirname "$scene")")"
  echo "=== $name"
  declare -A images=()
  for backend in reference cpu cuda; do
    image="$OUT_DIR/images/${name}_${backend}.png"
    if render "$scene" "$name" "$backend" "$image"; then
      images[$backend]="$image"
      echo "  $backend: ok"
    else
      echo "  $backend: FAILED (see $OUT_DIR/logs/${name}_${backend}.log)"
      echo "$name,$backend,,,,,,,render-failed" >> "$SUMMARY"
      overall=1
    fi
  done
  if [ -n "${images[reference]:-}" ] && [ -n "${images[cpu]:-}" ]; then
    compare "$name" "cpu_vs_reference" "${images[reference]}" "${images[cpu]}" || overall=1
  fi
  if [ -n "${images[reference]:-}" ] && [ -n "${images[cuda]:-}" ]; then
    compare "$name" "cuda_vs_reference" "${images[reference]}" "${images[cuda]}" || overall=1
  fi
  if [ -n "${images[cpu]:-}" ] && [ -n "${images[cuda]:-}" ]; then
    compare "$name" "cuda_vs_cpu" "${images[cpu]}" "${images[cuda]}" || overall=1
  fi
done

echo
python3 - "$SUMMARY" <<'PYTHON'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
for row in rows:
    print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
PYTHON
exit $overall
