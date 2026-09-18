#!/usr/bin/env python3
"""Compare per-frame render cost of the compute fallback and the CPU backend.

Runs every scene twice -- once through the compute ray tracing fallback, once on
the CPU with the host graphics backend -- and reports the steady state frame
time and the camera ray throughput of each.

The scenes are rewritten to a common resolution, sample count and bounce limit
before the run, so the two backends do the same amount of work and the numbers
are comparable. Warmup frames are dropped, and the median of what remains is
reported, because the first frames include acceleration structure construction.

Both runs use host side timestamps (--profile-cpu-only): GPU timestamps are
Vulkan only, and using the same clock for both keeps the comparison honest.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_rt_fallback import ROOT, read_scene  # noqa: E402

DEFAULT_SCENES = ["cornell_box", "area_light", "point_light", "principled", "specular", "texture",
                  "blender_classroom", "blender_junkshop", "blender_monster"]

# The stage that brackets one whole frame, on both backends.
STAGE = "cpu_ms/render_wall"


def profile(cli: Path, scene: Path, backend: str, pipeline: str, frames: int, warmup: int,
            output: Path) -> tuple[float, float] | None:
    """Runs one configuration and returns (median frame ms, first frame ms)."""
    log = output.with_suffix(".log")
    with log.open("w") as stream:
        code = subprocess.run(
            [str(cli), str(scene), "--backend", backend, "--pipeline", pipeline,
             "--frames", str(frames), "--profile", str(output.with_suffix(".csv")),
             "--profile-cpu-only", "-o", str(output.with_suffix(".png"))],
            stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    if code != 0 or not output.with_suffix(".csv").exists():
        return None
    first, steady = None, []
    with output.with_suffix(".csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["domain"] + "/" + row["stage"] != STAGE:
                continue
            frame, value = int(row["frame"]), float(row["value"])
            if frame == 0:
                first = value
            if frame >= warmup:
                steady.append(value)
    if not steady:
        return None
    return statistics.median(steady), (first if first is not None else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", type=Path, required=True, help="path to demo_sparkium_cli")
    parser.add_argument("--output", type=Path, default=ROOT / "out/backend-profile")
    parser.add_argument("--scenes-list", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--gpu-backend", default="metal", choices=("auto", "metal", "vulkan", "d3d12"))
    parser.add_argument("--gpu-pipeline", default="rt_fallback", choices=("rt_fallback", "ray_query"))
    parser.add_argument("--size", type=int, default=512, help="square resolution both backends render at")
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=12)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2, help="frames to drop before taking the median")
    parser.add_argument("--scenes", type=Path, default=ROOT / "assets/scenes")
    parser.add_argument("--only", choices=("gpu", "cpu"), default=None, help="profile just one backend")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.frames:
        parser.error("warmup must be below the frame count")

    cli = args.cli.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    ray_count = args.size * args.size * args.spp
    rows, results = [], {}

    for name in args.scenes_list:
        scene_json = args.scenes / name / "scene.json"
        if not scene_json.exists():
            continue
        scene = read_scene(scene_json)
        if args.size:
            scene["film"].update(width=args.size, height=args.size)
        scene["renderer"].update(samples_per_dispatch=args.spp, max_bounces=args.bounces)
        staged = args.output / f"{name}.json"
        staged.write_text(json.dumps(scene, indent=2) + "\n")

        entry = {"scene": name, "width": scene["film"]["width"], "height": scene["film"]["height"],
                 "spp": args.spp, "bounces": args.bounces}
        for label, backend, pipeline in (("gpu", args.gpu_backend, args.gpu_pipeline), ("cpu", "host", "cpu")):
            if args.only and args.only != label:
                continue
            measured = profile(cli, staged, backend, pipeline, args.frames, args.warmup,
                               args.output / f"{name}-{label}")
            if measured is None:
                entry[label] = None
                continue
            median_ms, first_ms = measured
            entry[label] = {"median_ms": round(median_ms, 2), "first_ms": round(first_ms, 2),
                            "camera_rays_per_second": round(ray_count / (median_ms / 1000.0) / 1e6, 2)}
        if entry.get("gpu") and entry.get("cpu"):
            entry["cpu_over_gpu"] = round(entry["cpu"]["median_ms"] / entry["gpu"]["median_ms"], 2)
        results[name] = entry
        rows.append(entry)
        print(f"{name} done", flush=True)

    # Table: what the two backends cost for exactly the same work.
    header = (f"{'scene':<18}{'res':>6}{'spp':>5}{'bounce':>7}"
              f"{'GPU ms':>10}{'GPU Mray/s':>12}{'CPU ms':>10}{'CPU Mray/s':>12}{'cpu/gpu':>9}")
    lines = [header, "-" * len(header)]
    for entry in rows:
        gpu, cpu = entry.get("gpu"), entry.get("cpu")
        lines.append(
            f"{entry['scene']:<18}{entry['width']:>6}{entry['spp']:>5}{entry['bounces']:>7}"
            f"{(gpu['median_ms'] if gpu else float('nan')):>10.2f}"
            f"{(gpu['camera_rays_per_second'] if gpu else float('nan')):>12.2f}"
            f"{(cpu['median_ms'] if cpu else float('nan')):>10.2f}"
            f"{(cpu['camera_rays_per_second'] if cpu else float('nan')):>12.2f}"
            f"{entry.get('cpu_over_gpu', float('nan')):>9.2f}")
    table = "\n".join(lines)
    print("\n" + table)

    (args.output / "summary.json").write_text(json.dumps(
        {"gpu_backend": args.gpu_backend, "gpu_pipeline": args.gpu_pipeline, "size": args.size,
         "spp": args.spp, "bounces": args.bounces, "frames": args.frames, "warmup": args.warmup,
         "stage": STAGE, "scenes": results}, indent=2) + "\n")
    (args.output / "table.txt").write_text(table + "\n")
    print(f"\nWrote {args.output / 'table.txt'} and {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
