"""Alternate baseline/RHI UI runs; report warmed-up frame-loop throughput.

Both binaries must include demo/frame_benchmark.h instrumentation. On a headless
Linux host run this entire script under one xvfb-run display, selecting Vulkan.
PNG comparisons require Pillow; benchmark collection uses only the standard library.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["d3d12", "vulkan", "metal"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6120)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.frames <= 120 or args.repeats < 1:
        parser.error("frames must exceed 120 and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    cases = {
        "2048": ("2048", []),
        "gol-128": ("gol", ["128", "128", "--random", "0.35"]),
        "gol-256": ("gol", ["256", "256", "--random", "0.35"]),
    }
    roots = {"baseline": args.baseline_root, "rhi": args.candidate_root}
    rows = []
    env = dict(os.environ, LONGMARCH_DEMO_BENCHMARK="1", LONGMARCH_DEMO_OFFSCREEN="1", LONGMARCH_DEMO_SEED="1")
    if os.name != "nt" and args.backend == "vulkan":
        env["LONGMARCH_WINDOW_SYSTEM"] = "x11"
        env.pop("WAYLAND_DISPLAY", None)
    for case, (demo, options) in cases.items():
        for repeat in range(args.repeats):
            # Alternate which implementation runs first to reduce ordering bias.
            order = ["baseline", "rhi"] if repeat % 2 == 0 else ["rhi", "baseline"]
            for variant in order:
                executable = roots[variant] / demo / ("demo_" + demo + suffix)
                stem = f"{case}-{variant}-{repeat}"
                command = [str(executable.resolve()), "--backend", args.backend,
                           "--frames", str(args.frames), *options]
                if repeat == 0:
                    command += ["--screenshot", str((args.output / (stem + ".png")).resolve())]
                result = subprocess.run(command, env=env, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=180)
                log = result.stdout + result.stderr
                (args.output / (stem + ".log")).write_text(log, encoding="utf-8")
                match = re.search(r"DEMO_BENCHMARK (\{[^\n]+\})", log)
                if result.returncode or not match:
                    raise RuntimeError(f"{stem} failed ({result.returncode}); see its log")
                row = dict(case=case, variant=variant, repeat=repeat, **json.loads(match[1]))
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = []
    for case in cases:
        medians = {variant: statistics.median(row["fps"] for row in rows
                                            if row["case"] == case and row["variant"] == variant)
                   for variant in roots}
        summary.append(dict(case=case, median_fps=medians,
                            change_percent=100 * (medians["rhi"] / medians["baseline"] - 1)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
