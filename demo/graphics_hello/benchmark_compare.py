"""Compare matching graphics/RHI offscreen demos, serially, using Release builds."""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("cmake-build-release"))
    parser.add_argument("--output", type=Path, default=Path("out/graphics-rhi-headless"))
    parser.add_argument("--frames", type=positive, default=100000)
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--backend", choices=("d3d12", "vulkan", "metal"), action="append")
    args = parser.parse_args()
    backends = args.backend or ["d3d12", "vulkan"]
    suffix = ".exe" if os.name == "nt" else ""
    executables = {
        library: (args.build / "demo" / f"{library}_hello" / f"demo_{library}_hello{suffix}").resolve()
        for library in ("graphics", "rhi")
    }
    for executable in executables.values():
        if not executable.is_file():
            parser.error(f"Build the demo first: {executable}")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # These runs measure Release performance; use separate validation smoke runs.
    env["VK_LOADER_LAYERS_DISABLE"] = "*"
    for key in ("VK_INSTANCE_LAYERS", "VK_LAYER_SETTINGS_PATH", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION"):
        env.pop(key, None)
    results = []
    scenes = ("window", "triangle", "cube", "texture")
    for run in range(args.runs):
        for backend in backends:
            for scene in scenes:
                # Alternate order to reduce systematic warm GPU/order bias.
                libraries = ("graphics", "rhi") if run % 2 == 0 else ("rhi", "graphics")
                for library in libraries:
                    module = scene if library == "graphics" else f"hello_{scene}"
                    command = [
                        str(executables[library]), "--headless", "--module", module,
                        "--backend", backend, "--frames", str(args.frames),
                    ]
                    completed = subprocess.run(command, env=env, capture_output=True, timeout=300)
                    output = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
                    log = args.output / f"{run + 1}-{backend}-{scene}-{library}.log"
                    log.write_text(output, encoding="utf-8")
                    match = re.search(r"(\d+) frames, ([\d.]+) s, ([\d.]+) FPS", output)
                    if completed.returncode or not match:
                        raise RuntimeError(f"Benchmark failed ({completed.returncode}); see {log}")
                    if int(match[1]) != args.frames:
                        raise RuntimeError(f"Unexpected frame count; see {log}")
                    result = dict(
                        run=run + 1, backend=backend, scene=scene, library=library,
                        seconds=float(match[2]), fps=float(match[3]), command=command,
                    )
                    results.append(result)
                    (args.output / "runs.json").write_text(
                        json.dumps(results, indent=2), encoding="utf-8"
                    )
                    print(f"Run {run + 1}: {backend} {scene} {library}: {result['fps']:.2f} FPS", flush=True)
    summary = []
    print("\nBackend | Scene | Graphics FPS | RHI FPS | RHI/Graphics")
    for backend in backends:
        for scene in scenes:
            samples = {
                library: [
                    r["fps"] for r in results
                    if r["backend"] == backend and r["scene"] == scene and r["library"] == library
                ]
                for library in executables
            }
            medians = {library: statistics.median(values) for library, values in samples.items()}
            ratio = medians["rhi"] / medians["graphics"]
            summary.append(dict(backend=backend, scene=scene, medians=medians, samples=samples, ratio=ratio))
            print(f"{backend} | {scene} | {medians['graphics']:.0f} | {medians['rhi']:.0f} | {ratio:.2f}x")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
