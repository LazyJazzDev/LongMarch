#!/usr/bin/env python3
"""Render the bundled Sparkium scenes with the CPU, CUDA and reference backends.

The script is the regression harness of the offline (CPU/CUDA) renderers: every
scene is rendered at identical resolution, samples per dispatch, bounce limit and
frame count by

  * --backend cpu      (host worker threads)
  * --backend cuda     (CUDA kernels)
  * --pipeline rt_fallback --backend vulkan   (reference software BVH pipeline)

and the resulting PNGs are compared pairwise. All scenes, logs, PNGs and the
comparison numbers go to --output; results.json is rewritten after every scene.

Two families of numbers are reported per pair of images:

  * png_rgb_mae / png_rgb_rmse / max_channel_difference: pixel metrics. The CPU
    and CUDA backends draw their random numbers from the same Sobol rows as the
    reference, but the HLSL the reference runs and the transcribed portable core
    are different instructions, so occasional branching decisions (a glass
    Fresnel test, a rough specular sample next to the horizon) differ and the
    whole remaining path of that sample diverges. In scenes with bright clamped
    light sources that shows up as Monte Carlo noise at the same scale as
    rendering the scene twice, which is why these metrics are only bounded
    loosely here.
  * mean_brightness_bias: the difference of the two image means. Averaging over
    the whole frame cancels Monte Carlo noise, so this is the metric that catches
    a systematic shading error (a wrong texture, a missing BSDF term, a wrong
    light sampler) and it is gated tightly.

--noise-study-spp additionally renders every scene at a higher sample count and
stores the same numbers under "noise_study", which makes the noise floor of each
scene explicit (see /results/REPORT.md).

requires Pillow (same dependency as scripts/check_rt_fallback.py).
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ("cornell_box", "area_light", "point_light", "principled", "specular", "texture")


def absolute_assets(value, directory, texture_slots=False):
    """Rewrite every asset path to an absolute one; the scenes are rendered from
    the output directory, so relative paths would no longer resolve."""
    if isinstance(value, dict):
        return {
            key: str((directory / item).resolve())
            if isinstance(item, str) and (key == "path" or texture_slots)
            else absolute_assets(item, directory, key == "textures")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [absolute_assets(item, directory) for item in value]
    return value


def read_scene(path):
    return absolute_assets(json.loads(path.read_text()), path.parent)


def parse_override(text, option):
    """`key=value` with a JSON value, so numbers, bools and strings all work."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise ValueError(f"{option} expects key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def compare(a, b, ImageChops, ImageStat, np):
    difference = ImageChops.difference(a, b)
    stats = ImageStat.Stat(difference)
    array = np.asarray(difference).astype(int)
    mean_a = float(np.asarray(a).mean())
    mean_b = float(np.asarray(b).mean())
    return {
        "png_rgb_mae": sum(stats.mean) / (3 * 255),
        "png_rgb_rmse": math.sqrt(sum(v * v for v in stats.rms) / 3) / 255,
        "max_channel_difference": int(array.max()),
        "pixels_differing": int((array.max(axis=2) > 0).sum()),
        "pixels_differing_fraction": float((array.max(axis=2) > 0).mean()),
        # Image-mean difference of the two renderers. Unlike the MAE this is not
        # dominated by Monte Carlo noise, so it is what catches a systematic
        # shading error (a wrong texture, a missing term, ...).
        "mean_brightness_bias": abs(mean_a - mean_b) / 255,
        "mean_brightness_ratio": mean_a / mean_b if mean_b else float("inf"),
    }


def render_and_compare(args, name, scene, size, spp, bounces, frames, tag, image_ops):
    """Renders one scene with every requested backend and compares the results.

    Returns (item, tiles, failed) where item holds the run descriptions and the
    pairwise comparison numbers; the PNGs and logs are written next to them.
    """
    Image, ImageChops, ImageDraw, ImageStat, np = image_ops
    scene = json.loads(json.dumps(scene))
    scene["film"].update(width=size, height=size)
    scene["renderer"].update(samples_per_dispatch=spp, max_bounces=bounces)
    scene["film"].update(args.film_overrides)
    scene["renderer"].update(args.renderer_overrides)
    path = args.output / f"{name}{tag}.json"
    path.write_text(json.dumps(scene, indent=2) + "\n")
    item = {"scene": name, "width": size, "height": size, "spp": spp, "bounces": bounces,
            "frames": frames, "scene_json": str(path), "runs": {}, "comparisons": {}}
    frame_arguments = ["--frames", str(frames)] if frames != 1 else []
    runs = [(backend, [args.cli, str(path), "--backend", backend, *frame_arguments, "-o",
                       str(args.output / f"{name}-{backend}{tag}.png")]) for backend in args.backends]
    if not args.skip_reference:
        runs.append(("reference", [args.cli, str(path), "--pipeline", args.reference_pipeline, "--backend",
                                   args.reference_backend, *frame_arguments, "-o",
                                   str(args.output / f"{name}-reference{tag}.png")]))
    images, tiles, failed = {}, [], False
    for label, command in runs:
        if args.debug:
            command.append("--debug")
        log = args.output / f"{name}-{label}{tag}.log"
        started = time.monotonic()
        with log.open("w") as stream:
            stream.write("$ " + " ".join(command) + "\n\n")
            stream.flush()
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        elapsed = time.monotonic() - started
        item["runs"][label] = {"command": command, "returncode": run.returncode, "wall_seconds": elapsed,
                               "log": str(log)}
        if run.returncode:
            failed = True
            print(f"FAIL {name}{tag}/{label}: see {log}", flush=True)
            break
        with Image.open(args.output / f"{name}-{label}{tag}.png") as image:
            images[label] = image.convert("RGB")
        tile = Image.new("RGB", (size, size + 32), "#202020")
        tile.paste(images[label], (0, 32))
        ImageDraw.Draw(tile).text((3, 2), f"{name}{tag}\n{label}\n{spp} spp", fill="white")
        tiles.append(tile)
    labels = list(images)
    for index, first in enumerate(labels):
        for second in labels[index + 1:]:
            key = f"{first}-vs-{second}"
            item["comparisons"][key] = compare(images[first], images[second], ImageChops, ImageStat, np)
            if args.diff_images:
                difference = ImageChops.difference(images[first], images[second]).point(lambda value: value * 4)
                difference.save(args.output / f"{name}-diff-{first}-{second}{tag}.png")

    # Gates. Both of them only run when the reference rendered successfully.
    if "reference" in images:
        for backend in args.backends:
            key = f"{backend}-vs-reference"
            if key not in item["comparisons"]:
                continue
            comparison = item["comparisons"][key]
            if backend == "cuda" and args.max_cuda_reference_mae is not None:
                if comparison["png_rgb_mae"] > args.max_cuda_reference_mae:
                    failed = True
                    print(f"FAIL {name}{tag}: cuda MAE {comparison['png_rgb_mae']:.4f} against the reference "
                          f"exceeds {args.max_cuda_reference_mae}", flush=True)
            if args.max_reference_mae is not None:
                if comparison["png_rgb_mae"] > args.max_reference_mae:
                    failed = True
                    print(f"FAIL {name}{tag}: {backend} MAE {comparison['png_rgb_mae']:.4f} against the reference "
                          f"exceeds {args.max_reference_mae}", flush=True)
            if args.max_mean_bias is not None:
                if comparison["mean_brightness_bias"] > args.max_mean_bias:
                    failed = True
                    print(f"FAIL {name}{tag}: {backend} image mean differs from the reference by "
                          f"{comparison['mean_brightness_bias']:.5f}, more than {args.max_mean_bias}", flush=True)
    return item, tiles, failed


def main():
    from PIL import Image, ImageChops, ImageDraw, ImageStat
    import numpy as np

    image_ops = (Image, ImageChops, ImageDraw, ImageStat, np)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "out/offline-backends")
    parser.add_argument("--scenes", nargs="+", default=list(DEMOS), help="names under assets/scenes")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--bounces", type=int, default=12)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--reference-pipeline", default="rt_fallback")
    parser.add_argument("--reference-backend", default="vulkan")
    parser.add_argument("--backends", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--noise-study-spp", type=int, default=0,
                        help="also render every scene at this sample count and store the same numbers")
    parser.add_argument("--noise-study-size", type=int, help="resolution of the noise study (default: --size)")
    parser.add_argument("--max-reference-mae", type=float,
                        help="fail when the cpu/cuda MAE against the reference exceeds this")
    parser.add_argument("--max-cuda-reference-mae", type=float, default=0.005,
                        help="CUDA has to reproduce the HLSL reference up to Monte Carlo noise; the measured "
                             "32 spp residual of the noisiest bundled scene is 0.0027")
    parser.add_argument("--max-mean-bias", type=float, default=0.002,
                        help="fail when the mean pixel value of a backend differs from the reference by more "
                             "than this; Monte Carlo noise averages out of the image mean, so this is the gate "
                             "that catches systematic shading errors")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--diff-images", action="store_true",
                        help="also write the pairwise differences amplified by four as PNGs")
    parser.add_argument("--film", action="append", default=[], metavar="KEY=VALUE",
                        help="override a scene film field, e.g. --film view_transform=standard")
    parser.add_argument("--renderer", action="append", default=[], metavar="KEY=VALUE",
                        help="override a scene renderer field, e.g. --renderer alpha_shadow=true")
    args = parser.parse_args()
    if min(args.size, args.spp, args.bounces, args.frames) <= 0:
        parser.error("size, spp, bounces and frames must be positive")
    if args.noise_study_spp and args.noise_study_spp <= 0:
        parser.error("--noise-study-spp must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cli = str(args.cli.resolve())
    noise_size = args.noise_study_size or args.size
    args.film_overrides = dict(parse_override(text, "--film") for text in args.film)
    args.renderer_overrides = dict(parse_override(text, "--renderer") for text in args.renderer)

    def save_sheet(tile_list, filename):
        if not tile_list:
            return
        columns = min(3, len(tile_list))
        rows = math.ceil(len(tile_list) / columns)
        tile_width = max(tile.width for tile in tile_list)
        tile_height = max(tile.height for tile in tile_list)
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202020")
        for index, tile in enumerate(tile_list):
            sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
        sheet.save(args.output / filename)

    results, tiles, study_tiles, failed = [], [], [], False
    for name in args.scenes:
        scene = read_scene(ROOT / "assets/scenes" / name / "scene.json")
        item, scene_tiles, scene_failed = render_and_compare(args, name, scene, args.size, args.spp, args.bounces,
                                                            args.frames, "", image_ops)
        failed = failed or scene_failed
        tiles += scene_tiles
        if args.noise_study_spp:
            study, scene_study_tiles, study_failed = render_and_compare(
                args, name, scene, noise_size, args.noise_study_spp, args.bounces, args.frames,
                f"-n{args.noise_study_spp}", image_ops)
            failed = failed or study_failed
            study_tiles += scene_study_tiles
            item["noise_study"] = study
        results.append(item)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in item.items() if k != "runs"}), flush=True)
    save_sheet(tiles, "contact-sheet.png")
    save_sheet(study_tiles, "contact-sheet-noise.png")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
