#!/usr/bin/env python3
"""Render reproducible Sparkium snapshots; optionally compare with actual hardware RT.

Requires Pillow. All generated scenes, logs and images go to --output.
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


def main():
    from PIL import Image, ImageChops, ImageDraw, ImageStat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "out/rt-fallback")
    parser.add_argument("--scenes", nargs="+", default=list(DEMOS), help="names under assets/scenes")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--bounces", type=int, default=12)
    parser.add_argument("--compare-hardware", action="store_true")
    parser.add_argument("--max-rmse", type=float, help="optional failure threshold for normalized PNG RGB RMSE")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if min(args.size, args.spp, args.bounces) <= 0:
        parser.error("size, spp and bounces must be positive")
    if args.max_rmse is not None and not args.compare_hardware:
        parser.error("--max-rmse requires --compare-hardware")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cli = str(args.cli.resolve())
    scenes = [(name, read_scene(ROOT / "assets/scenes" / name / "scene.json")) for name in args.scenes]
    results, tiles = [], []
    failed = False
    for name, scene in scenes:
        scene["film"].update(width=args.size, height=args.size)
        scene["renderer"].update(samples_per_dispatch=args.spp, max_bounces=args.bounces)
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(scene, indent=2) + "\n")
        item = {"scene": name, "width": args.size, "height": args.size, "spp": args.spp, "bounces": args.bounces}
        pipelines = ["rt_fallback", "ray_tracing"] if args.compare_hardware else ["rt_fallback"]
        images = []
        for pipeline in pipelines:
            stem = f"{name}-{pipeline}"
            png, log = args.output / f"{stem}.png", args.output / f"{stem}.log"
            png.unlink(missing_ok=True)
            command = [cli, str(path), "--pipeline", pipeline, "-o", str(png)]
            if pipeline == "ray_tracing":
                command.append("--require-hardware-rt")
            if args.debug:
                command.append("--debug")
            started = time.monotonic()
            with log.open("w") as stream:
                run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            item[pipeline] = {"returncode": run.returncode, "wall_seconds": time.monotonic() - started}
            if run.returncode:
                failed = True
                print(f"FAIL {stem}: {log}", flush=True)
                break
            with Image.open(png) as image:
                images.append(image.convert("RGB"))
            tile = Image.new("RGB", (args.size, args.size + 32), "#202020")
            tile.paste(images[-1], (0, 32))
            ImageDraw.Draw(tile).text((3, 2), name + "\n" + pipeline, fill="white")
            tiles.append(tile)
        if len(images) == 2:
            stats = ImageStat.Stat(ImageChops.difference(*images))
            item["png_rgb_mae"] = sum(stats.mean) / (3 * 255)
            item["png_rgb_rmse"] = math.sqrt(sum(v * v for v in stats.rms) / 3) / 255
            if args.max_rmse is not None and item["png_rgb_rmse"] > args.max_rmse:
                failed = True
        results.append(item)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(item), flush=True)
    if tiles:
        columns = min(4, len(tiles))
        sheet = Image.new("RGB", (columns * args.size, math.ceil(len(tiles) / columns) * (args.size + 32)), "#202020")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * args.size, (index // columns) * (args.size + 32)))
        sheet.save(args.output / "contact-sheet.png")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
