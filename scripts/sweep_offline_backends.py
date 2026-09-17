#!/usr/bin/env python3
"""Locate where the CPU and CUDA offline backends start to differ.

`check_offline_backends.py` compares whole scenes. This script bisects the
remaining difference of a single scene along the two axes that explain it:

  * the path depth (`--bounces`): the CPU and the CUDA backend execute the same
    portable core, but they are compiled by different compilers, so an
    expression like `a * b + c` may or may not be contracted into an FMA. The
    resulting 1e-7 relative difference stays invisible until a deep specular or
    glass path amplifies it into a different branch decision.
  * the geometry (`--drop-material`): renders the scene again with every entity
    that uses the named material removed, which shows which part of the scene
    produces those paths.

For every combination the scene is rendered with both offline backends and the
difference metrics of check_offline_backends.py are printed. This is a
diagnostic tool: it reports numbers and never fails a build.

usage:
  scripts/sweep_offline_backends.py --cli build/demo/sparkium_cli/demo_sparkium_cli \
      --scene assets/scenes/texture/scene.json --bounces 1 2 4 8 12 \
      --drop-material glass --drop-material ground --output /results/offline_backends
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_offline_backends import compare, read_scene  # noqa: E402


def main():
    from PIL import Image, ImageChops, ImageStat
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "out/offline-sweep")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--bounces", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--drop-material", action="append", default=[],
                        help="also render with the entities of this material removed")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cli = str(args.cli.resolve())
    base = read_scene(args.scene.resolve() if args.scene.is_absolute() else ROOT / args.scene)
    variants = [("baseline", base)] + [
        (f"drop-{material}", {**base,
                              "entities": [e for e in base["entities"] if e.get("material") != material]})
        for material in args.drop_material
    ]

    results = []
    for name, scene in variants:
        for bounces in args.bounces:
            scene = json.loads(json.dumps(scene))
            scene["film"].update(width=args.size, height=args.size)
            scene["renderer"].update(samples_per_dispatch=args.spp, max_bounces=bounces)
            path = args.output / f"{name}-b{bounces}.json"
            path.write_text(json.dumps(scene, indent=2) + "\n")
            images = {}
            for backend in ("cpu", "cuda"):
                image_path = args.output / f"{name}-b{bounces}-{backend}.png"
                command = [cli, str(path), "--backend", backend, "-o", str(image_path)]
                if args.debug:
                    command.append("--debug")
                with (args.output / f"{name}-b{bounces}-{backend}.log").open("w") as stream:
                    stream.write("$ " + " ".join(command) + "\n\n")
                    stream.flush()
                    run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                if run.returncode:
                    print(f"FAIL {name} bounces={bounces} {backend}: see the log", flush=True)
                    return 1
                with Image.open(image_path) as image:
                    images[backend] = image.convert("RGB")
            metrics = compare(images["cpu"], images["cuda"], ImageChops, ImageStat, np)
            item = {"variant": name, "bounces": bounces, "size": args.size, "spp": args.spp, **metrics}
            results.append(item)
            print(f"{name:22s} bounces={bounces:3d} mae={metrics['png_rgb_mae']:.6f} "
                  f"max={metrics['max_channel_difference']:3d} px_differing={metrics['pixels_differing']:5d} "
                  f"mean_bias={metrics['mean_brightness_bias']:.7f}", flush=True)
    (args.output / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
