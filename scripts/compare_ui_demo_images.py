"""Compare paired *-baseline*.png / *-rhi*.png demo captures (requires Pillow)."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--max-difference", type=int, default=0)
    args = parser.parse_args()
    passed = True
    for directory in args.directories:
        results = []
        for baseline in sorted(directory.glob("*-baseline*.png")):
            candidate = baseline.with_name(baseline.name.replace("-baseline", "-rhi"))
            with Image.open(baseline) as a, Image.open(candidate) as b:
                if a.size != b.size:
                    raise ValueError(f"Image size mismatch: {baseline.name}: {a.size} != {b.size}")
                delta = ImageChops.difference(a.convert("RGBA"), b.convert("RGBA"))
                maximum = max(high for low, high in delta.getextrema())
                channels = delta.split()
                per_pixel = channels[0]
                for channel in channels[1:]:
                    per_pixel = ImageChops.lighter(per_pixel, channel)
                changed = a.width * a.height - per_pixel.histogram()[0]
                results.append(dict(baseline=baseline.name, candidate=candidate.name, size=a.size,
                                    max_difference=maximum, mean_difference=ImageStat.Stat(delta).mean,
                                    changed_pixels=changed, passed=maximum <= args.max_difference))
                passed = passed and maximum <= args.max_difference
        if not results:
            raise ValueError(f"No image pairs in {directory}")
        (directory / "images.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(directory, json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
