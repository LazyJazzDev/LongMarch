#!/usr/bin/env python3
"""Compare the Sparkium CPU backend against the compute ray tracing fallback.

Renders every basic scene twice -- once through the compute fallback, once on
the CPU with the host graphics backend -- and reports the mean absolute pixel
difference of the tone mapped images. Writes both renders, the CLI logs and a
results.json under --output.

The two backends build their own acceleration structures, so rays that tie, or
that a light sampling CDF rounds differently, take different paths. The
difference therefore falls with sample count; --frames lets that be checked,
which is the difference between Monte Carlo divergence and a real defect.

Only the standard library is required. A contact sheet is written when Pillow
is available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENES = ["cornell_box", "area_light", "point_light", "principled", "specular", "texture",
                  "blender_classroom", "blender_junkshop", "blender_monster"]


def decode_png(path: Path) -> tuple[int, int, bytes]:
    """Decode a PNG to tightly packed RGB bytes, without Pillow."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    position, compressed, header = 8, b"", None
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        chunk = data[position + 8:position + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBB", chunk[:10])
        elif kind == b"IDAT":
            compressed += chunk
        position += 12 + length
    width, height, _, color_type = header
    channels = 4 if color_type == 6 else 3
    raw = zlib.decompress(compressed)
    stride = width * channels
    previous, rows, index = bytearray(stride), [], 0
    for _ in range(height):
        filter_type = raw[index]
        index += 1
        line = bytearray(raw[index:index + stride])
        index += stride
        for x in range(stride):
            a = line[x - channels] if x >= channels else 0
            b = previous[x]
            c = previous[x - channels] if x >= channels else 0
            if filter_type == 1:
                line[x] = (line[x] + a) & 255
            elif filter_type == 2:
                line[x] = (line[x] + b) & 255
            elif filter_type == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + predictor) & 255
        rows.append(bytes(line))
        previous = line
    packed = bytearray()
    for row in rows:
        for x in range(0, stride, channels):
            packed += row[x:x + 3]
    return width, height, bytes(packed)


def write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    """Write a tightly packed RGB buffer as a PNG, without Pillow."""
    raw = b"".join(b"\x00" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


def scale_to(rgb: bytes, width: int, height: int, limit: int) -> tuple[int, int, bytes]:
    """Nearest neighbour downscale, which is all a contact sheet needs."""
    if width <= limit and height <= limit:
        return width, height, rgb
    factor = max((width + limit - 1) // limit, (height + limit - 1) // limit)
    new_width, new_height = width // factor, height // factor
    out = bytearray()
    for y in range(new_height):
        row = y * factor * width
        for x in range(new_width):
            index = (row + x * factor) * 3
            out += rgb[index:index + 3]
    return new_width, new_height, bytes(out)


def compare(reference: Path, actual: Path) -> dict:
    width, height, expected = decode_png(reference)
    other_width, other_height, got = decode_png(actual)
    if (width, height) != (other_width, other_height):
        raise ValueError("image sizes differ")
    difference = sum(abs(a - b) for a, b in zip(expected, got)) / len(expected)

    def luminance_mean(pixels: bytes) -> float:
        values = [0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]
                  for i in range(0, len(pixels), 3)]
        return round(statistics.mean(values), 2)

    return {
        "width": width,
        "height": height,
        "mean_absolute_error": round(difference, 3),
        "reference_luminance_mean": luminance_mean(expected),
        "cpu_luminance_mean": luminance_mean(got),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", type=Path, required=True, help="path to demo_sparkium_cli")
    parser.add_argument("--scenes", type=Path, default=ROOT / "assets/scenes", help="directory of scenes")
    parser.add_argument("--output", type=Path, default=ROOT / "out/cpu-backend", help="where to write results")
    parser.add_argument("--frames", type=int, default=1, help="frames per render; more reduces Monte Carlo noise")
    parser.add_argument("--gpu-backend", default="metal", help="backend for the reference render")
    parser.add_argument("--gpu-pipeline", default="rt_fallback", choices=["rt_fallback", "ray_query"])
    parser.add_argument("--max-mae", type=float, default=None, help="fail if any scene exceeds this")
    parser.add_argument("--scenes-list", nargs="*", default=DEFAULT_SCENES, help="scene directory names")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results, failures = {}, 0
    print(f"{'scene':<20}{'mae/255':>9}{'gpu mean':>10}{'cpu mean':>10}")
    for name in args.scenes_list:
        scene = args.scenes / name / "scene.json"
        if not scene.exists():
            print(f"{name:<20}{'skipped':>9}  (no scene.json)")
            continue
        reference = args.output / f"{name}-gpu.png"
        actual = args.output / f"{name}-cpu.png"
        for output, backend, pipeline in ((reference, args.gpu_backend, args.gpu_pipeline), (actual, "host", "cpu")):
            log = (args.output / f"{name}-{backend}.log").open("w")
            code = subprocess.run(
                [str(args.cli), str(scene), "--backend", backend, "--pipeline", pipeline,
                 "--frames", str(args.frames), "-o", str(output)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            log.close()
            if code != 0 or not output.exists():
                print(f"{name:<20}{'FAILED':>9}  ({backend} exited {code})")
                results[name] = {"error": f"{backend} exited {code}"}
                failures += 1
                break
        else:
            metrics = compare(reference, actual)
            results[name] = metrics
            print(f"{name:<20}{metrics['mean_absolute_error']:>9.3f}"
                  f"{metrics['reference_luminance_mean']:>10.2f}{metrics['cpu_luminance_mean']:>10.2f}")
            if args.max_mae is not None and metrics["mean_absolute_error"] > args.max_mae:
                print(f"  {name}: exceeds --max-mae {args.max_mae}")
                failures += 1

    (args.output / "results.json").write_text(json.dumps(
        {"frames": args.frames, "gpu_backend": args.gpu_backend, "gpu_pipeline": args.gpu_pipeline,
         "scenes": results}, indent=2) + "\n")

    # One figure with the reference and the CPU render side by side for every
    # scene, so a regression is visible rather than inferred from a number.
    tile_limit = 240
    tiles = []
    for name in sorted({p.stem.rsplit("-", 1)[0] for p in args.output.glob("*-gpu.png")}):
        for suffix in ("gpu", "cpu"):
            path = args.output / f"{name}-{suffix}.png"
            if path.exists():
                width, height, rgb = decode_png(path)
                tiles.append(scale_to(rgb, width, height, tile_limit))
    if tiles:
        gap, columns = 4, 4
        tile_width = max(t[0] for t in tiles)
        tile_height = max(t[1] for t in tiles)
        rows = (len(tiles) + columns - 1) // columns
        sheet_width = columns * (tile_width + gap) + gap
        sheet_height = rows * (tile_height + gap) + gap
        sheet = bytearray(b"\x18\x18\x18" * (sheet_width * sheet_height))
        for index, (width, height, rgb) in enumerate(tiles):
            left = gap + (index % columns) * (tile_width + gap)
            top = gap + (index // columns) * (tile_height + gap)
            for y in range(height):
                start = ((top + y) * sheet_width + left) * 3
                sheet[start:start + width * 3] = rgb[y * width * 3:(y + 1) * width * 3]
        write_png(args.output / "contact-sheet.png", sheet_width, sheet_height, bytes(sheet))
        print(f"\nWrote {len(tiles)} tiles to {args.output / 'contact-sheet.png'}: for each scene in"
              " alphabetical order, the reference render then the CPU render")

    return 1 if failures else 0

    # Side by side sheets, reference on the left, CPU on the right.
    tiles = []
    for name in written:
        for suffix in ("gpu", "cpu"):
            path = args.output / f"{name}-{suffix}.png"
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((320, 320))
            tiles.append(image)
    if tiles:
        columns = 4
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 330, rows * 330), (24, 24, 24))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * 330 + 5, (index // columns) * 330 + 5))
        sheet.save(args.output / "contact-sheet.png")
        print(f"\nWrote {len(tiles)} tiles and a contact sheet to {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
