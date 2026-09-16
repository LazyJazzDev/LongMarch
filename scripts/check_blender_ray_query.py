#!/usr/bin/env python3
"""Render complete Blender scenes with native ray queries and Metal validation.

Requires Pillow and the Blender LFS asset snapshot. Saves images, per-frame native
query counters, logs, and results; any failed render or silent fallback fails.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

from check_rt_fallback import ROOT, read_scene


def main():
    from PIL import Image, ImageDraw, ImageStat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/blender-ray-query')
    parser.add_argument('--scenes', nargs='+', default=['blender_monster', 'blender_classroom', 'blender_junkshop'])
    parser.add_argument('--pipeline', choices=('ray_query', 'auto', 'scene'), default='ray_query')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--spp', type=int, default=32, help='samples per frame')
    parser.add_argument('--frames', type=int, default=2)
    parser.add_argument('--bounces', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if min(args.size, args.spp, args.frames, args.bounces, args.timeout) <= 0:
        parser.error('size, spp, frames, bounces and timeout must be positive')
    cli, output = args.cli.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MTL_DEBUG_LAYER='1', MTL_SHADER_VALIDATION='1')
    results, tiles = [], []
    for name in args.scenes:
        scene = read_scene(ROOT / 'assets/scenes' / name / 'scene.json')
        scene['film'].update(width=args.size, height=args.size)
        scene['renderer'].update(samples_per_dispatch=args.spp, max_bounces=args.bounces)
        path, png, log, profile = [output / (name + suffix) for suffix in ('.json', '.png', '.log', '.csv')]
        path.write_text(json.dumps(scene, indent=2) + '\n')
        png.unlink(missing_ok=True)
        profile.unlink(missing_ok=True)
        command = [str(cli), str(path), '--backend', 'metal',
                   '--frames', str(args.frames), '--debug', '--profile', str(profile),
                   '--profile-cpu-only', '-o', str(png)]
        if args.pipeline != 'scene':
            command.extend(['--pipeline', args.pipeline])
        started = time.monotonic()
        with log.open('w') as stream:
            try:
                rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
        result = dict(scene=name, pipeline=args.pipeline, size=args.size, spp_per_frame=args.spp,
                      frames=args.frames, bounces=args.bounces, returncode=rc,
                      wall_seconds=time.monotonic() - started, passed=False)
        if rc == 0 and png.exists() and profile.exists():
            with profile.open() as stream:
                rows = list(csv.DictReader(stream))
            query_frames = [int(r['frame']) for r in rows if r['domain'] == 'count'
                            and r['stage'] == 'native_ray_query' and float(r['value']) == 1]
            result['native_query_frames'] = query_frames
            with Image.open(png) as loaded:
                image = loaded.convert('RGB')
            result['rgb_mean'] = ImageStat.Stat(image).mean
            result['nonconstant'] = any(low != high for low, high in image.getextrema())
            result['passed'] = (query_frames == list(range(args.frames))
                                and image.size == (args.size, args.size) and result['nonconstant'])
            tile = Image.new('RGB', (args.size, args.size + 32), '#202020')
            tile.paste(image, (0, 32))
            ImageDraw.Draw(tile).text((3, 2), name + '\n' + args.pipeline, fill='white')
            tiles.append(tile)
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if tiles:
        sheet = Image.new('RGB', (args.size * len(tiles), args.size + 32))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (index * args.size, 0))
        sheet.save(output / 'contact-sheet.png')
    return int(not all(r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
