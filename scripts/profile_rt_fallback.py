#!/usr/bin/env python3
"""Profile frames with Vulkan GPU timestamps."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

from check_rt_fallback import ROOT, DEMOS, read_scene, graph_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/rt-profile')
    parser.add_argument('--scenes', nargs='+', default=list(DEMOS))
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--spp', type=int, default=8)
    parser.add_argument('--bounces', type=int, default=12)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--graph-smoke', action='store_true')
    parser.add_argument('--cpu-only', action='store_true', help='disable GPU timestamps to measure profiling overhead')
    args = parser.parse_args()
    if min(args.size, args.spp, args.bounces, args.repeat) <= 0 or not 0 <= args.warmup < args.frames:
        parser.error('invalid size/spp/bounces/repeat or warmup/frame count')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = [(name, read_scene(ROOT / 'assets/scenes' / name / 'scene.json')) for name in args.scenes]
    if args.graph_smoke:
        scenes.append(('graph_smoke', graph_smoke()))
    summaries = []
    for name, scene in scenes:
        scene['film'].update(width=args.size, height=args.size)
        scene['renderer'].update(samples_per_dispatch=args.spp, max_bounces=args.bounces)
        scene_path = args.output / f'{name}.json'
        scene_path.write_text(json.dumps(scene, indent=2) + '\n')
        samples, cold = {}, []
        for repeat in range(args.repeat):
            prefix = args.output / f'{name}-{repeat}'
            command = [str(args.cli.resolve()), str(scene_path), '--pipeline', 'rt_fallback',
                       '--frames', str(args.frames), '--profile', str(prefix.with_suffix('.csv')),
                       '-o', str(prefix.with_suffix('.png'))]
            if args.cpu_only:
                command.append('--profile-cpu-only')
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            first_frame = {}
            with prefix.with_suffix('.csv').open() as stream:
                for row in csv.DictReader(stream):
                    key = row['domain'] + '/' + row['stage']
                    value = float(row['value'])
                    if int(row['frame']) == 0:
                        first_frame[key] = value
                    if int(row['frame']) >= args.warmup:
                        samples.setdefault(key, []).append(value)
            cold.append(first_frame)
            print(f'{name}: run {repeat + 1}/{args.repeat} done', flush=True)
        def summarize(values):
            ordered = sorted(values)
            return {'mean': statistics.mean(values), 'median': statistics.median(values),
                    'p95': ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
                    'min': min(values), 'max': max(values)}
        result = {'scene': name, 'size': args.size, 'spp': args.spp, 'bounces': args.bounces,
                  'gpu_timestamps': not args.cpu_only, 'frames_per_run': args.frames, 'warmup_per_run': args.warmup, 'runs': args.repeat,
                  'steady': {key: summarize(values) for key, values in samples.items()}, 'first_frames': cold}
        summaries.append(result)
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(f"{name}: frame median={result['steady']['cpu_ms/frame_wall']['median']:.3f} ms", flush=True)


if __name__ == '__main__':
    main()
