#!/usr/bin/env python3
"""Compare native Metal and Vulkan nbody_cs frame latency with interleaved repeats."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('out/nbody-profile/results'))
    parser.add_argument('--particles', type=int, nargs='+', default=[16384, 32768, 65536])
    parser.add_argument('--modes', nargs='+', choices=['compute', 'offscreen', 'window'], default=['compute', 'offscreen'])
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-gpu-timing', action='store_true')
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error('invalid frame or repeat count')
    args.output.mkdir(parents=True, exist_ok=True)
    exe = str(args.exe.resolve())
    environment = dict(os.environ)
    for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION', 'MVK_CONFIG_DEBUG'):
        environment.pop(key, None)
    metadata = dict(platform=platform.platform(), executable=exe,
                    gpu_timing=not args.no_gpu_timing,
                    environment={k: v for k, v in environment.items()
                                 if k.startswith(('MVK_', 'MTL_', 'VK_', 'LONGMARCH_'))})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    runs = []
    for mode in args.modes:
        for count in args.particles:
            for repeat in range(args.repeats):
                backends = ['metal', 'vulkan'] if repeat % 2 == 0 else ['vulkan', 'metal']
                for backend in backends:
                    name = f'{mode}-{count}-{backend}-{repeat}'
                    csv_path = args.output / f'{name}.csv'
                    command = [exe, '--backend', backend, '--mode', mode, '--particles', str(count),
                               '--warmup', str(args.warmup), '--frames', str(args.frames), '--seed', '1',
                               '--width', '1920', '--height', '1080', '--csv', str(csv_path.resolve())]
                    if args.no_gpu_timing:
                        command.append('--no-gpu-timing')
                    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=600)
                    (args.output / f'{name}.log').write_text(result.stdout + result.stderr)
                    if result.returncode:
                        raise RuntimeError(f'{name} failed: {result.stderr[-2000:]}')
                    entry = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT ')))
                    if entry['frames'] != args.frames:
                        raise RuntimeError(f'{name}: window closed before all frames completed')
                    with csv_path.open() as stream:
                        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
                    entry.update(repeat=repeat, wall_p50_ms=statistics.median(row['wall_ms'] for row in rows),
                                 wall_p95_ms=sorted(row['wall_ms'] for row in rows)[int((len(rows)-1)*0.95)],
                                 record_ms=statistics.mean(row['record_ms'] for row in rows),
                                 submit_ms=statistics.mean(row['submit_ms'] for row in rows),
                                 wait_ms=statistics.mean(row['wait_ms'] for row in rows), command=command)
                    runs.append(entry)
                    (args.output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
                    print(name, f"wall={entry['wall_ms']:.3f} ms GPU={entry['gpu_ms']:.3f} ms", flush=True)
    summary = []
    for mode in args.modes:
        for count in args.particles:
            item = dict(mode=mode, particles=count)
            for backend in ('Metal', 'Vulkan'):
                entries = [r for r in runs if r['backend'] == backend and r['mode'] == mode and r['particles'] == count]
                item[backend] = {key: statistics.median(r[key] for r in entries)
                                 for key in ('wall_ms','gpu_ms','wall_p50_ms','wall_p95_ms','record_ms','submit_ms','wait_ms')}
                item[backend]['run_wall_ms'] = [r['wall_ms'] for r in entries]
            item['metal_over_vulkan_wall'] = item['Metal']['wall_ms'] / item['Vulkan']['wall_ms']
            summary.append(item)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
