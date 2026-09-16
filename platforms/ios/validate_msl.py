#!/usr/bin/env python3
"""Compile every cached MSL stage for the actual iOS target (requires Metal Toolchain)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--resources', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
files = sorted((args.resources / 'shaders').glob('msl-*'))
if not files:
    parser.error('no cached MSL stages')
def compile_stage(path):
    # SHA-256 integrity prefix, then one line of reflection, then the MSL source.
    source = path.read_bytes()[64:].split(b'\n', 1)[1]
    target = args.output / (path.name + '.metal')
    target.write_bytes(source)
    result = subprocess.run(['xcrun', '--sdk', 'iphoneos', 'metal', '-std=metal3.0',
                             '-target', 'air64-apple-ios18.0', '-ffast-math', '-c', str(target),
                             '-o', str(target.with_suffix('.air'))], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(path.name + '\n' + result.stderr)
    return path.name
with ThreadPoolExecutor(max_workers=2) as pool:
    for name in pool.map(compile_stage, files):
        print('PASS', name, flush=True)
print(f'Compiled {len(files)} iOS MSL stages.')
