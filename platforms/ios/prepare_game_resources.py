#!/usr/bin/env python3
"""Extract a standalone game's resources from a prepared LongMarch bundle.

A standalone app ships only its game's shaders and the GoL pattern library, not
the Sparkium scenes. The game runs in a replay build (SPARKIUM_PREPARE=OFF) of
mobile_demo_check against an empty shader folder; each shader it reports
missing is copied from the full bundle until the game renders. Both the Slang
SPIR-V and the Metal entries it reads are kept, so the output also serves as the
source of a HarmonyOS bundle.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='Prepared bundle (prepare_resources.py output)')
parser.add_argument('--checker', type=Path, required=True, help='Replay build of mobile_demo_check')
parser.add_argument('--demo', default='gol')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('output already exists; choose a new output directory or remove the generated directory first')
args.output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='game-bundle-', dir=args.output.parent) as temp:
    staging = Path(temp) / 'Resources'
    (staging / 'shaders').mkdir(parents=True)
    shutil.copytree(args.source / 'Patterns', staging / 'Patterns')
    while True:
        run = subprocess.run([str(args.checker.resolve()), str(staging), args.demo, str(Path(temp) / 'check')],
                             capture_output=True, text=True)
        missing = re.search(r'Missing bundled shader ((?:slang|msl)-[0-9a-f]{64})', run.stdout + run.stderr)
        if missing is None:
            if run.returncode != 0:
                raise RuntimeError(f'{args.demo} failed:\n{run.stdout}{run.stderr}')
            break
        shutil.copy2(args.source / 'shaders' / missing.group(1), staging / 'shaders')
    staging.rename(args.output)
print(f'Prepared {args.demo} with {len(list((args.output / "shaders").iterdir()))} shaders in {args.output}')
