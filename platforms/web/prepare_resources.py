#!/usr/bin/env python3
"""Stage the browser apps' resources: the WGSL shader cache and 2048's font.

web_shader_prepare (a SPARKIUM_PREPARE build of platforms/ios) compiles the
apps' Slang requests to WGSL with the same cache keys the WebGPU backend reads.
The web build preloads the output directory as /res.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--shader-tool', type=Path, required=True, help='web_shader_prepare executable')
parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
parser.add_argument('--output', type=Path, default=ROOT / 'out/web/resources')
args = parser.parse_args()
output = args.output.resolve()
if output.exists():
    parser.error('output already exists; choose a new output directory or remove the generated directory first')
output.parent.mkdir(parents=True, exist_ok=True)
font = args.assets / 'fonts/ClearSans-Bold-webfont.woff'
if font.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
    raise RuntimeError(f'LFS object missing: {font}. Run git lfs pull in assets.')
with tempfile.TemporaryDirectory(prefix='web-resources-', dir=output.parent) as temp:
    staging = Path(temp) / 'resources'
    subprocess.run([str(args.shader_tool.resolve()), str(staging / 'shaders')], check=True)
    (staging / 'assets/fonts').mkdir(parents=True)
    shutil.copy2(font, staging / 'assets/fonts')
    staging.rename(output)
print(f'Prepared {len(list((output / "shaders").iterdir()))} WGSL shaders in {output}')
