#!/usr/bin/env python3
"""Exercise the actual no-compiler rendering path and its bundle failure modes."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--renderer', type=Path, required=True)
parser.add_argument('--resources', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
args.resources = args.resources.resolve()
catalog = json.loads((args.resources / 'catalog.json').read_text())
report = []
def render(resources, scene, output, dimension=128, samples=2):
    return subprocess.run([str(args.renderer.resolve()), str(resources), scene, str(output),
                           'replay', str(dimension), str(samples)], text=True, capture_output=True)
for scene in catalog:
    target = args.output / (scene['id'] + '.png')
    result = render(args.resources, scene['id'], target)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    reference = args.resources.parent / (args.resources.name + '-previews') / target.name
    identical = reference.exists() and reference.read_bytes() == target.read_bytes()
    if reference.exists() and not identical:
        raise RuntimeError(f"Prepared and replay images differ: {scene['id']}")
    print(scene['id'], result.stdout.strip(), 'identical' if identical else 'no reference', flush=True)
    report.append({'scene': scene['id'], 'identical_to_preparation': identical, 'result': result.stdout.strip()})
# Different resolution and accumulation count must not need new shader variants.
result = render(args.resources, 'cornell_box', args.output / 'cornell-256-32spp.png', 256, 32)
if result.returncode or '32 spp' not in result.stdout:
    raise RuntimeError(result.stdout + result.stderr)
with tempfile.TemporaryDirectory(prefix='sparkium-cache-errors-') as directory:
    bundle = Path(directory)
    (bundle / 'assets').symlink_to(args.resources / 'assets', target_is_directory=True)
    cache = bundle / 'shaders'
    cache.mkdir()
    result = render(bundle, 'cornell_box', args.output / 'unexpected.png')
    assert result.returncode != 0 and 'Missing bundled shader' in result.stderr, result
    for entry in (args.resources / 'shaders').iterdir():
        (cache / entry.name).write_bytes(b'0' * 64 + b'intentionally damaged cached shader')
    result = render(bundle, 'cornell_box', args.output / 'unexpected.png')
    assert result.returncode != 0 and 'Corrupt bundled shader' in result.stderr, result
report.append({'resolution_and_accumulation': '256 px, 32 spp', 'missing_and_corrupt_cache': 'pass'})
(args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print('All bundle checks passed.')
