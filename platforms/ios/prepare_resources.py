#!/usr/bin/env python3
"""Package LFS scenes with smaller textures and compile mobile shaders on a Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from texture_assets import copy_asset

parser = argparse.ArgumentParser()
parser.add_argument('--renderer', type=Path, required=True, help='SPARKIUM_PREPARE build of sparkium_mobile_check')
parser.add_argument('--demo-renderer', type=Path, help='SPARKIUM_PREPARE mobile_demo_check (defaults to renderer sibling)')
parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[2] / 'assets')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'out/ios/Resources')
parser.add_argument('--texture-max-dimension', type=int, default=1024,
                    help='Maximum PNG/JPEG texture edge in pixels (default: 1024; 0 keeps originals)')
args = parser.parse_args()
if args.texture_max_dimension < 0:
    parser.error('texture-max-dimension must be nonnegative')
args.output = args.output.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    parser.error('output already exists; choose a new output directory or remove the generated directory first')
scenes = sorted((args.assets / 'scenes').glob('*/scene.json'))
if len(scenes) != 9:
    parser.error(f'expected the nine existing demo scenes, found {len(scenes)}')
with tempfile.TemporaryDirectory(prefix='sparkium-bundle-', dir=args.output.parent) as temp:
    staging = Path(temp) / 'Resources'
    staging.mkdir()
    data = staging / 'assets/data'
    data.mkdir(parents=True)
    shutil.copy2(args.assets / 'data/new-joe-kuo-7.21201', data)
    catalog = []
    texture_report = {'max_dimension': args.texture_max_dimension, 'textures': {}}
    table = data / 'new-joe-kuo-7.21201'
    if table.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise RuntimeError('Sobol LFS object is missing; run git lfs pull in assets')
    manifest = {'data/new-joe-kuo-7.21201': hashlib.sha256(table.read_bytes()).hexdigest()}
    for scene in scenes:
        print('Bundling', scene.parent.name, flush=True)
        document = json.loads(scene.read_text())
        name = ('Blender ' + scene.parent.name.removeprefix('blender_').replace('_', ' ').title()
                if scene.parent.name.startswith('blender_') else document['name'])
        catalog.append({'id': scene.parent.name, 'name': name,
                        'width': document['film']['width'], 'height': document['film']['height']})
        for source in sorted(scene.parent.rglob('*')):
            if not source.is_file() or source.name == 'conversion-report.json':
                continue
            relative = source.relative_to(args.assets)
            target = staging / 'assets' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as stream:
                if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise RuntimeError(f'LFS object missing: {source}. Run git lfs pull in assets.')
            texture = copy_asset(source, target, args.texture_max_dimension)
            if texture is not None:
                texture_report['textures'][relative.as_posix()] = texture
            manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        # Existing JSON paths are relative to the scene, and remain byte-for-byte unchanged.
        def check_paths(value):
            if isinstance(value, dict):
                for child in value.values(): check_paths(child)
            elif isinstance(value, list):
                for child in value: check_paths(child)
            elif isinstance(value, str) and Path(value).suffix.lower() in {
                    '.spmesh', '.sphair', '.obj', '.png', '.jpg', '.jpeg', '.hdr', '.exr'}:
                resource = staging / 'assets/scenes' / scene.parent.name / value
                if not resource.is_file() or not resource.resolve().is_relative_to(staging):
                    raise RuntimeError(f'Missing or external bundled resource: {resource}')
        check_paths(document)
    (staging / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    (staging / 'asset-hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (staging / 'texture-report.json').write_text(json.dumps(texture_report, indent=2) + '\n')
    previews = Path(temp) / 'previews'
    previews.mkdir()
    for scene in catalog:
        print('Preparing shaders:', scene['id'], flush=True)
        subprocess.run([str(args.renderer.resolve()), str(staging), scene['id'],
                        str(previews / (scene['id'] + '.png')), 'prepare', '128', '2'], check=True)
    demo_renderer = args.demo_renderer or args.renderer.with_name('mobile_demo_check')
    subprocess.run([str(demo_renderer.resolve()), str(staging), 'all', str(previews / 'demos'), 'prepare'], check=True)
    if not list((staging / 'shaders').glob('msl-*')):
        raise RuntimeError('No Metal shaders were prepared')
    staging.rename(args.output)
    shutil.copytree(previews, args.output.parent / (args.output.name + '-previews'), dirs_exist_ok=True)
print(f'Prepared {len(catalog)} scenes in {args.output}')
