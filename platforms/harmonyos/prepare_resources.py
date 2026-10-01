#!/usr/bin/env python3
"""Stage shared mobile assets and SPIR-V into a HarmonyOS rawfile bundle.

The iOS preparation tool records Slang's Vulkan 1.2 SPIR-V before converting it
into MSL. Both backends issue the same Slang requests. Only slang-* cache entries
are portable; MSL is never shipped to HarmonyOS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def stage(source: Path, output: Path, fallback_renderer: Path | None = None):
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError('Output exists; choose a fresh output directory')
    catalog = json.loads((source / 'catalog.json').read_text())
    if not catalog:
        raise ValueError('Scene catalog is empty')
    shaders = sorted((source / 'shaders').glob('slang-*'))
    if not shaders:
        raise ValueError('No SPIR-V shaders; prepare the iOS resource bundle first')
    for shader in shaders:
        data = shader.read_bytes()
        payload = data[64:]
        digest = hashlib.sha256(str(len(payload)).encode() + b':' + payload).hexdigest().encode()
        if data[:64] != digest or payload[:4] != b'\x03\x02\x23\x07':
            raise ValueError(f'Invalid SPIR-V cache: {shader.name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='harmony-bundle-') as temp:
        bundle = Path(temp) / 'Resources'
        bundle.mkdir()
        for name in ['assets', 'Patterns']:
            if (source / name).is_dir():
                shutil.copytree(source / name, bundle / name)
        for name in ['catalog.json', 'texture-report.json', 'asset-hashes.json']:
            if (source / name).is_file():
                shutil.copy2(source / name, bundle / name)
        (bundle / 'shaders').mkdir()
        for shader in shaders:
            shutil.copy2(shader, bundle / 'shaders' / shader.name)
        if fallback_renderer:
            for scene in catalog:
                subprocess.run([str(fallback_renderer.resolve()), str(bundle), scene['id']], check=True)
            # A Metal host can prepare the common SPIR-V without Vulkan's host
            # descriptor limits; its generated MSL is not part of the HAP.
            for metal_shader in (bundle / 'shaders').glob('msl-*'):
                metal_shader.unlink()
        files = []
        for path in sorted(bundle.rglob('*')):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'Missing LFS asset: {path}')
            files.append({'path': path.relative_to(bundle).as_posix(),
                          'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        manifest = {'version': 1, 'files': files}
        (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        bundle.rename(output)
    print(f'Staged {len(catalog)} scenes, {len(shaders)} SPIR-V shaders, {len(files)} files in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fallback-renderer', type=Path,
                        help='harmony_fallback_prepare host executable, for GPUs without ray queries')
    args = parser.parse_args()
    stage(args.source, args.output, args.fallback_renderer)
