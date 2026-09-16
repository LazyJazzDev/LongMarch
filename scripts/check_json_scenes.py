#!/usr/bin/env python3
"""Check basic JSON scene loading, relative assets, raster output, and invalid inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('area_light', 'cornell_box', 'point_light', 'principled', 'specular', 'texture')


def relative_assets(value, source, destination, textures=False):
    if isinstance(value, dict):
        return {key: os.path.relpath((source / item).resolve(), destination)
                if isinstance(item, str) and (key == 'path' or textures)
                else relative_assets(item, source, destination, key == 'textures')
                for key, item in value.items()}
    if isinstance(value, list):
        return [relative_assets(item, source, destination) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/json-validation')
    args = parser.parse_args()
    cli, output = args.cli.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    working_directory = output / 'different-working-directory'
    working_directory.mkdir(exist_ok=True)
    listed = subprocess.run([str(cli), '--list', str(ROOT / 'assets/scenes')],
                            capture_output=True, text=True, check=True)
    names = {Path(line).parent.name for line in listed.stdout.splitlines()}
    assert set(SCENES).issubset(names), 'scene discovery omitted a basic demo'
    results = []

    def run(name, scene, valid=True, raw=None):
        document, png = output / f'{name}.json', output / f'{name}.png'
        document.write_text(raw if raw is not None else json.dumps(scene))
        png.unlink(missing_ok=True)
        result = subprocess.run([str(cli), str(document), '--pipeline', 'rasterization', '-o', str(png)],
                                cwd=working_directory, capture_output=True, text=True, timeout=120)
        (output / f'{name}.log').write_text(result.stdout + result.stderr)
        if valid:
            passed = result.returncode == 0 and png.exists()
            if passed:
                data = png.read_bytes()
                passed = data[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', data[16:24]) == (96, 96)
        else:
            passed = result.returncode == 1 and 'sparkium_cli:' in result.stderr and not png.exists()
        results.append(dict(case=name, expected='load' if valid else 'reject',
                            returncode=result.returncode, passed=passed))
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f"{'PASS' if passed else 'FAIL'} {name} (exit {result.returncode})", flush=True)

    for name in SCENES:
        path = ROOT / 'assets/scenes' / name / 'scene.json'
        scene = relative_assets(json.loads(path.read_text()), path.parent, output)
        scene['film'].update(width=96, height=96)
        scene['renderer'].update(samples_per_dispatch=1, max_bounces=1)
        run(name, scene)

    base = json.loads((output / 'cornell_box.json').read_text())
    geometry_id = next(iter(base['geometries']))
    for name, change in (
        ('missing-film', lambda d: d.pop('film')),
        ('unknown-version', lambda d: d.update(version=2)),
        ('negative-width', lambda d: d['film'].update(width=-1)),
        ('unknown-reference', lambda d: d['entities'][0].update(geometry='missing')),
        ('invalid-pipeline', lambda d: d['renderer'].update(pipeline='missing')),
        ('wrong-format-type', lambda d: d.update(format=123)),
        ('wrong-width-type', lambda d: d['film'].update(width='wide')),
        ('wrong-vector-type', lambda d: d['camera'].update(eye=['x', 0, 0])),
        ('wrong-boolean-type', lambda d: d['renderer'].update(alpha_shadow='yes')),
        ('wrong-object-type', lambda d: d.update(materials=[])),
        ('wrong-array-type', lambda d: d.update(entities={})),
        ('invalid-camera', lambda d: d['camera'].update(target=d['camera']['eye'])),
        ('invalid-fov', lambda d: d['camera'].update(fov_degrees=180)),
        ('nonfinite-float', lambda d: d['camera'].update(fov_degrees=1e100)),
        ('zero-samples', lambda d: d['renderer'].update(samples_per_dispatch=0)),
        ('zero-bounces', lambda d: d['renderer'].update(max_bounces=0)),
        ('invalid-mesh-index', lambda d: d['geometries'][geometry_id].update(indices=[0, 1, 999999])),
        ('incomplete-triangle', lambda d: d['geometries'][geometry_id].update(indices=[0, 1])),
        ('missing-mesh-uvs', lambda d: d['geometries'][geometry_id].update(tex_coords=[])),
        ('wrong-mesh-positions', lambda d: d['geometries'][geometry_id].update(positions='vertices')),
        ('invalid-sphere', lambda d: d['geometries'].update(invalid={'type': 'sphere', 'longitude_segments': 0})),
        ('invalid-transform', lambda d: d['entities'][0].update(transform='identity')),
        ('unknown-material', lambda d: d['materials'].update(invalid={'type': 'missing'})),
        ('unknown-geometry', lambda d: d['geometries'].update(invalid={'type': 'missing'})),
    ):
        scene = copy.deepcopy(base)
        change(scene)
        run(name, scene, valid=False)
    run('invalid-json', None, valid=False, raw='{')
    print(f"{sum(r['passed'] for r in results)}/{len(results)} checks passed")
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
