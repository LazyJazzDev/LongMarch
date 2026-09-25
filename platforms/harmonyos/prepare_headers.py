#!/usr/bin/env python3
"""Copy only portable headers; never link macOS vcpkg libraries into a HAP."""
import argparse
from pathlib import Path
import shutil

NAMES = ['fmt', 'glm', 'eigen3', 'rapidjson', 'GLFW', 'vk_mem_alloc.h', 'tiny_obj_loader.h']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='vcpkg installed/<host-triplet>/include')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'out/harmonyos/deps/include')
    args = parser.parse_args()
    files = [args.source / name for name in NAMES]
    files += list(args.source.glob('stb*.h'))
    if (args.source / 'stb').is_dir():
        files.append(args.source / 'stb')
    if not any(p.name.startswith('stb') for p in files):
        parser.error('Missing stb headers')
    for source in files:
        if not source.exists():
            parser.error(f'Missing {source}; install the HarmonyOS vcpkg manifest for the host')
    if args.output.exists():
        parser.error('Output already exists; choose a fresh directory')
    args.output.mkdir(parents=True)
    for source in files:
        target = args.output / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    print(f'Portable headers: {args.output}; Vulkan headers come from the target SDK')


if __name__ == '__main__':
    main()
