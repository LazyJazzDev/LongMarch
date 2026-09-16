#!/usr/bin/env python3
"""Build and run the actual ObjC++ render controller against a macOS replay build."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--build', type=Path, required=True, help='macOS replay build with CMAKE_EXPORT_COMPILE_COMMANDS=ON')
parser.add_argument('--resources', type=Path, required=True)
args = parser.parse_args()
build = args.build.resolve()
entry = next(e for e in json.loads((build / 'compile_commands.json').read_text())
             if Path(e['file']).name == 'RenderSession.cpp')
command = shlex.split(entry['command'])
flags = []
i = 0
while i < len(command):
    if command[i] == '-o': i += 2; continue
    if command[i] == '-c' or command[i] == entry['file']: i += 1; continue
    flags.append(command[i]); i += 1
source = Path(__file__).resolve().parent
output = build / 'RenderControllerCheck'
flags += ['-x', 'objective-c++', '-fobjc-arc', str(source.parent / 'SparkiumBridge.mm'),
          str(source / 'RenderControllerCheck.mm'), '-x', 'none', str(build / 'libsparkium_mobile.a'),
          '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'QuartzCore',
          '-framework', 'CoreGraphics', '-o', str(output)]
subprocess.run(flags, cwd=entry['directory'], check=True)
subprocess.run([str(output), str(args.resources.resolve())], check=True, timeout=130)
