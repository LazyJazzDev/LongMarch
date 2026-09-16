#!/usr/bin/env python3
"""Embed HLSL as a VFS using a host tool during cross compilation."""
import json
from pathlib import Path
import sys
root, output = map(Path, sys.argv[1:])
lines = ['::grassland::VirtualFileSystem GetShaderVirtualFileSystem() {',
         '  ::grassland::VirtualFileSystem vfs;']
for path in sorted(root.rglob('*')):
    if path.suffix not in ('.hlsl', '.hlsli'):
        continue
    name = json.dumps(path.relative_to(root).as_posix())
    content = json.dumps(path.read_text(), ensure_ascii=True)
    lines.append(f'  vfs.WriteFile({name}, std::string({content}));')
lines.extend(['  return vfs;', '}'])
output.write_text('\n'.join(lines) + '\n')
