"""Regress DXC's lost ByteAddressBuffer NonUniform access-chain decorations."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('dxc') and shutil.which('spirv-val'),
                     'DXC and SPIR-V Tools are required')
class NonUniformTests(unittest.TestCase):
    def test_compiled_buffer_accesses(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'repair.cpp'
            source.write_text('''
#include "grassland/vulkan/spirv_nonuniform.h"
#include <fstream>
int main(int argc, char **argv) {
  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(size_t(input.tellg()) / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), words.size() * 4);
  auto fixed = grassland::vulkan::RestoreStorageBufferNonUniform(words);
  if (grassland::vulkan::RestoreStorageBufferNonUniform(fixed) != fixed) return 1;
  std::ofstream output(argv[2], std::ios::binary);
  output.write(reinterpret_cast<const char *>(fixed.data()), fixed.size() * 4);
}
''')
            binary = directory / 'repair'
            subprocess.run(['c++', '-std=c++17', '-I', str(ROOT / 'code'),
                            str(source), '-o', str(binary)], check=True)
            for local in [False, True]:
                for divergent in [False, True]:
                    with self.subTest(local=local, divergent=divergent):
                        index = 'NonUniformResourceIndex(id.x)' if divergent else '0'
                        access = f'buffers[{index}]'
                        body = f'ByteAddressBuffer b = {access}; output[id.x] = b.Load(0);' if local else f'output[id.x] = {access}.Load(0);'
                        shader = directory / 'test.hlsl'
                        shader.write_text('ByteAddressBuffer buffers[] : register(t0, space0);\n'
                                          'RWStructuredBuffer<uint> output : register(u0, space1);\n'
                                          '[numthreads(64,1,1)] void Main(uint3 id:SV_DispatchThreadID) {' + body + '}')
                        original, fixed = directory / 'input.spv', directory / 'fixed.spv'
                        subprocess.run(['dxc', '-spirv', '-T', 'cs_6_0', '-E', 'Main',
                                        '-fspv-target-env=vulkan1.2', str(shader), '-Fo', str(original)], check=True)
                        subprocess.run([str(binary), str(original), str(fixed)], check=True)
                        subprocess.run(['spirv-val', '--target-env', 'vulkan1.2', str(fixed)], check=True)
                        data = fixed.read_bytes()
                        words = struct.unpack(f'<{len(data) // 4}I', data)
                        instructions, offset = [], 5
                        while offset < len(words):
                            size = words[offset] >> 16
                            instructions.append((words[offset] & 0xffff, words[offset + 1:offset + size]))
                            offset += size
                        decorations = {args[0] for op, args in instructions if op == 71 and args[1] == 5300}
                        storage_types = {args[0] for op, args in instructions if op == 32 and args[1] == 12}
                        if divergent:
                            self.assertTrue(any(op == 17 and args == (5308,) for op, args in instructions))
                            pointers = [args[1] for op, args in instructions if op == 65 and args[0] in storage_types
                                        and any(index in decorations for index in args[2:])]
                            self.assertTrue(pointers)
                            self.assertTrue(all(pointer in decorations for pointer in pointers))
                        else:
                            self.assertEqual(original.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
