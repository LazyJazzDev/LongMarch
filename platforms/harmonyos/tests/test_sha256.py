"""Check the portable shader-cache digest against Python's SHA-256."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class SHA256Tests(unittest.TestCase):
    def test_streaming_digest_and_block_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'digest.cpp'
            source.write_text('''
#include "grassland/graphics/sha256.h"
#include <iostream>
int main() {
  grassland::graphics::detail::SHA256 digest;
  char c;
  while (std::cin.get(c)) digest.Update(std::string_view(&c, 1));
  std::cout << digest.Finish();
}
''')
            binary = directory / 'digest'
            subprocess.run(['c++', '-std=c++17', '-I', str(ROOT / 'code'),
                            str(source), '-o', str(binary)], check=True)
            for data in [b'', b'abc', b'a' * 55, b'b' * 56, b'c' * 63,
                         b'd' * 64, b'e' * 65, bytes(range(256)), b'a' * 1_000_000]:
                with self.subTest(length=len(data)):
                    output = subprocess.check_output([str(binary)], input=data).decode()
                    self.assertEqual(output, hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    unittest.main()
