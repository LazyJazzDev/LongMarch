import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('harmony_resources', Path(__file__).parents[1] / 'prepare_resources.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResourceTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        (source / 'shaders').mkdir(parents=True)
        (source / 'assets').mkdir()
        (source / 'catalog.json').write_text('[{"id":"cornell_box"}]')
        (source / 'assets' / 'sample.txt').write_text('asset data')
        payload = bytes.fromhex('03022307') + b'\x00' * 16
        digest = hashlib.sha256(str(len(payload)).encode() + b':' + payload).hexdigest().encode()
        (source / 'shaders' / 'hlsl-test').write_bytes(digest + payload)
        (source / 'shaders' / 'msl-test').write_text('Metal only')
        return source

    def test_manifest_matches_copied_bytes_and_omits_metal(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = self.fixture(root)
            output = root / 'output'
            MODULE.stage(source, output)
            self.assertFalse((output / 'shaders' / 'msl-test').exists())
            for entry in json.loads((output / 'manifest.json').read_text())['files']:
                data = (output / entry['path']).read_bytes()
                self.assertEqual(entry['size'], len(data))
                self.assertEqual(entry['sha256'], hashlib.sha256(data).hexdigest())
            with self.assertRaises(ValueError):
                MODULE.stage(source, output)

    def test_corrupt_shader_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            with (source / 'shaders' / 'hlsl-test').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'Invalid SPIR-V'):
                MODULE.stage(source, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_missing_lfs_object_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            (source / 'assets' / 'sample.txt').write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'Missing LFS'):
                MODULE.stage(source, root / 'output')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
