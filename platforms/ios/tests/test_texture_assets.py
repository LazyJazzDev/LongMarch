"""Check packaged texture dimensions, alpha, and preservation of source assets."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from texture_assets import copy_asset


class TextureAssetsTest(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[3] / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_resize_png_preserves_alpha_aspect_and_source(self):
        source, target = self.root / 'source.png', self.root / 'packaged.png'
        Image.new('RGBA', (2048, 1024), (50, 100, 200, 128)).save(source)
        original_hash = hashlib.sha256(source.read_bytes()).digest()
        report = copy_asset(source, target, 1024)
        with Image.open(target) as image:
            self.assertEqual(image.size, (1024, 512))
            self.assertEqual(image.getextrema()[3], (128, 128))
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), original_hash)
        self.assertEqual(report['packaged_rgba8_bytes'], 1024 * 512 * 4)

    def test_jpeg_preserves_format_and_portrait_aspect(self):
        source, target = self.root / 'source.jpg', self.root / 'packaged.jpg'
        Image.new('RGB', (1024, 2048), (90, 120, 150)).save(source)
        copy_asset(source, target, 1024)
        with Image.open(target) as image:
            self.assertEqual(image.format, 'JPEG')
            self.assertEqual(image.size, (512, 1024))

    def test_palette_transparency_is_retained(self):
        source, target = self.root / 'source.png', self.root / 'packaged.png'
        image = Image.new('P', (32, 32), 0)
        image.save(source, transparency=0)
        copy_asset(source, target, 16)
        with Image.open(target) as image:
            self.assertEqual(image.mode, 'RGBA')
            self.assertEqual(image.getextrema()[3], (0, 0))

    def test_rgb_color_key_transparency_is_retained(self):
        source, target = self.root / 'source.png', self.root / 'packaged.png'
        Image.new('RGB', (32, 32), (20, 40, 60)).save(source, transparency=(20, 40, 60))
        copy_asset(source, target, 16)
        with Image.open(target) as image:
            self.assertEqual(image.mode, 'RGBA')
            self.assertEqual(image.getextrema()[3], (0, 0))

    def test_small_disabled_and_nontexture_assets_are_unchanged(self):
        source, target = self.root / 'source.png', self.root / 'packaged.png'
        Image.new('RGB', (64, 32)).save(source)
        for limit in (1024, 0):
            copy_asset(source, target, limit)
            self.assertEqual(source.read_bytes(), target.read_bytes())
        scene, packaged = self.root / 'scene.json', self.root / 'packaged.json'
        scene.write_text('{"film": {"width": 2000, "height": 1000}}\n')
        self.assertIsNone(copy_asset(scene, packaged, 1024))
        self.assertEqual(scene.read_bytes(), packaged.read_bytes())


if __name__ == '__main__':
    unittest.main()
