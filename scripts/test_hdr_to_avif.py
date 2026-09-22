#!/usr/bin/env python3
"""HDR export conversion checks; requires the same tools as hdr_to_avif.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hdr_to_avif import linear_srgb_to_pq


class HDRConversionTest(unittest.TestCase):
    def test_reference_white_and_pq_limits(self):
        # Published ST 2084 code values: 100 nits ~= 0.508078, 10000 nits = 1.
        values = linear_srgb_to_pq(np.array([[0., 0., 0.], [1., 1., 1.],
                                            [100., 100., 100.], [-1., -1., -1.]]), 100.)
        np.testing.assert_allclose(values[1], 0.508078, atol=1e-6)
        np.testing.assert_allclose(values[2], 1., atol=1e-6)
        self.assertLess(values[0].max(), 1e-6)
        np.testing.assert_equal(values[0], values[3])

    def test_browser_file_signaling_and_luminance(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = [Path(directory) / name for name in ('levels.hdr', 'levels.avif')]
            # Flat RGBE patches at linear 0.25, 1, 4 and 16, without RLE.
            row = b''.join(bytes([128, 128, 128, exponent]) * 64 for exponent in (127, 129, 131, 133))
            source.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 256\n' + row * 64)
            subprocess.run([sys.executable, str(Path(__file__).with_name('hdr_to_avif.py')),
                            str(source), str(output)], check=True, capture_output=True)
            metadata = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output)]))['streams'][0]
            self.assertEqual(metadata['color_transfer'], 'smpte2084')
            self.assertEqual(metadata['color_primaries'], 'bt2020')
            self.assertEqual(metadata['pix_fmt'], 'yuv420p10le')
            raw = subprocess.check_output([
                'ffmpeg', '-v', 'error', '-i', str(output), '-frames:v', '1',
                '-vf', 'scale=in_color_matrix=bt2020:in_range=tv:out_range=pc',
                '-pix_fmt', 'rgb48le', '-f', 'rawvideo', 'pipe:1'])
            pq = np.frombuffer(raw, dtype='<u2').reshape(64, 256, 3)[32, [32, 96, 160, 224]] / 65535.
            power = pq ** (32 / 2523)
            nits = 10000 * (np.maximum(power - 3424 / 4096, 0) /
                            (2413 / 128 - 2392 / 128 * power)) ** (16384 / 2610)
            np.testing.assert_allclose(nits.mean(axis=1), np.array([0.25, 1, 4, 16]) * 203, rtol=0.025)


if __name__ == '__main__':
    unittest.main()
