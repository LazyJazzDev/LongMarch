#!/usr/bin/env python3
"""Convert Sparkium linear-sRGB Radiance HDR to browser-viewable HDR AVIF.

Requires numpy and FFmpeg/ffprobe with libsvtav1 and the AVIF muxer.
Linear 1.0 maps to 203 cd/m2 by default (HDR reference white). No tone mapping
is applied; values outside PQ's 0..10000 cd/m2 range are clipped. Output is
10-bit BT.2020 non-constant-luminance YUV 4:2:0 with ST 2084 (PQ) signaling.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def linear_srgb_to_pq(rgb, reference_white):
    # D65 linear sRGB / BT.709 -> D65 linear BT.2020 (no chromatic adaptation).
    matrix = np.array([[0.627404, 0.329283, 0.043313],
                       [0.069097, 0.919540, 0.011362],
                       [0.016391, 0.088013, 0.895595]], dtype=np.float64)
    luminance = np.clip((rgb @ matrix.T) * (reference_white / 10000.0), 0, 1)
    m1, m2 = 2610 / 16384, 2523 / 32
    c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
    power = luminance ** m1
    return ((c1 + c2 * power) / (1 + c3 * power)) ** m2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='linear-sRGB .hdr exported by sparkium_cli')
    parser.add_argument('output', type=Path, help='HDR .avif output')
    parser.add_argument('--reference-white', type=float, default=203.0, help='cd/m2 for linear 1.0 (default 203)')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    args = parser.parse_args()
    if not np.isfinite(args.reference_white) or not 0 < args.reference_white <= 10000:
        parser.error('reference white must be finite and within (0, 10000]')
    if args.input.suffix.lower() != '.hdr' or args.output.suffix.lower() != '.avif':
        parser.error('input must be .hdr and output must be .avif')
    probe = json.loads(subprocess.check_output([
        args.ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height', '-of', 'json', str(args.input)]))['streams'][0]
    width, height = probe['width'], probe['height']
    if width % 2 or height % 2:
        parser.error('YUV 4:2:0 output requires even width and height')
    decoded = subprocess.check_output([
        args.ffmpeg, '-v', 'error', '-i', str(args.input), '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', 'gbrpf32le', 'pipe:1'])
    planes = np.frombuffer(decoded, dtype='<f4').reshape(3, height, width)
    rgb = np.stack([planes[2], planes[0], planes[1]], axis=-1)
    if not np.all(np.isfinite(rgb)):
        parser.error('input contains nonfinite pixels')
    pq = linear_srgb_to_pq(rgb, args.reference_white)
    encoded_rgb = np.rint(pq * 65535).astype('<u2')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        args.ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb48le',
        '-video_size', f'{width}x{height}', '-i', 'pipe:0', '-frames:v', '1',
        '-vf', ('scale=out_color_matrix=bt2020:out_range=tv,'
                'setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc'),
        '-pix_fmt', 'yuv420p10le',
        '-c:v', 'libsvtav1', '-crf', '10', '-preset', '4',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc',
        '-color_range', 'tv', str(args.output)], input=encoded_rgb.tobytes(), check=True)
    metadata = json.loads(subprocess.check_output([
        args.ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=pix_fmt,color_primaries,color_transfer,color_space', '-of', 'json',
        str(args.output)]))['streams'][0]
    expected = dict(pix_fmt='yuv420p10le', color_primaries='bt2020',
                    color_transfer='smpte2084', color_space='bt2020nc')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Encoder did not preserve HDR signaling: {metadata}')
    print(json.dumps(dict(width=width, height=height, reference_white_nits=args.reference_white,
                          linear_rgb_peak=float(rgb.max()),
                          pixels_above_reference_white=int(np.count_nonzero(rgb.max(axis=-1) > 1)),
                          pq_clipped_pixels=int(np.count_nonzero(
                              rgb.max(axis=-1) * args.reference_white > 10000))), indent=2))


if __name__ == '__main__':
    main()
