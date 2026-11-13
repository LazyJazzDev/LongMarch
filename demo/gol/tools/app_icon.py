#!/usr/bin/env python3
"""Renders the Game of Life app icon from the boundary-mode button.

The icon is the button without its enclosure: the neutral plate and an enlarged
glider in its first recorded frame (BoundaryGlider mask 0x2470), still offset
toward the lower left. Colors follow IconTheme in shaders/super.slang, which the
app writes to the framebuffer unconverted: color * (2 - 0.3 * |p - (-1, -1)|) for
button-local p in [-1, 1], y downward.

Outputs full-bleed squares without rounded corners; each system applies its mask.
  <output>/AppIcon.icon    Icon Composer icon for iOS: the plate gradient as the
                           fill and the glider as a Liquid Glass layer
  <output>/ios.png         opaque 1024 x 1024 flat rendering
  <output>/background.png  opaque HarmonyOS layered-icon background
  <output>/foreground.png  transparent HarmonyOS layered-icon foreground

Usage: app_icon.py <output-dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 1024
SUPERSAMPLE = 4
PLATE = np.array([0.16, 0.18, 0.21])  # button_palette::Background() at rest
NEUTRAL = np.array([0.47, 0.49, 0.52])  # button_palette::kNeutral
GLIDER = 0x2470  # bit y * 4 + x of the 4 x 4 display
PITCH = 0.32  # the button uses 0.18


def render():
    n = SIZE * SUPERSAMPLE
    coordinates = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(coordinates, coordinates)
    scale = 2.0 - np.hypot(x + 1.0, y + 1.0) * 0.3
    plate = PLATE * scale[..., None]
    cells = np.zeros((n, n), dtype=bool)
    for bit in range(16):
        if GLIDER >> bit & 1:
            cx, cy = (bit % 4 - 1.5) * PITCH, (bit // 4 - 1.5) * PITCH
            cells |= (np.abs(x - cx) <= PITCH * 0.5) & (np.abs(y - cy) <= PITCH * 0.5)
    glider = NEUTRAL * scale[..., None]

    def resolve(image):
        return image.reshape(SIZE, SUPERSAMPLE, SIZE, SUPERSAMPLE, -1).mean(axis=(1, 3))

    coverage = resolve(cells[..., None].astype(float))
    plate, glider = resolve(plate), resolve(glider)
    composite = plate * (1.0 - coverage) + glider * coverage
    to_bytes = lambda image: np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    foreground = np.concatenate([glider, coverage], axis=-1)
    return to_bytes(composite), to_bytes(plate), to_bytes(foreground)


def srgb(color):
    return 'srgb:' + ','.join(f'{v:.5f}' for v in (*color, 1.0))


def glider_outline():
    """SVG path of the live cells' union: cell edges not shared by two live cells, chained into loops."""
    live = {(bit % 4, bit // 4) for bit in range(16) if GLIDER >> bit & 1}
    edges = {}
    for x, y in live:
        # Clockwise on screen (y downward), so shared edges cancel.
        for a, b, neighbor in (((x, y), (x + 1, y), (x, y - 1)), ((x + 1, y), (x + 1, y + 1), (x + 1, y)),
                               ((x + 1, y + 1), (x, y + 1), (x, y + 1)), ((x, y + 1), (x, y), (x - 1, y))):
            if neighbor not in live:
                edges.setdefault(a, []).append(b)
    point = lambda v: tuple(((c - 2) * PITCH + 1.0) * SIZE / 2 for c in v)
    path = []
    while edges:
        start = next(iter(edges))
        loop, current = [start], start
        while True:
            following = edges[current].pop()
            if not edges[current]:
                del edges[current]
            if following == start:
                break
            loop.append(following)
            current = following
        # Keep only corners.
        corners = [v for i, v in enumerate(loop)
                   if (loop[i - 1][0] - v[0]) * (loop[(i + 1) % len(loop)][1] - v[1]) !=
                   (loop[i - 1][1] - v[1]) * (loop[(i + 1) % len(loop)][0] - v[0])]
        path.append('M' + ' L'.join('{:.2f} {:.2f}'.format(*point(v)) for v in corners) + ' Z')
    return ' '.join(path)


def glider_svg():
    # IconTheme's scale runs from 2.0 at the top-left corner to 1.15 at the
    # bottom-right; the glass layer keeps that diagonal ramp.
    stops = ''.join(f'<stop offset="{offset}" stop-color="#{"".join(f"{round(min(v * scale, 1.0) * 255):02x}" for v in NEUTRAL)}"/>'
                    for offset, scale in ((0, 2.0), (1, 2.0 - 2.0 * np.sqrt(2.0) * 0.3)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">\n'
            f'<defs><linearGradient id="shade" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{SIZE}" y2="{SIZE}">'
            f'{stops}</linearGradient></defs>\n<path fill="url(#shade)" d="{glider_outline()}"/>\n</svg>\n')


def write_icon_composer(path):
    (path / 'Assets').mkdir(parents=True, exist_ok=True)
    (path / 'Assets/glider.svg').write_text(glider_svg())
    low = 2.0 - 2.0 * np.sqrt(2.0) * 0.3
    document = {
        'fill': {'linear-gradient': [srgb(PLATE * 2.0), srgb(PLATE * low)],
                 'orientation': {'start': {'x': 0, 'y': 0}, 'stop': {'x': 1, 'y': 1}}},
        'groups': [{
            'layers': [{'image-name': 'glider.svg', 'name': 'glider', 'glass': True}],
            'shadow': {'kind': 'neutral', 'opacity': 0.5},
            'specular': True,
            'translucency': {'enabled': True, 'value': 0.4},
        }],
        'supported-platforms': {'squares': ['iOS']},
    }
    (path / 'icon.json').write_text(json.dumps(document, indent=2) + '\n')


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    composite, background, foreground = render()
    Image.fromarray(composite, 'RGB').save(output / 'ios.png')
    Image.fromarray(background, 'RGB').save(output / 'background.png')
    Image.fromarray(foreground, 'RGBA').save(output / 'foreground.png')
    write_icon_composer(output / 'AppIcon.icon')


if __name__ == '__main__':
    main()
