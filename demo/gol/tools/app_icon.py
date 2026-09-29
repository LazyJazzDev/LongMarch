#!/usr/bin/env python3
"""Renders the Game of Life app icon from the boundary-mode button.

The icon is the button without its enclosure: the neutral plate and an enlarged
glider in its first recorded frame (BoundaryGlider mask 0x2470), still offset
toward the lower left. Colors follow IconTheme in shaders/super.slang, which the
app writes to the framebuffer unconverted: color * (2 - 0.3 * |p - (-1, -1)|) for
button-local p in [-1, 1], y downward.

Outputs full-bleed squares without rounded corners; each system applies its mask.
  <output>/ios.png         opaque 1024 x 1024 App Store and home screen icon
  <output>/background.png  opaque HarmonyOS layered-icon background
  <output>/foreground.png  transparent HarmonyOS layered-icon foreground

Usage: app_icon.py <output-dir>
"""

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


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    composite, background, foreground = render()
    Image.fromarray(composite, 'RGB').save(output / 'ios.png')
    Image.fromarray(background, 'RGB').save(output / 'background.png')
    Image.fromarray(foreground, 'RGBA').save(output / 'foreground.png')


if __name__ == '__main__':
    main()
