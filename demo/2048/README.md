# 2048

The 2048 puzzle game, ported from the GUI of
[Yao-class-cpp-studio/Assignment-3](https://github.com/Yao-class-cpp-studio/Assignment-3)
to `grassland::graphics`. The original targeted the legacy grassland Vulkan
wrapper; this version runs on every LongMarch backend (Vulkan, D3D12, Metal).

```sh
cmake --build build --target demo_2048
build/demo/2048/demo_2048
```

Use the arrow keys to move the blocks. **MENU** opens the menu with
**KEEP GOING** and **NEW GAME**; a game-over screen offers **TRY AGAIN**.

Options:

- `--backend auto|vulkan|d3d12|metal`: select the graphics backend.
- `--frames N`: exit after `N` rendered frames.
- `--screenshot FILE`: save the last frame as PNG on exit.

`2048_lib/` holds `update_step`, the part students implement in the assignment;
the demo ships a reference implementation that passes the assignment's tests.

Text is vector geometry: `font/` triangulates FreeType glyph outlines, as the
old `grassland::font::Factory` did, without its on-disk cache. The font is
`fonts/ClearSans-Bold-webfont.woff` from the assets submodule (Clear Sans,
Apache License 2.0). Menu and game-over screens fade in by rendering the overlay
into a second frame that the resolve pass blends over the board.

The `application/` layer is shared in copy with `demo/gol`: models are drawn as
instances in framebuffer pixel coordinates with depth testing, rendered into a
supersampled target, and resolved to the window for anti-aliasing.
