# Game of Life

Conway's Game of Life, ported from the `gol` GUI of
[Yao-class-cpp-studio/Assignment-2](https://github.com/Yao-class-cpp-studio/Assignment-2)
to `grassland::graphics`. The original targeted the legacy grassland Vulkan
wrapper; this version runs on every LongMarch backend (Vulkan, D3D12, Metal).

```sh
cmake --build build --target demo_gol
build/demo/gol/demo_gol                      # 40 x 30 grid
build/demo/gol/demo_gol 60 40 --random 0.3   # random 60 x 40 grid
```

Click cells to toggle them. The buttons play/pause the simulation, cycle the
speed (1x, 2x, 5x), clear the grid, and randomize it with the dice button.
Each click independently gives every cell a 50% chance of being alive; the
play/pause state is preserved. The layout follows the window size.

Options:

- `WIDTH HEIGHT`: cell grid size, each in [2, 100].
- `--random DENSITY`: start with random live cells, e.g. `0.3`.
- `--backend auto|vulkan|d3d12|metal`: select the graphics backend.
- `--frames N`: exit after `N` rendered frames.
- `--screenshot FILE`: save the last frame as PNG on exit.

`game_of_life_lib/` holds `update_step`, the part students implement in the
assignment; the demo ships a reference implementation. The dead-boundary rule
matches the assignment's test data.

The `application/` layer replaces the old Vulkan renderer: models are drawn as
instances in framebuffer pixel coordinates with depth testing, rendered into a
supersampled target, and resolved to the window for anti-aliasing (the original
used MSAA, which `grassland::graphics` does not expose).
