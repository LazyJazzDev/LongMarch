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
speed (1x, 2x, 5x, lightning), clear the grid, and randomize it with the dice button.
Lightning mode shows a warm yellow bolt and runs generations back-to-back without
an iteration delay. It yields to input and rendering after roughly 8 ms of
computation, so pause, speed changes, and grid editing remain responsive.
The die has six separated rounded faces with cut-out pips. Each click tumbles it
in 3D and lands on a randomly chosen different face, tilted toward the viewer.
Each click independently gives every cell a 50% chance of being alive; the
play/pause state is preserved. The layout follows the window size.

The rounded `W` and `H` sliders between randomize and speed adjust the number of
columns and rows from 2 to 200. They fill the space between the buttons: two
vertical bars in the sidebar, two horizontal bars in the bottom panel. A straight
color boundary marks each value (increasing upward or rightward); centered pixel
labels rotate with the bars. Drag or click anywhere on a bar; after selecting a slider,
use arrow keys for single-cell steps or Home/End for the endpoints. Resizing keeps
cells in the overlapping upper-left region and leaves new cells empty. The view
automatically fits and centers the resized grid. Playback state is preserved.

Grid navigation:

- Two-finger scrolling pans the enlarged grid horizontally and vertically.
- Ctrl + two-finger scrolling or mouse wheel zooms around the pointer (1x–12x).
- Ctrl + `+` / `-` zooms in/out.
- Ctrl + `0` restores the fitted, centered view; Shift + wheel pans horizontally.
- Input uses GLFW on every platform. Native pinch gestures are not handled; pinch
  only zooms if the system or driver translates it into scrolling with a Ctrl
  key state that GLFW can observe.
- The controls stay fixed; cells outside the grid viewport are clipped and cannot be clicked.

Options:

- `WIDTH HEIGHT`: cell grid size, each in [2, 200].
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
