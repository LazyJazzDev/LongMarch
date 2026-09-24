# Game of Life

Conway's Game of Life, ported from the `gol` GUI of
[Yao-class-cpp-studio/Assignment-2](https://github.com/Yao-class-cpp-studio/Assignment-2)
to `grassland::graphics`. The original targeted the legacy grassland Vulkan
wrapper; this version runs on every LongMarch backend (Vulkan, D3D12, Metal).

```sh
cmake --build build --target demo_gol
build/demo/gol/demo_gol                      # 40 x 30 grid
build/demo/gol/demo_gol 60 40 --random 0.3   # random 60 x 40 grid
build/demo/gol/demo_gol 200 200 --pattern demo/gol/patterns/295P5H1V1.cells
build/demo/gol/demo_gol 200 200 --pattern demo/gol/patterns/gosper-glider-gun.cells --play
```

Click cells to toggle them. The buttons play/pause the simulation, cycle the
speed (1x, 2x, 5x, lightning), clear the grid, and randomize it with the dice button.
Lightning mode shows a warm yellow bolt and advances exactly one generation per
frame while playing, without an additional iteration delay. Simulation speed
therefore follows the frame rate, with input and rendering between generations.
The boundary button next to speed toggles between **periodic** (four open portals,
the default) and **fixed/dead** (a solid enclosure). A five-cell glider rests in the lower-left corner of a 4x4 icon grid.
Switching to periodic opens the wall, then plays 16 recorded Life generations
moving diagonally up and right before returning to the initial state. The masks
come from empty-space evolution projected modulo 4; playback performs no simulation.
Cells are full squares with no gaps or clipped halves, sharing the button-wide
lighting gradient instead of shading each cell independently. The icon animation does
not change the main grid or interrupt playback.
The die has six separated rounded faces with cut-out pips. Each click tumbles it
in 3D and lands on a randomly chosen different face, tilted toward the viewer.
Each click independently gives every cell a 50% chance of being alive; the
play/pause state is preserved. The layout follows the window size.

The folder/up-arrow button opens a `.cells` file; the tray/down-arrow button
saves the current grid through **tinyfiledialogs**. Ctrl+O / Ctrl+S (Cmd+O / Cmd+S
on macOS) provide the same actions. Saving preserves the exact grid dimensions,
live cells, and empty borders, including completely empty grids. A `.cells`
extension is added when omitted, with confirmation before replacing a file.
Files are plain text: `O` for alive, `.` for dead, one complete row per line;
lines starting with `!` are comments. Rows must have equal lengths and fit in
200 x 200 cells; files larger than 1 MiB are rejected.

Opening keeps each current grid dimension when it is larger than the file, and
expands dimensions that are too small. The file is centered independently on both
axes; any odd extra cell is placed on the right/bottom. The remaining cells are
cleared, the sliders and fitted view are updated, and playback is paused.
Patterns only one cell wide or high are padded to the minimum grid size of two.
Canceling or failing to open preserves the current grid and playback state.
Saving resumes the previous playback state without advancing through time spent
in the dialog. Neither playback speed nor the current zoom is stored in the file.
Arrows move in the direction of the action; a brief green check confirms success,
and errors show an explanatory dialog followed by a red pulse and a gentle shake.

The toolbar groups file actions above board actions in a two-column block.
Dimension sliders occupy the flexible middle space and playback controls sit at
the opposite end. The same groups rearrange into a bottom panel for wide grids.

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
- Hold the right mouse button and drag from inside the grid viewport to pan.
  Enlarged grid edges can be pulled inward with 5% viewport padding on each side.
- Ctrl + two-finger scrolling or mouse wheel zooms around the pointer (1x–12x).
- Ctrl + `+` / `-` zooms in/out.
- Ctrl + `0` restores the fitted, centered view; Shift + wheel pans horizontally.
- Native pinch zoom uses `graphics::Window::MagnifyEvent()` (currently supported
  on macOS). The game contains no platform-specific gesture code. See the
  [window input API](../../docs/graphics-window-input.md) for event semantics.
- Windows/Linux retain Ctrl + scrolling; driver-emulated pinch only works if
  GLFW can observe both the scrolling and Ctrl key state.
- The controls stay fixed; cells outside the grid viewport are clipped and cannot be clicked.

Options:

- `WIDTH HEIGHT`: cell grid size, each in [2, 200].
- `--random DENSITY`: start with random live cells, e.g. `0.3`.
- `--pattern FILE`: center a Life `.cells` pattern on the grid, initially paused.
- `--play`: start the simulation immediately.
- `--backend auto|vulkan|d3d12|metal`: select the graphics backend.
- `--frames N`: exit after `N` rendered frames.
- `--screenshot FILE`: save the last frame as PNG on exit.

The included [295P5H1V1](https://playgameoflife.com/lexicon/295P5H1V1)
spaceship is a 52 x 52, 295-cell pattern from Stephen A. Silver's Life Lexicon
(CC BY-SA 3.0). On a 200 x 200 grid it starts at cells (74, 74) through
(125, 125). Press play to watch it travel up and left by one cell every five
generations.

The included [Gosper glider gun](https://conwaylife.com/wiki/Gosper_glider_gun)
starts with 36 live cells and emits one glider every 30 generations. Load it
on the 200 x 200 grid with `--play` to watch the stream. Gliders wrap around
the grid edges and can eventually interact with the gun or other gliders.

`game_of_life_lib/` holds `update_step`, the part students implement in the
assignment; this demo defaults to periodic boundaries, with a button to restore the
assignment's fixed/dead boundaries. Left/right and top/bottom edges connect, including diagonal
neighbors across corners. All eight directional offsets count; when an axis is
two cells long, opposite directions count the same cell twice. Iteration still
uses the existing buffer without allocating a second grid.

The `application/` layer replaces the old Vulkan renderer: models are drawn as
instances in framebuffer pixel coordinates with depth testing, rendered into a
supersampled target, and resolved to the window for anti-aliasing (the original
used MSAA, which `grassland::graphics` does not expose).
