# Game of Life

Conway's Game of Life, ported from the `gol` GUI of
[Yao-class-cpp-studio/Assignment-2](https://github.com/Yao-class-cpp-studio/Assignment-2)
to `grassland::rhi`. The original targeted the legacy grassland Vulkan
wrapper; this version runs on every LongMarch backend (Vulkan, D3D12, Metal).

```sh
cmake --build cmake-build-release --target demo_gol
cmake-build-release/demo/gol/demo_gol                      # 64 x 64 grid
cmake-build-release/demo/gol/demo_gol 60 40 --random 0.3   # random 60 x 40 grid
cmake-build-release/demo/gol/demo_gol 200 200 --pattern demo/gol/patterns/295P5H1V1.cells
cmake-build-release/demo/gol/demo_gol 200 200 --pattern demo/gol/patterns/gosper-glider-gun.cells --play
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
come from empty-space evolution projected modulo 4. Only cells in the same
checkerboard color of 4x4 tiles as the starting tile are shown: cells crossing
one edge disappear, and cells crossing both axes reappear. This is an icon-only
visibility filter; playback performs no simulation and main-grid rules are unchanged.
Switching to fixed plays the recorded 4x4 dead-edge evolution: the glider hits
the upper-right wall and settles into a block. After a brief hold, the icon
restores the lower-left glider. Both sequences play only once per switch.
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
256 x 256 cells; files larger than 1 MiB are rejected.

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
the opposite end. In portrait windows the same groups form a panel along the bottom
edge, with reset and randomize along the top. The controls always sit on the window's short edges, so changing the grid
dimensions never moves them.

The rounded `W` and `H` sliders between randomize and speed adjust the number of
columns and rows from 2 to 256. They fill the space between the buttons: two
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
- Native pinch zoom uses the shared demo window helper on top of the RHI-owned
  GLFW window (macOS). The Cocoa implementation is retained; this migration
  has not been validated on macOS.
- Windows/Linux retain Ctrl + scrolling; driver-emulated pinch only works if
  GLFW can observe both the scrolling and Ctrl key state.
- The controls stay fixed; cells outside the grid viewport are clipped and cannot be clicked.

Options:

- `WIDTH HEIGHT`: cell grid size, each in [2, 256], defaulting to 64 × 64.
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

## RHI rendering and validation

The renderer uses RHI device resources, immutable programs and binding groups,
command-context uploads, the Main queue, and submission/download tokens. Mesh
replacement relies on RHI resource retention; two submission slots bound the
number of frames in flight. The existing supersampling, depth ordering and
second-frame transitions are preserved. PNG export reads the resolved RGBA8
image through an RHI download token.

The demos link `grassland_rhi` and utility dependencies directly, without the
legacy `grassland_graphics` renderer. Window input uses RHI events and GLFW
conveniences in `demo/demo_window.*`.

For reproducible validation, `LONGMARCH_DEMO_SEED=1` fixes 2048's spawn seed;
normal launches still use a time-based seed. `LONGMARCH_DEMO_BENCHMARK=1` reports
frame-loop timings after 120 warmup frames. `LONGMARCH_DEMO_OFFSCREEN=1` skips
presentation while retaining all scene/resolve rendering. These environment
variables are intended for validation, not normal gameplay.

See [migration validation](../../docs/demo-rhi-migration.md) for baseline setup,
image comparisons, replay coverage and performance measurements.
